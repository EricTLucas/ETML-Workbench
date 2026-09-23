import json
from pathlib import Path
import numpy as np
from ..base import ModelAdapter


class NeuralAdapter(ModelAdapter):
    """Shared epoch lifecycle; framework subclasses implement optimization and storage."""
    def __init__(self, config, task_type):
        self.config, self.task_type = config, task_type
        defaults = {'hidden_sizes':[64,32], 'epochs':30, 'batch_size':64,
                    'learning_rate':0.001, 'patience':5, 'min_delta':0.0,
                    'optimizer':'adam','regularizer':'none','regularization_strength':0.0001,
                    'loss':'auto','activation':'relu'}
        unknown = set(config.params)-set(defaults)
        if unknown:
            raise ValueError(f'Unsupported MLP settings: {sorted(unknown)}')
        self.options = {**defaults,**config.params}
        if self.options['optimizer'] not in {'adam','adamw','sgd','rmsprop'}:
            raise ValueError('optimizer must be adam, adamw, sgd or rmsprop')
        if self.options['regularizer'] not in {'none','l1','l2'}:
            raise ValueError('regularizer must be none, l1 or l2')
        if self.options['activation'] not in {'relu','tanh','gelu'}:
            raise ValueError('activation must be relu, tanh or gelu')
        loss = self.options['loss']
        allowed = {'auto','cross_entropy'} if task_type=='classification' else {'auto','mse','mae','huber'}
        if loss not in allowed:
            raise ValueError('Loss is incompatible with this task')
        self.loss_name = ('cross_entropy' if task_type=='classification' else 'mse') if loss=='auto' else loss
        for key in ('epochs','batch_size'):
            if type(self.options[key]) is not int or self.options[key]<1:
                raise ValueError(f'{key} must be a positive integer')
        patience = self.options['patience']
        if patience is not None and (type(patience) is not int or patience<1):
            raise ValueError('patience must be positive or null to disable early stopping')
        hidden = self.options['hidden_sizes']
        if not isinstance(hidden,(tuple,list)) or any(type(v) is not int or v<1 for v in hidden):
            raise ValueError('hidden_sizes must be a list of positive integers')
        for key in ('learning_rate','min_delta','regularization_strength'):
            value = self.options[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
                raise ValueError(f'Invalid {key}')
        if self.options['learning_rate']==0:
            raise ValueError('learning_rate must be positive')
        self.max_bytes = 512*1024**2

    def _dense(self, matrix):
        if matrix.shape[0]*matrix.shape[1]*4 > self.max_bytes:
            raise ValueError('Neural batch exceeds dense memory limit; reduce batch_size')
        result = matrix.toarray() if hasattr(matrix,'toarray') else matrix
        return np.asarray(result,dtype=np.float32)

    def _outputs(self,x):
        size = self.options['batch_size']
        width = self.architecture['outputs']
        if len_range := x.shape[0]:
            if len_range*width*4 > self.max_bytes:
                raise ValueError('Neural prediction output exceeds memory limit')
            return np.concatenate([self._forward(self._dense(x[i:i+size]))
                                   for i in range(0,len_range,size)])
        return np.empty((0,width),dtype=np.float32)

    def predict_proba(self,x):
        if self.task_type!='classification':
            raise ValueError('Regression has no class probabilities')
        logits = self._outputs(x).astype(np.float64)
        logits -= logits.max(axis=1,keepdims=True)
        values = np.exp(logits)
        return values/values.sum(axis=1,keepdims=True)

    def predict(self,x):
        return self._outputs(x).argmax(axis=1) if self.task_type=='classification' else self._outputs(x).reshape(-1)

    def _metrics(self,x,y):
        if self.task_type=='classification':
            probability = self.predict_proba(x)
            loss = float(-np.log(np.clip(probability[np.arange(len(y)),y],1e-15,1.)).mean())
            return {'loss':loss,'accuracy':float((probability.argmax(axis=1)==y).mean())}
        error = self.predict(x).astype(np.float64)-y
        mse = float(np.mean(error**2))
        loss = mse if self.loss_name=='mse' else float(np.mean(np.abs(error))) if self.loss_name=='mae' else float(np.mean(np.where(np.abs(error)<=1,0.5*error**2,np.abs(error)-0.5)))
        return {'loss':loss,'rmse':float(np.sqrt(mse))}

    def fit(self,x,y):
        raise ValueError('Neural training requires explicit validation_data')

    def fit_validation(self,x,y,*,validation_data,progress=None,checkpoint=None,
                       resume=None,reset_patience=False,max_bytes=512*1024**2):
        self.max_bytes = max_bytes
        self.architecture = {'inputs':int(x.shape[1]),'outputs':int(np.max(y))+1 if self.task_type=='classification' else 1,
                             'hidden_sizes':list(self.options['hidden_sizes'])}
        dimensions = [self.architecture['inputs'],*self.architecture['hidden_sizes'],self.architecture['outputs']]
        # Weights, gradients, Adam moments and best-state copy require several arrays.
        if sum((a+1)*b for a,b in zip(dimensions,dimensions[1:]))*24 > max_bytes:
            raise ValueError('Neural architecture exceeds the parameter memory budget')
        if resume:
            self._load_training(Path(resume))
            self.state = json.loads((Path(resume)/'state.json').read_text(encoding='utf-8'))
            if self.options['epochs']<=self.state['epoch']:
                raise ValueError('epochs is the total target and must exceed the checkpoint epoch')
            if self.state['stopped_early'] and not reset_patience:
                raise ValueError('Checkpoint reached early stopping; use --reset-patience to explicitly continue')
            if reset_patience:
                self.state['bad_epochs']=0
                self.state['stopped_early']=False
        else:
            self._build()
            self.state = {'epoch':0,'best_epoch':0,'best_loss':None,'bad_epochs':0,
                          'stopped_early':False,'history':[]}
        size = self.options['batch_size']
        for epoch in range(self.state['epoch']+1,self.options['epochs']+1):
            order = np.random.default_rng(np.random.SeedSequence([self.config.seed,epoch])).permutation(len(y))
            for start in range(0,len(y),size):
                indices = order[start:start+size]
                self._train_batch(self._dense(x[indices]),y[indices])
            train_metrics = self._metrics(x,y)
            val_metrics = self._metrics(*validation_data) if validation_data is not None else {}
            if not all(np.isfinite(v) for v in [*train_metrics.values(),*val_metrics.values()]):
                raise ValueError('Training diverged: nonfinite epoch metrics')
            row = {'epoch':epoch,**{'train_'+k:v for k,v in train_metrics.items()},
                   **{'validation_'+k:v for k,v in val_metrics.items()}}
            self.state['history'].append(row)
            best = self.state['best_loss']
            if validation_data is None or best is None or val_metrics['loss'] < best-self.options['min_delta']:
                self.state.update(best_loss=val_metrics.get('loss'),best_epoch=epoch,bad_epochs=0)
                self._snapshot_best()
            else:
                self.state['bad_epochs']+=1
            self.state['epoch']=epoch
            patience = self.options['patience'] if validation_data is not None else None
            self.state['stopped_early'] = patience is not None and self.state['bad_epochs']>=patience
            saved = checkpoint(self,self.state) if checkpoint else None
            if progress:
                progress({'stage':'epoch',**row,'checkpoint':str(saved) if saved else None})
            if self.state['stopped_early']:
                break
        self._restore_best()
        self.history = self.state['history']
        self.training_summary = {k:v for k,v in self.state.items() if k!='history'}
        self.training_summary['selection']='best validation loss' if validation_data is not None else 'final epoch; no validation/early stopping'
        return self

    def save(self,directory):
        from data.manifest import write_json
        write_json(Path(directory)/'architecture.json',self.architecture)
        return self._save_model(Path(directory))

    def load(self,directory):
        directory = Path(directory)
        self.architecture = json.loads((directory/'architecture.json').read_text(encoding='utf-8'))
        self._load_model(directory)
        return self
