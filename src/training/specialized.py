"""Training for image, language, forecasting and interaction datasets.

Specialized bundles own their input contract; they do not masquerade as a
tabular Predictor. Test data is retained in the input snapshot, never selected on.
"""
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from data.manifest import write_json,sha256_file
from data.model_inputs import load_model_input
from models.catalog import CATALOG
from .artifacts import staged_directory, seal, environment, verify_artifacts


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _classes(series):
    # Preserve JSON scalar types and distinguish e.g. numeric 1 from string '1'.
    values=[v.item() if hasattr(v,'item') else v for v in series.unique()]
    return sorted(values,key=lambda v:json.dumps(v,sort_keys=True))


def _labels(series,classes):
    mapping={json.dumps(v):i for i,v in enumerate(classes)}
    try: return np.array([mapping[json.dumps(v.item() if hasattr(v,'item') else v)] for v in series],dtype=np.int64)
    except KeyError as exc: raise ValueError('Evaluation label was not present in training') from exc


def _classification(y,pred,prob=None):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss
    result={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'f1_macro':float(f1_score(y,pred,average='macro',zero_division=0))}
    if prob is not None: result['log_loss']=float(log_loss(y,prob,labels=np.arange(prob.shape[1])))
    return result


def _regression(y,pred):
    return {'rmse':float(np.sqrt(np.mean((np.asarray(y)-pred)**2))), 'mae':float(np.mean(np.abs(np.asarray(y)-pred)))}


def train_specialized(input_path,key,destination,*,params=None,seed=42,max_bytes=512*1024**2,progress=None):
    entry=CATALOG[key]
    root,info=load_model_input(input_path)
    if entry.modality=='tabular' or entry.modality!=info['modality']:
        raise ValueError('The selected model requires a different input modality')
    task=info.get('task') or entry.tasks[0]
    if task not in entry.tasks: raise ValueError('Model and input task do not match')
    options={**entry.defaults,**(params or {})}
    train=pd.read_parquet(root/'splits/train.parquet'); val=pd.read_parquet(root/'splits/validation.parquet')
    if sum(f.memory_usage(deep=True).sum() for f in (train,val))>max_bytes:
        raise ValueError('Input tables exceed the training memory budget')
    context={'model':key,'options':options,'seed':seed,'input':str(root),'input_metadata':info,
             'input_manifest_sha256':sha256_file(root/'manifest.json'),'task':task,'test_used_for_selection':False}
    started=time.perf_counter()
    with staged_directory(destination) as output:
        if entry.modality=='recommendation':
            result=_fit_recommendation(train,val,info['columns'],key,options,seed,output,max_bytes)
        elif key=='statsmodels:arima':
            result=_fit_arima(train,val,info['columns'],options,output)
        else:
            result=_fit_torch(root,train,val,context,output,max_bytes,progress)
        result['fit_seconds']=time.perf_counter()-started
        if any(isinstance(v,float) and not np.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite model results; adjust model settings/data')
        write_json(output/'config.json',context)
        write_json(output/'metrics.json',result)
        write_json(output/'environment.json',environment())
        (output/'SETUP.md').write_text(
            '# ETML specialized model\n\nInstall this ETML version and the optional '+(entry.extra or 'base')+
            ' dependencies. See environment.json for installed versions. config.json records all settings and the input snapshot. '
            'Keep the hashed input snapshot to retrain. Library/hardware changes can affect numerical reproducibility.\n\n'
            'Predict with `workbench models specialized-predict --bundle PATH --values JSON`. '
            'Export with `workbench models specialized-export --bundle PATH --output NEW_PATH`. '
            'Test splits are not used for model selection. See metrics.json for the evaluation protocol.\n',encoding='utf-8')
        seal(output,'specialized_model',{'model':key,'input_manifest':str(root/'manifest.json')})
    return {'directory':str(Path(destination).resolve()),**result}


def _fit_arima(train,val,columns,options,output):
    from statsmodels.tsa.arima.model import ARIMA
    if set(options)-{'order'}: raise ValueError('ARIMA supports order=[p,d,q]')
    order=options['order']
    if len(order)!=3 or any(type(v) is not int or v<0 for v in order): raise ValueError('ARIMA order must contain three nonnegative integers')
    y=train[columns['value']].to_numpy(float); actual=val[columns['value']].to_numpy(float)
    fitted=ARIMA(y,order=tuple(order)).fit()
    pred=np.asarray(fitted.forecast(len(actual)))
    np.savez(output/'arima.npz',training=y,parameters=np.asarray(fitted.params))
    pd.DataFrame({'actual':actual,'prediction':pred}).to_csv(output/'validation_predictions.csv',index=False)
    return {'validation':_regression(actual,pred),'baseline':_regression(actual,np.full(len(actual),y[-1])),
            'protocol':'Fixed-origin multi-step forecast from the end of training',
            'converged':bool(fitted.mle_retvals.get('converged',True))}


def _fit_recommendation(train,val,columns,key,options,seed,output,max_bytes):
    from scipy.sparse import csr_matrix
    users=sorted(train[columns['user']].unique()); items=sorted(train[columns['item']].unique())
    um={v:i for i,v in enumerate(users)}; im={v:i for i,v in enumerate(items)}
    rows=train[columns['user']].map(um).to_numpy(); cols=train[columns['item']].map(im).to_numpy()
    weight=train[columns['weight']].to_numpy(float) if columns.get('weight') else np.ones(len(train))
    factors=options['factors']
    if type(factors) is not int or not 1<=factors<min(len(users),len(items)):
        raise ValueError('factors must be positive and smaller than both the training user and item counts')
    if (len(users)+len(items))*factors*16>max_bytes: raise ValueError('Factor matrices exceed memory budget')
    matrix=csr_matrix((weight,(rows,cols)),shape=(len(users),len(items)),dtype=np.float32)
    if key=='scipy:svd':
        if set(options)-{'factors'}: raise ValueError('SVD supports factors')
        from scipy.sparse.linalg import svds
        u,s,v=svds(matrix,k=factors,random_state=seed)
        user_factors=u*s; item_factors=v.T
    else:
        if set(options)-{'factors','iterations','regularization'}: raise ValueError('ALS supports factors, iterations, regularization')
        if type(options['iterations']) is not int or options['iterations']<1 or options['regularization']<0:
            raise ValueError('Invalid ALS iterations or regularization')
        from implicit.cpu.als import AlternatingLeastSquares
        model=AlternatingLeastSquares(factors=factors,iterations=options['iterations'],regularization=options['regularization'],
                                      random_state=seed,num_threads=1)
        model.fit(matrix,show_progress=False)
        user_factors=model.user_factors; item_factors=model.item_factors
    popularity=np.asarray(matrix.sum(axis=0)).reshape(-1)
    np.savez(output/'factors.npz',users=user_factors,items=item_factors,seen_rows=rows,seen_columns=cols,popularity=popularity)
    write_json(output/'identities.json',{'users':users,'items':items})
    k=min(10,len(items)); scores=[]; baseline=[]; cold_users=0; cold_items=0
    for user,group in val.groupby(columns['user']):
        if user not in um: cold_users+=len(group); continue
        relevant={im[item] for item in group[columns['item']] if item in im}
        cold_items+=sum(item not in im for item in group[columns['item']])
        if not relevant: continue
        idx=um[user]; seen=matrix[idx].indices
        ranking=user_factors[idx] @ item_factors.T; ranking[seen]=-np.inf
        popular=popularity.copy(); popular[seen]=-np.inf
        available=len(items)-len(seen); count=min(k,available)
        if not count: continue
        top=np.argsort(-ranking,kind='stable')[:count]; pop=np.argsort(-popular,kind='stable')[:count]
        scores.append(len(set(top)&relevant)/len(relevant)); baseline.append(len(set(pop)&relevant)/len(relevant))
    return {'validation':{'recall_at_10':float(np.mean(scores)) if scores else None,'evaluated_users':len(scores),
                           'cold_user_rows':cold_users,'cold_item_rows':cold_items},
            'baseline':{'popularity_recall_at_10':float(np.mean(baseline)) if baseline else None},
            'protocol':'Held-out positive pairs; training-seen items excluded; cold-start rows reported separately'}


def _torch_model(context,extra):
    import torch
    key=context['model']; o=context['options']
    torch.manual_seed(context['seed'])
    if key.startswith('torchvision:'):
        algorithm=key.split(':')[1]; outputs=len(extra['classes'])
        if algorithm=='lenet':
            if o.get('pretrained'): raise ValueError('LeNet-style CNN has no pretrained weights')
            model=torch.nn.Sequential(torch.nn.Conv2d(3,6,5),torch.nn.Tanh(),torch.nn.AvgPool2d(2),
                torch.nn.Conv2d(6,16,5),torch.nn.Tanh(),torch.nn.AvgPool2d(2),torch.nn.Flatten(),
                torch.nn.Linear(16*5*5,120),torch.nn.Tanh(),torch.nn.Linear(120,84),torch.nn.Tanh(),torch.nn.Linear(84,outputs))
        else:
            from torchvision import models
            # Saved bundles never need to download weights again.
            model=models.get_model(algorithm,weights='DEFAULT' if o.get('pretrained') and not extra.get('loading') else None)
            if algorithm=='resnet18': model.fc=torch.nn.Linear(model.fc.in_features,outputs)
            elif algorithm=='vit_b_16': model.heads.head=torch.nn.Linear(model.heads.head.in_features,outputs)
            else: model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,outputs)
        return model,None
    if key in {'pytorch:lstm','pytorch:gru'}:
        class Recurrent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cls=torch.nn.LSTM if key.endswith('lstm') else torch.nn.GRU
                self.rnn=cls(1,o['hidden_size'],num_layers=o['num_layers'],batch_first=True)
                self.head=torch.nn.Linear(o['hidden_size'],1)
            def forward(self,x):
                values,_=self.rnn(x)
                return self.head(values[:,-1]).reshape(-1)
        return Recurrent(),None
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModelForCausalLM,AutoModelForSeq2SeqLM
    checkpoint=extra.get('saved_path') or o['checkpoint']
    kw={'trust_remote_code':False}
    if not extra.get('saved_path'): kw['revision']=o['revision']
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,**kw)
    cls={'transformers:bert':AutoModelForSequenceClassification,'transformers:gpt':AutoModelForCausalLM,
         'transformers:t5':AutoModelForSeq2SeqLM}[key]
    if key.endswith('bert') and not extra.get('saved_path'):
        kw.update(num_labels=len(extra['classes']),ignore_mismatched_sizes=True)
    model=cls.from_pretrained(checkpoint,use_safetensors=True,**kw)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model.config.pad_token_id=tokenizer.pad_token_id
    return model,tokenizer


def _image_tensor(path,key):
    from PIL import Image,ImageOps
    from torchvision.transforms import functional as F
    size=32 if key.endswith('lenet') else 224
    with Image.open(path) as original:
        image=ImageOps.fit(ImageOps.exif_transpose(original).convert('RGB'),(size,size))
        value=F.to_tensor(image)
    return F.normalize(value,[.485,.456,.406],[.229,.224,.225])


def _neural_options(context):
    o=context['options']; key=context['model']; entry=CATALOG[key]
    allowed=set(entry.defaults)|{'device','patience','min_delta'}
    if set(o)-allowed: raise ValueError('Unknown neural parameters: '+str(sorted(set(o)-allowed)))
    for name in ('epochs','batch_size'):
        if type(o[name]) is not int or o[name]<1: raise ValueError(name+' must be a positive integer')
    if not np.isfinite(o['learning_rate']) or o['learning_rate']<=0: raise ValueError('learning_rate must be positive')
    if o['optimizer'] not in {'adam','adamw','sgd','rmsprop'}: raise ValueError('Unknown optimizer')
    if o['regularizer'] not in {'none','l1','l2'}: raise ValueError('regularizer must be none, l1 or l2')
    if not np.isfinite(o['regularization_strength']) or o['regularization_strength']<0: raise ValueError('Invalid regularization strength')
    expected='mse' if context['task']=='forecasting' else 'cross_entropy'
    choices={'auto',expected}|({'mae','huber'} if expected=='mse' else set())
    if o['loss'] not in choices: raise ValueError('Loss is incompatible with the task: '+str(sorted(choices)))
    if o.get('patience',5) is not None and (type(o.get('patience',5)) is not int or o.get('patience',5)<1): raise ValueError('Invalid patience')
    if not np.isfinite(o.get('min_delta',0)) or o.get('min_delta',0)<0: raise ValueError('Invalid min_delta')
    return expected if o['loss']=='auto' else o['loss']


def _fit_torch(root,train,val,context,output,max_bytes,progress):
    import torch
    o=context['options']; key=context['model']; columns=context['input_metadata']['columns']; task=context['task']
    loss_name=_neural_options(context)
    device=o.get('device','cpu')
    if device not in {'cpu','cuda'}: raise ValueError('device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is not available')
    # Adam training also stores gradients/moments; catch the largest architectures before allocation.
    minimum={'torchvision:alexnet':1200,'torchvision:vgg16':2400,'torchvision:vit_b_16':1600,
             'torchvision:resnet18':250,'transformers:bert':2000,'transformers:gpt':2400,'transformers:t5':1300}
    if minimum.get(key,0)*1024**2>max_bytes:
        raise ValueError(f'{key} requires a larger model memory budget; raise --max-memory-mb to at least {minimum[key]} (batch activations need additional RAM)')
    extra={}; y_train=y_val=None
    if task=='classification':
        if 'target' not in columns: raise ValueError('Classification requires a target column')
        extra['classes']=_classes(train[columns['target']])
        if len(extra['classes'])<2: raise ValueError('Classification requires at least two training classes')
        y_train=_labels(train[columns['target']],extra['classes']); y_val=_labels(val[columns['target']],extra['classes'])
    if key.endswith('t5') and 'target' not in columns: raise ValueError('T5 requires target text')
    if key in {'pytorch:lstm','pytorch:gru'}:
        for name in ('window','hidden_size','num_layers'):
            if type(o[name]) is not int or o[name]<1: raise ValueError(name+' must be a positive integer')
        raw=train[columns['value']].to_numpy(np.float32); actual=val[columns['value']].to_numpy(np.float32)
        if len(raw)<=o['window']: raise ValueError('Training series must be longer than window')
        extra.update(mean=float(raw.mean()),std=float(raw.std()) or 1.,tail=raw[-o['window']:].tolist())
        series=np.concatenate([raw,actual]); series=(series-extra['mean'])/extra['std']
        train_positions=np.arange(o['window'],len(raw)); val_positions=np.arange(len(raw),len(series))
        y_train=raw[o['window']:]; y_val=actual
    if key.startswith('transformers:'):
        if type(o['max_length']) is not int or o['max_length']<2: raise ValueError('max_length must be at least 2')
    model,tokenizer=_torch_model(context,extra)
    if sum(p.numel()*p.element_size() for p in model.parameters())*5>max_bytes:
        raise ValueError('Model weights/gradients/optimizer exceed memory budget')
    model.to(device)
    optimizer={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW,'sgd':torch.optim.SGD,'rmsprop':torch.optim.RMSprop}[o['optimizer']]
    opt=optimizer(model.parameters(),lr=o['learning_rate'],**({'weight_decay':0.0} if o['optimizer']=='adamw' else {}))
    size=o['batch_size']; history=[]; best=math.inf; bad=0
    def batch(frame,indices,training):
        if key.startswith('torchvision:'):
            x=torch.stack([_image_tensor(root/str(frame.iloc[i][columns['image']]),key) for i in indices]).to(device)
            labels=torch.tensor((y_train if training else y_val)[indices],device=device)
            result=model(x); return torch.nn.functional.cross_entropy(result,labels),result,len(indices)
        if key in {'pytorch:lstm','pytorch:gru'}:
            positions=(train_positions if training else val_positions)[indices]
            x=np.stack([series[p-o['window']:p] for p in positions]).astype(np.float32)[...,None]
            result=model(torch.from_numpy(x).to(device)); labels=torch.tensor(series[positions],device=device)
            fn={'mse':torch.nn.functional.mse_loss,'mae':torch.nn.functional.l1_loss,'huber':torch.nn.functional.huber_loss}[loss_name]
            return fn(result,labels),result,len(indices)
        texts=frame.iloc[indices][columns['text']].astype(str).tolist()
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=o['max_length'],return_tensors='pt')
        if key.endswith('bert'):
            tokens['labels']=torch.tensor((y_train if training else y_val)[indices])
            count=len(indices)
        elif key.endswith('gpt'):
            tokens['labels']=tokens['input_ids'].clone(); tokens['labels'][tokens['attention_mask']==0]=-100
            count=int((tokens['labels'][:,1:]!=-100).sum())
            if count==0: raise ValueError('GPT examples need at least two tokens after tokenization')
        else:
            targets=tokenizer(text_target=frame.iloc[indices][columns['target']].astype(str).tolist(),padding=True,
                              truncation=True,max_length=o['max_length'],return_tensors='pt')
            labels=targets['input_ids']; labels[targets['attention_mask']==0]=-100
            tokens['labels']=labels; count=int((labels!=-100).sum())
        out=model(**{k:v.to(device) for k,v in tokens.items()})
        return out.loss,out.logits if key.endswith('bert') else None,count
    n_train=len(y_train) if key in {'pytorch:lstm','pytorch:gru'} else len(train)
    for epoch in range(1,o['epochs']+1):
        model.train(); train_loss=0.; train_count=0
        order=np.random.default_rng(np.random.SeedSequence([context['seed'],epoch])).permutation(n_train)
        for start in range(0,n_train,size):
            indices=order[start:start+size]; opt.zero_grad()
            loss,_,count=batch(train,indices,True)
            train_loss+=float(loss.detach())*count; train_count+=count
            if o['regularizer']!='none':
                penalty=sum(p.abs().sum() if o['regularizer']=='l1' else p.square().sum() for p in model.parameters())
                loss=loss+o['regularization_strength']*penalty
            if not torch.isfinite(loss): raise ValueError('Training diverged; lower learning rate/check data')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); val_loss=0.; val_count=0
        with torch.no_grad():
            for start in range(0,len(val),size):
                loss,_,count=batch(val,np.arange(start,min(start+size,len(val))),False)
                val_loss+=float(loss)*count; val_count+=count
        row={'epoch':epoch,'train_loss':train_loss/train_count,'validation_loss':val_loss/val_count}
        if not np.isfinite(row['validation_loss']): raise ValueError('Validation loss is nonfinite')
        history.append(row)
        if row['validation_loss']<best-o.get('min_delta',0.):
            best=row['validation_loss']; bad=0; extra['best_epoch']=epoch
            torch.save(model.state_dict(),output/'model.pt')
        else: bad+=1
        if progress: progress({'stage':'epoch',**row})
        if o.get('patience',5) is not None and bad>=o.get('patience',5): break
    model.load_state_dict(torch.load(output/'model.pt',map_location=device,weights_only=True)); model.eval()
    write_json(output/'history.json',history); write_json(output/'model_details.json',extra)
    if tokenizer:
        model.save_pretrained(output/'huggingface',safe_serialization=True); tokenizer.save_pretrained(output/'huggingface')
        (output/'model.pt').unlink()
    predictions=[]
    if task in {'classification','forecasting'}:
        with torch.no_grad():
            for start in range(0,len(val),size):
                _,logits,_=batch(val,np.arange(start,min(start+size,len(val))),False)
                predictions.append((torch.softmax(logits,dim=1) if task=='classification' else logits).cpu().numpy())
        pred=np.concatenate(predictions)
        if task=='classification':
            metrics=_classification(y_val,pred.argmax(axis=1),pred)
            baseline=_classification(y_val,np.full(len(y_val),np.bincount(y_train).argmax()))
            pd.DataFrame({'actual':y_val,'prediction':pred.argmax(axis=1)}).to_csv(output/'validation_predictions.csv',index=False)
        else:
            pred=pred*extra['std']+extra['mean']; metrics=_regression(y_val,pred)
            baseline=_regression(y_val,np.concatenate([raw[-1:],actual[:-1]]))
            pd.DataFrame({'actual':y_val,'prediction':pred}).to_csv(output/'validation_predictions.csv',index=False)
    else:
        metrics={'token_cross_entropy':best}
        if task=='language_modeling': metrics['perplexity']=math.exp(min(best,700))
        baseline=None
    return {'validation':metrics,'baseline':baseline,'best_epoch':extra['best_epoch'],'epochs_completed':len(history),
            'protocol':'Rolling one-step forecasts with observed history' if task=='forecasting' else 'Held-out validation; best validation loss selects epoch'}


def predict_specialized(bundle,values):
    """JSON input: image path; text; history/horizon; or user and k."""
    root=Path(bundle).resolve(); verify_artifacts(root,'specialized_model')
    context=_json(root/'config.json'); key=context['model']; o=context['options']
    if key=='statsmodels:arima':
        from statsmodels.tsa.arima.model import ARIMA
        horizon=values.get('horizon',1)
        if type(horizon) is not int or not 1<=horizon<=10000: raise ValueError('horizon must be 1..10000')
        with np.load(root/'arima.npz',allow_pickle=False) as arrays:
            model=ARIMA(arrays['training'],order=tuple(o['order'])).filter(arrays['parameters'])
        forecast=model.get_forecast(horizon)
        return {'forecast':np.asarray(forecast.predicted_mean).tolist(),'confidence_interval_95':np.asarray(forecast.conf_int()).tolist()}
    if context['task']=='recommendation':
        identities=_json(root/'identities.json'); user=str(values['user']); k=values.get('k',10)
        if type(k) is not int or not 1<=k<=1000: raise ValueError('k must be 1..1000')
        with np.load(root/'factors.npz',allow_pickle=False) as a:
            if user not in identities['users']:
                scores=a['popularity'].copy(); fallback=True
            else:
                idx=identities['users'].index(user); scores=a['users'][idx]@a['items'].T
                seen=a['seen_columns'][a['seen_rows']==idx]; scores[seen]=-np.inf; fallback=False
            top=[i for i in np.argsort(-scores,kind='stable') if np.isfinite(scores[i])][:k]
            return {'recommendations':[{'item':identities['items'][i],'score':float(scores[i])} for i in top],
                    'popularity_fallback':fallback}
    import torch
    extra=_json(root/'model_details.json'); extra.update(loading=True)
    if key.startswith('transformers:'): extra['saved_path']=str(root/'huggingface')
    model,tokenizer=_torch_model(context,extra)
    if tokenizer is None: model.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True))
    model.eval()
    with torch.no_grad():
        if key.startswith('torchvision:'):
            logits=model(_image_tensor(Path(values['image']),key).unsqueeze(0))
        elif key in {'pytorch:lstm','pytorch:gru'}:
            history=np.asarray(values.get('history',extra['tail']),dtype=np.float32)
            if history.ndim!=1 or len(history)<o['window'] or not np.isfinite(history).all():
                raise ValueError('history must contain at least window finite numeric values')
            horizon=values.get('horizon',1)
            if type(horizon) is not int or not 1<=horizon<=1000: raise ValueError('horizon must be 1..1000')
            predictions=[]
            for _ in range(horizon):
                x=(history[-o['window']:]-extra['mean'])/extra['std']
                value=float(model(torch.tensor(x).reshape(1,-1,1))[0])*extra['std']+extra['mean']
                predictions.append(value); history=np.append(history,value).astype(np.float32)
            return {'forecast':predictions,'protocol':'Recursive forecast from supplied history (training tail by default)'}
        else:
            tokens=tokenizer(str(values['text']),truncation=True,max_length=o['max_length'],return_tensors='pt')
            if key.endswith('bert'): logits=model(**tokens).logits
            else:
                n=values.get('max_new_tokens',32)
                if type(n) is not int or not 1<=n<=512: raise ValueError('max_new_tokens must be 1..512')
                generated=model.generate(**tokens,max_new_tokens=n,do_sample=False,pad_token_id=tokenizer.pad_token_id)
                return {'text':tokenizer.decode(generated[0],skip_special_tokens=True)}
        proba=torch.softmax(logits,dim=1)[0].numpy()
        return {'prediction':extra['classes'][int(proba.argmax())],
                'probabilities':[{'class':c,'probability':float(p)} for c,p in zip(extra['classes'],proba)]}


def export_specialized(bundle,destination):
    root=Path(bundle); verify_artifacts(root,'specialized_model')
    with staged_directory(destination) as out:
        shutil.copytree(root,out,dirs_exist_ok=True)
    return Path(destination).resolve()
