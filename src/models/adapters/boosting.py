"""Optional native-format boosting adapters over the shared tabular encoder."""
from pathlib import Path
import numpy as np
from ..base import ModelAdapter


class LightGBMAdapter(ModelAdapter):
    def __init__(self,config,task_type):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('Install LightGBM: python -m pip install -e ".[lightgbm]"') from exc
        self.lgb,self.config,self.task_type=lgb,config,task_type
        params=dict(config.params)
        self.patience=params.pop('early_stopping_rounds',None)
        if self.patience is not None and (type(self.patience) is not int or self.patience<1):
            raise ValueError('early_stopping_rounds must be positive')
        if 'callbacks' in params:
            raise ValueError('Custom callbacks are not supported')
        cls=lgb.LGBMClassifier if task_type=='classification' else lgb.LGBMRegressor
        self.estimator=cls(**{'n_estimators':100,'learning_rate':.1,'num_leaves':31,
            'random_state':config.seed,'n_jobs':1,'verbosity':-1,**params})
        self.booster=None

    def fit(self,x,y):
        self.estimator.fit(x,y)
        self.booster=self.estimator.booster_
        return self

    def fit_validation(self,x,y,*,validation_data,progress=None,checkpoint=None,resume=None,
                       reset_patience=False,max_bytes=512*1024**2):
        if resume: raise ValueError('LightGBM checkpoint resume is not supported')
        callbacks=[self.lgb.log_evaluation(0)]
        if self.patience: callbacks.append(self.lgb.early_stopping(self.patience,verbose=False))
        self.estimator.fit(x,y,eval_set=[validation_data],callbacks=callbacks)
        self.booster=self.estimator.booster_
        self.training_summary={'boosting_rounds':self.booster.current_iteration(),
                               'best_iteration':self.booster.best_iteration}
        return self

    def predict_proba(self,x):
        values=np.asarray(self.booster.predict(x,num_threads=1))
        return np.column_stack([1-values,values]) if values.ndim==1 else values

    def predict(self,x):
        return self.predict_proba(x).argmax(axis=1) if self.task_type=='classification' else self.booster.predict(x,num_threads=1)

    def save(self,directory):
        path=Path(directory)/'model.lightgbm.txt'
        self.booster.save_model(str(path))
        return path

    def load(self,directory):
        self.booster=self.lgb.Booster(model_file=str(Path(directory)/'model.lightgbm.txt'))
        return self


class CatBoostAdapter(ModelAdapter):
    def __init__(self,config,task_type):
        try:
            from catboost import CatBoostClassifier,CatBoostRegressor
        except ImportError as exc:
            raise ImportError('Install CatBoost: python -m pip install -e ".[catboost]"') from exc
        self.config,self.task_type=config,task_type
        params=dict(config.params)
        self.patience=params.pop('early_stopping_rounds',None)
        if self.patience is not None and (type(self.patience) is not int or self.patience<1):
            raise ValueError('early_stopping_rounds must be positive')
        # Workbench owns artifact locations; the backend must not create unrelated directories.
        for key in ('train_dir','snapshot_file','save_snapshot','allow_writing_files'):
            if key in params: raise ValueError(f'Workbench manages {key}; omit it')
        cls=CatBoostClassifier if task_type=='classification' else CatBoostRegressor
        self.estimator=cls(**{'iterations':100,'depth':6,'learning_rate':.1,'random_seed':config.seed,
                             'thread_count':1,'verbose':False,'allow_writing_files':False,**params})

    def fit(self,x,y):
        self.estimator.fit(x,y,verbose=False)
        return self

    def fit_validation(self,x,y,*,validation_data,progress=None,checkpoint=None,resume=None,
                       reset_patience=False,max_bytes=512*1024**2):
        if resume: raise ValueError('CatBoost checkpoint resume is not supported')
        self.estimator.fit(x,y,eval_set=validation_data,early_stopping_rounds=self.patience,verbose=False)
        self.training_summary={'boosting_rounds':self.estimator.tree_count_,
                               'best_iteration':self.estimator.get_best_iteration()}
        return self

    def predict(self,x):
        return np.asarray(self.estimator.predict(x)).reshape(-1)

    def predict_proba(self,x):
        return self.estimator.predict_proba(x)

    def save(self,directory):
        path=Path(directory)/'model.cbm'
        self.estimator.save_model(str(path))
        return path

    def load(self,directory):
        self.estimator.load_model(str(Path(directory)/'model.cbm'))
        return self
