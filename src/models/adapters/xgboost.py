from pathlib import Path
from ..base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    def __init__(self, config, task_type):
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError('Install the xgboost extra: pip install -e ".[xgboost]"') from exc
        if config.algorithm != 'boosted_trees':
            raise ValueError('XGBoost currently supports boosted_trees')
        if 'callbacks' in config.params:
            raise ValueError('Use early_stopping_rounds instead of custom callbacks')
        patience = config.params.get('early_stopping_rounds')
        if patience is not None and (type(patience) is not int or patience < 1):
            raise ValueError('early_stopping_rounds must be a positive integer')
        self.task_type = task_type
        cls = XGBClassifier if task_type == 'classification' else XGBRegressor
        self.estimator = cls(**{'n_estimators':100, 'max_depth':6, 'tree_method':'hist',
                               'random_state':config.seed, 'n_jobs':1, **config.params})

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def fit_validation(self, x, y, *, validation_data, progress=None, checkpoint=None,
                       resume=None, reset_patience=False, max_bytes=512*1024**2):
        if resume is not None:
            raise ValueError('Checkpoint resumption currently supports neural models only')
        self.estimator.fit(x, y, eval_set=[(x,y),validation_data], verbose=False)
        self.history = self.estimator.evals_result()
        self.training_summary = {'monitor':'validation (last eval_set)',
            'best_iteration':getattr(self.estimator,'best_iteration',None),
            'boosting_rounds':self.estimator.get_booster().num_boosted_rounds()}
        if progress:
            progress({'stage':'boosting_complete',**self.training_summary})
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def save(self, directory):
        path = Path(directory)/'model.ubj'
        self.estimator.save_model(path)
        return path

    def load(self, directory):
        self.estimator.load_model(Path(directory)/'model.ubj')
        return self
