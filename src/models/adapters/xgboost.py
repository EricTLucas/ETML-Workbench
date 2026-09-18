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
        if 'early_stopping_rounds' in config.params or 'callbacks' in config.params:
            raise ValueError('Early stopping/callbacks are not configured by this adapter yet')
        self.task_type = task_type
        cls = XGBClassifier if task_type == 'classification' else XGBRegressor
        self.estimator = cls(**{'n_estimators':100, 'max_depth':6, 'tree_method':'hist',
                               'random_state':config.seed, 'n_jobs':1, **config.params})

    def fit(self, x, y):
        self.estimator.fit(x, y)
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
