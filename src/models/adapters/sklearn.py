from pathlib import Path
from ..base import ModelAdapter


def dump_safe(value, path):
    import skops.io as sio
    sio.dump(value, path)


def load_safe(path):
    import skops.io as sio
    unknown = sio.get_untrusted_types(file=path)
    # Explicitly allow only the additional types used by our built-in encoders/forests.
    allowed = {'numpy.dtype', 'sklearn.tree._tree.Tree'}
    if set(unknown) - allowed:
        raise ValueError(f'Artifact contains unsupported custom types: {unknown}')
    return sio.load(path, trusted=list(allowed))


class SklearnAdapter(ModelAdapter):
    def __init__(self, config, task_type):
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        self.config, self.task_type = config, task_type
        classes = {'dummy': (DummyClassifier, DummyRegressor),
                   'linear': (LogisticRegression, Ridge),
                   'random_forest': (RandomForestClassifier, RandomForestRegressor)}
        if config.algorithm not in classes:
            raise ValueError('Unknown sklearn algorithm')
        classifier = task_type == 'classification'
        defaults = {}
        if config.algorithm == 'linear':
            defaults = {'max_iter': 1000, 'random_state': config.seed} if classifier else {'solver': 'lsqr'}
        elif config.algorithm == 'random_forest':
            defaults = {'n_estimators': 100, 'random_state': config.seed, 'n_jobs': 1}
        elif classifier:
            defaults = {'strategy': 'prior', 'random_state': config.seed}
        self.estimator = classes[config.algorithm][0 if classifier else 1](**{**defaults, **config.params})

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def save(self, directory):
        path = Path(directory)/'model.skops'
        dump_safe(self.estimator, path)
        return path

    def load(self, directory):
        self.estimator = load_safe(Path(directory)/'model.skops')
        return self
