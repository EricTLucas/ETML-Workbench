from pathlib import Path
from ..base import ModelAdapter


def dump_safe(value, path):
    import skops.io as sio
    sio.dump(value, path)


def load_safe(path):
    import skops.io as sio
    unknown = sio.get_untrusted_types(file=path)
    # Explicitly allow only the additional types used by our built-in encoders/forests.
    allowed = {'numpy.dtype', 'sklearn.tree._tree.Tree', 'scipy.sparse._csr.csr_matrix',
               'sklearn.utils._bunch.Bunch'}
    if set(unknown) - allowed:
        raise ValueError(f'Artifact contains unsupported custom types: {unknown}')
    return sio.load(path, trusted=list(allowed))


class SklearnAdapter(ModelAdapter):
    def __init__(self, config, task_type):
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
            AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor,
            StackingClassifier, StackingRegressor)
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
        self.config, self.task_type = config, task_type
        classes = {'dummy': (DummyClassifier, DummyRegressor),
                   'linear': (LogisticRegression, Ridge),
                   'linear_regression':(None,LinearRegression),
                   'ridge':(None,Ridge),'lasso':(None,Lasso),'elastic_net':(None,ElasticNet),
                   'logistic_regression':(LogisticRegression,None),
                   'multinomial_logistic':(LogisticRegression,None),
                   'decision_tree':(DecisionTreeClassifier,DecisionTreeRegressor),
                   'random_forest': (RandomForestClassifier, RandomForestRegressor),
                   'extra_trees':(ExtraTreesClassifier,ExtraTreesRegressor),
                   'gradient_boosting':(GradientBoostingClassifier,GradientBoostingRegressor),
                   'adaboost':(AdaBoostClassifier,AdaBoostRegressor),
                   'svm':(SVC,SVR),'knn':(KNeighborsClassifier,KNeighborsRegressor),
                   'gaussian_nb':(GaussianNB,None),'multinomial_nb':(MultinomialNB,None),
                   'bernoulli_nb':(BernoulliNB,None),
                   'voting':(VotingClassifier,VotingRegressor),'stacking':(StackingClassifier,StackingRegressor)}
        if config.algorithm not in classes:
            raise ValueError('Unknown sklearn algorithm')
        classifier = task_type == 'classification'
        defaults = {}
        if config.algorithm in {'linear','logistic_regression','multinomial_logistic','ridge'}:
            defaults = {'max_iter': 1000, 'random_state': config.seed} if classifier else {'solver': 'lsqr'}
        elif config.algorithm in {'random_forest','extra_trees'}:
            defaults = {'n_estimators': 100, 'random_state': config.seed, 'n_jobs': 1}
        elif config.algorithm in {'decision_tree','gradient_boosting','adaboost'}:
            defaults = {'random_state':config.seed}
        elif config.algorithm=='svm' and classifier:
            defaults={'probability':True,'random_state':config.seed}
            if config.params.get('probability',True) is not True:
                raise ValueError('Workbench SVM classification requires probability=True')
        elif config.algorithm=='knn':
            defaults={'n_neighbors':5,'n_jobs':1,'algorithm':'brute'}
        elif config.algorithm in {'lasso','elastic_net'}:
            defaults={'max_iter':2000,'random_state':config.seed}
        elif config.algorithm in {'voting','stacking'}:
            estimators = [('linear',LogisticRegression(max_iter=1000,random_state=config.seed) if classifier else Ridge(solver='lsqr')),
                          ('forest',RandomForestClassifier(n_estimators=100,random_state=config.seed,n_jobs=1) if classifier
                           else RandomForestRegressor(n_estimators=100,random_state=config.seed,n_jobs=1))]
            defaults={'estimators':estimators,'n_jobs':1}
            if config.algorithm=='voting' and classifier:
                defaults['voting']='soft'
                if config.params.get('voting','soft')!='soft':
                    raise ValueError('Classification voting uses soft voting to provide probabilities')
            if config.algorithm=='stacking':
                defaults['cv']=3
            if 'estimators' in config.params or 'final_estimator' in config.params:
                raise ValueError('Use the built-in linear + forest ensemble members; estimator objects are not JSON parameters')
        elif config.algorithm=='dummy' and classifier:
            defaults = {'strategy': 'prior', 'random_state': config.seed}
        if config.algorithm=='multinomial_logistic' and config.params.get('solver','lbfgs')=='liblinear':
            raise ValueError('Multinomial logistic regression needs a multinomial solver such as lbfgs or saga')
        cls=classes[config.algorithm][0 if classifier else 1]
        if cls is None:
            raise ValueError(f'{config.algorithm} does not support {task_type}')
        self.estimator = cls(**{**defaults, **config.params})
        self.max_bytes=512*1024**2

    def _matrix(self,x):
        if self.config.algorithm=='gaussian_nb' and hasattr(x,'toarray'):
            if x.shape[0]*x.shape[1]*8>self.max_bytes:
                raise ValueError('Gaussian Naive Bayes dense input exceeds the memory limit')
            return x.toarray()
        return x

    def fit_validation(self,x,y,*,validation_data,progress=None,checkpoint=None,resume=None,
                       reset_patience=False,max_bytes=512*1024**2):
        if resume:
            raise ValueError('This model cannot resume a neural checkpoint')
        self.max_bytes=max_bytes
        return self.fit(x,y)

    def fit(self, x, y):
        self.estimator.fit(self._matrix(x), y)
        return self

    def predict(self, x):
        return self.estimator.predict(self._matrix(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self._matrix(x))

    def save(self, directory):
        path = Path(directory)/'model.skops'
        dump_safe(self.estimator, path)
        return path

    def load(self, directory):
        self.estimator = load_safe(Path(directory)/'model.skops')
        return self
