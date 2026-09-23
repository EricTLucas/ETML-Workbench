from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    factory: object
    task_types: tuple = ('classification', 'regression')
    probabilities: bool = True
    incremental: bool = False
    onnx: bool = False
    epochs: bool = False
    resumable: bool = False


def _sklearn(config, task_type):
    from .adapters.sklearn import SklearnAdapter
    return SklearnAdapter(config, task_type)


def _xgboost(config, task_type):
    from .adapters.xgboost import XGBoostAdapter
    return XGBoostAdapter(config, task_type)


_REGISTRY = {('sklearn', name): ModelSpec(_sklearn, onnx=name != 'dummy')
             for name in ('dummy', 'linear', 'random_forest')}
_REGISTRY['xgboost', 'boosted_trees'] = ModelSpec(_xgboost)
for name in ('decision_tree','extra_trees','gradient_boosting','adaboost','svm','knn','voting','stacking'):
    _REGISTRY['sklearn',name]=ModelSpec(_sklearn)
for name in ('linear_regression','ridge','lasso','elastic_net'):
    _REGISTRY['sklearn',name]=ModelSpec(_sklearn,task_types=('regression',),probabilities=False)
for name in ('logistic_regression','multinomial_logistic','gaussian_nb','multinomial_nb','bernoulli_nb'):
    _REGISTRY['sklearn',name]=ModelSpec(_sklearn,task_types=('classification',))


def _lightgbm(config,task_type):
    from .adapters.boosting import LightGBMAdapter
    return LightGBMAdapter(config,task_type)


def _catboost(config,task_type):
    from .adapters.boosting import CatBoostAdapter
    return CatBoostAdapter(config,task_type)


_REGISTRY['lightgbm','boosted_trees']=ModelSpec(_lightgbm)
_REGISTRY['catboost','boosted_trees']=ModelSpec(_catboost)


def _torch(config, task_type):
    from .adapters.pytorch import TorchAdapter
    return TorchAdapter(config,task_type)


def _tensorflow(config, task_type):
    from .adapters.tensorflow import TensorFlowAdapter
    return TensorFlowAdapter(config,task_type)


_REGISTRY['pytorch','mlp'] = ModelSpec(_torch,epochs=True,resumable=True)
_REGISTRY['tensorflow','mlp'] = ModelSpec(_tensorflow,epochs=True,resumable=True)


def register_model(backend, algorithm, spec):
    if (backend, algorithm) in _REGISTRY or not isinstance(spec, ModelSpec) or not callable(spec.factory):
        raise ValueError('Duplicate model or invalid adapter specification')
    _REGISTRY[backend, algorithm] = spec


def available_models():
    return {f'{backend}:{algorithm}': {'tasks':list(spec.task_types), 'probabilities':spec.probabilities,
                                     'incremental':spec.incremental, 'onnx_model_only':spec.onnx,
                                     'epochs':spec.epochs,'resumable':spec.resumable}
            for (backend, algorithm), spec in _REGISTRY.items()}


def create_model(config, task_type):
    key = (config.backend, config.algorithm)
    if key not in _REGISTRY:
        raise ValueError(f'Unsupported model {key}; use available_models()')
    spec = _REGISTRY[key]
    if task_type not in spec.task_types:
        raise ValueError('Model does not support this task')
    return spec.factory(config, task_type)
