import numpy as np
from sklearn import metrics

DIRECTIONS = {'accuracy':'max', 'balanced_accuracy':'max', 'f1_macro':'max', 'log_loss':'min',
              'roc_auc':'max', 'mae':'min', 'rmse':'min', 'r2':'max'}


def evaluate_predictions(y, prediction, task_type, *, probabilities=None, n_classes=None):
    if not len(y):
        raise ValueError('Evaluation split is empty')
    if task_type == 'regression':
        if not np.isfinite(prediction).all():
            raise ValueError('Model produced nonfinite predictions')
        result = {'mae':metrics.mean_absolute_error(y,prediction),
                  'rmse':np.sqrt(metrics.mean_squared_error(y,prediction)),
                  'r2':metrics.r2_score(y,prediction) if len(y)>1 else None}
    else:
        result = {'accuracy':metrics.accuracy_score(y,prediction),
                  'balanced_accuracy':metrics.balanced_accuracy_score(y,prediction),
                  'f1_macro':metrics.f1_score(y,prediction,average='macro',labels=list(range(n_classes)),zero_division=0)}
        if probabilities is not None:
            if (probabilities.shape != (len(y),n_classes) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1),1,atol=1e-5)):
                raise ValueError('Invalid class probabilities')
            result['log_loss'] = metrics.log_loss(y,probabilities,labels=list(range(n_classes)))
            result['roc_auc'] = metrics.roc_auc_score(y,probabilities[:,1]) if n_classes==2 and len(np.unique(y))==2 else None
    return {k:None if v is None or not np.isfinite(v) else float(v) for k,v in result.items()}


def validate_metric(metric, task_type):
    allowed = {'mae','rmse','r2'} if task_type=='regression' else {'accuracy','balanced_accuracy','f1_macro','log_loss','roc_auc'}
    if metric not in allowed:
        raise ValueError(f'Metric {metric!r} is incompatible with {task_type}')
    return DIRECTIONS[metric]
