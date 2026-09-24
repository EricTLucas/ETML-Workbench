from dataclasses import replace
import math
import random
from models import ModelConfig


def search_configs(base,space,*,method='random',trials=10,seed=42,max_trials=100):
    """Sample finite parameter choices without replacement; never materialize a large grid."""
    if not isinstance(base,ModelConfig) or not isinstance(space,dict) or not space:
        raise ValueError('Search needs a model and a nonempty parameter-choice dictionary')
    if method not in {'grid','random'} or type(seed) is not int:
        raise ValueError('Use grid or random search and an integer seed')
    if any(type(v) is not int or v<1 for v in (trials,max_trials)):
        raise ValueError('Trial limits must be positive integers')
    keys = sorted(space)
    choices = [space[k] for k in keys]
    if any(not isinstance(v,list) or not v for v in choices):
        raise ValueError('Every search parameter needs a nonempty JSON list of choices')
    total = math.prod(len(v) for v in choices)
    count = total if method=='grid' else min(trials,total)
    if count>max_trials or total>2**63-1:
        raise ValueError(f'Search exceeds its budget ({count} requested, {max_trials} allowed)')
    positions = range(total) if method=='grid' else random.Random(seed).sample(range(total),count)
    result = []
    for position in positions:
        params = {}
        for key,values in reversed(list(zip(keys,choices))):
            position,index = divmod(position,len(values))
            params[key] = values[index]
        result.append(replace(base,params={**base.params,**params}))
    return result


def search_models(workspace,dataset_id,task_id,preparation_run,*,base,space,method='random',
                  trials=10,seed=42,max_trials=100,**kwargs):
    from .runner import train_models
    configs = search_configs(base,space,method=method,trials=trials,seed=seed,max_trials=max_trials)
    experiment = {'method':method,'seed':seed,'space':space,'trials':len(configs),
                  'baseline_added_automatically':False,'selection_split':'validation'}
    return train_models(workspace,dataset_id,task_id,preparation_run,configs=configs,experiment=experiment,**kwargs)
