"""Conservative v1 starting points, not optimized hyperparameters."""
from copy import deepcopy
import math


def recommend(entry, rows, features):
    result=entry.to_dict();params=deepcopy(result['defaults']);notes=[]
    if entry.modality!='tabular' or rows<1:return result
    small=rows<1000;wide=features>max(20,rows//10)
    if 'n_neighbors' in params:
        params['n_neighbors']=min(rows, max(1,min(15,int(math.sqrt(rows)))))
        notes.append('Neighbor count is bounded by training rows and capped at 15.')
    if entry.key.endswith(':mlp'):
        params.update(hidden_sizes=[32,16] if small or wide else [128,64],batch_size=min(rows,32 if small else 128))
        notes.append('Smaller hidden layers for small or wide tables; batch size bounded by training rows.')
    if 'max_depth' in params and (small or wide):
        params['max_depth']=3 if wide else 5
        notes.append('Restrict tree depth for small or wide data.')
    if 'min_samples_leaf' in params:params['min_samples_leaf']=max(1,min(20,rows//200))
    if 'n_estimators' in params:params['n_estimators']=100 if rows<10000 else 200
    if 'num_leaves' in params and (small or wide):params['num_leaves']=15
    if 'depth' in params and (small or wide):params['depth']=4
    if 'n_components' in params and entry.key=='sklearn:pca':params['n_components']=max(1,min(2,rows,features))
    if 'perplexity' in params:params['perplexity']=max(1,min(30,(rows-1)/3)) if rows>1 else 1
    if 'n_clusters' in params:params['n_clusters']=min(params['n_clusters'],rows)
    if 'cv' in params:notes.append('Cross-validation also requires enough examples per class; inspect rare classes.')
    result['defaults']=params
    result['recommendation_note']=f'Heuristic starting point: {rows:,} prepared training rows, {features} input features. '+(' '.join(notes) or 'Standard defaults retained; no reliable size-only adjustment.')
    return result
