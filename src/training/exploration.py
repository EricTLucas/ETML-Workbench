"""Explicit full-input unsupervised exploration (no supervised test scores)."""
import inspect
from pathlib import Path
import time
import numpy as np
import pandas as pd
from models.catalog import CATALOG
from models.adapters.sklearn import dump_safe,load_safe
from data.model_inputs import _read
from data.manifest import write_json
from .features import infer_schema,fit_encoder,encode
from .artifacts import staged_directory,seal,verify_artifacts,environment

TRANSDUCTIVE={'hierarchical','dbscan','tsne'}


def train_exploration(source,key,destination,*,features=None,params=None,seed=42,max_rows=200000,max_bytes=512*1024**2):
    from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.ensemble import IsolationForest
    algorithm=key.split(':')[-1]
    classes={'kmeans':KMeans,'hierarchical':AgglomerativeClustering,'dbscan':DBSCAN,
             'gaussian_mixture':GaussianMixture,'pca':PCA,'tsne':TSNE,'isolation_forest':IsolationForest}
    if key not in CATALOG or algorithm not in classes: raise ValueError('Unknown exploration model')
    frame=_read(source,max_rows,max_bytes)
    if features:
        if not set(features)<=set(frame): raise ValueError('Unknown feature columns')
        frame=frame[list(features)]
    if not len(frame.columns): raise ValueError('Choose at least one feature')
    if algorithm in {'hierarchical','dbscan','tsne'} and len(frame)>10000:
        raise ValueError('This exploration adapter is limited to 10,000 rows; explicitly sample before fitting')
    if algorithm in {'hierarchical','dbscan'} and len(frame)**2*8>max_bytes:
        raise ValueError('Potential pairwise-distance memory exceeds budget; sample fewer rows')
    options={**CATALOG[key].defaults,**(params or {})}
    if algorithm in {'kmeans','gaussian_mixture','pca','tsne','isolation_forest'}: options.setdefault('random_state',seed)
    if algorithm=='tsne':
        if options.get('perplexity',30)>=len(frame): raise ValueError('t-SNE perplexity must be smaller than row count')
        if 'max_iter' in options and 'max_iter' not in inspect.signature(TSNE).parameters:
            options['n_iter']=options.pop('max_iter')
    started=time.perf_counter(); schema=infer_schema(frame); encoder=fit_encoder(frame,schema,scale=True)
    x=encode(encoder,frame,schema,max_bytes)
    if x.shape[0]*x.shape[1]*8>max_bytes: raise ValueError('Dense exploration input exceeds memory budget')
    if hasattr(x,'toarray'): x=x.toarray()
    model=classes[algorithm](**options)
    result={}; columns={}
    if algorithm in {'pca','tsne'}:
        embedding=model.fit_transform(x)
        columns={f'component_{i+1}':embedding[:,i] for i in range(embedding.shape[1])}
        if algorithm=='pca': result['explained_variance_ratio']=model.explained_variance_ratio_.tolist()
        else: result['kl_divergence']=float(model.kl_divergence_)
    else:
        labels=model.fit_predict(x); columns['cluster' if algorithm!='isolation_forest' else 'anomaly']=labels
        if algorithm=='isolation_forest':
            columns['normality_score']=model.decision_function(x); result['flagged_rows']=int((labels==-1).sum())
        else:
            unique,counts=np.unique(labels,return_counts=True); result['cluster_counts']={str(k):int(v) for k,v in zip(unique,counts)}
            from sklearn.metrics import silhouette_score
            observed=labels!=-1
            if 1<len(set(labels[observed]))<observed.sum():
                # Bounded sample; small clusters can disappear, making silhouette undefined.
                try: result['silhouette']=float(silhouette_score(x[observed],labels[observed],sample_size=min(2000,int(observed.sum())),random_state=seed))
                except ValueError: result['silhouette']=None
    result.update(fit_seconds=time.perf_counter()-started,rows=len(frame),protocol='Full-input exploratory fit; no held-out performance estimate')
    with staged_directory(destination) as root:
        dump_safe(model,root/'model.skops'); dump_safe(encoder,root/'encoder.skops')
        frame.to_parquet(root/'input.parquet',index=False)
        pd.DataFrame({'source_row':np.arange(len(frame)),**columns}).to_csv(root/'assignments.csv',index=False)
        write_json(root/'config.json',{'model':key,'options':options,'schema':schema,'seed':seed,
                                     'predicts_new_rows':algorithm not in TRANSDUCTIVE})
        write_json(root/'metrics.json',result); write_json(root/'environment.json',environment())
        seal(root,'exploration_model',{'model':key})
    return {'directory':str(Path(destination).resolve()),**result}


def predict_exploration(bundle,records):
    import json
    root=Path(bundle); verify_artifacts(root,'exploration_model')
    context=json.loads((root/'config.json').read_text())
    if not context['predicts_new_rows']:
        raise ValueError('This model has no out-of-sample prediction; inspect assignments.csv or fit a new exploration')
    frame=pd.DataFrame([records] if isinstance(records,dict) else records)
    matrix=encode(load_safe(root/'encoder.skops'),frame,context['schema'])
    if matrix.shape[0]*matrix.shape[1]*8>512*1024**2: raise ValueError('Dense prediction input exceeds memory budget')
    if hasattr(matrix,'toarray'): matrix=matrix.toarray()
    model=load_safe(root/'model.skops'); algorithm=context['model'].split(':')[-1]
    if algorithm=='pca': return {'embedding':model.transform(matrix).tolist()}
    output={'prediction':model.predict(matrix).tolist()}
    if algorithm=='gaussian_mixture': output['probabilities']=model.predict_proba(matrix).tolist()
    if algorithm=='isolation_forest': output['normality_score']=model.decision_function(matrix).tolist()
    return output
