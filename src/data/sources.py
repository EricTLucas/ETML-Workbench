"""Provider data is materialized into immutable local snapshots with provenance."""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit,urlunsplit
from urllib.request import urlopen,Request
from importlib.metadata import version
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .workspace import DatasetWorkspace
from .manifest import validate_filename

SKLEARN_DATASETS={'iris':'classification','wine':'classification','breast_cancer':'classification',
                  'digits':'classification','diabetes':'regression'}


def _workspace(workspace):
    return workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)


def _limits(max_rows,max_bytes):
    if any(type(v) is not int or v<1 for v in (max_rows,max_bytes)):
        raise ValueError('Source limits must be positive integers')


def _snapshot(frame,path,max_rows,max_bytes):
    if len(frame)>max_rows or int(frame.memory_usage(deep=True).sum())>max_bytes:
        raise ValueError('Dataset exceeds import limits; increase them explicitly')
    if not len(frame): raise ValueError('Dataset is empty')
    frame.to_parquet(path,index=False)
    if path.stat().st_size>max_bytes: raise ValueError('Snapshot exceeds max_bytes')


def import_sklearn(workspace,name,*,dataset_id=None,max_rows=200_000,max_bytes=512*1024**2):
    _limits(max_rows,max_bytes)
    if name not in SKLEARN_DATASETS: raise ValueError(f'Choose one of: {", ".join(SKLEARN_DATASETS)}')
    from sklearn import datasets
    bunch=getattr(datasets,'load_'+name)(as_frame=True)
    frame=bunch.frame.copy()
    provenance={'provider':'sklearn','name':name,'version':version('scikit-learn'),
                'target':'target','task_type':SKLEARN_DATASETS[name]}
    if provenance['task_type']=='classification': provenance['class_names']=[str(v) for v in bunch.target_names]
    with TemporaryDirectory() as temp:
        path=Path(temp)/(name+'.parquet')
        _snapshot(frame,path,max_rows,max_bytes)
        return _workspace(workspace).import_files(path,name=name,dataset_id=dataset_id,max_bytes=max_bytes,provenance=provenance)


def import_openml(workspace,data_id,*,dataset_id=None,max_rows=200_000,max_bytes=512*1024**2):
    _limits(max_rows,max_bytes)
    if type(data_id) is not int or data_id<1: raise ValueError('OpenML requires a positive numeric data ID')
    from sklearn.datasets import fetch_openml
    bunch=fetch_openml(data_id=data_id,as_frame=True)
    # fetch_openml itself is an in-memory API; limits apply to its returned frame.
    provenance={'provider':'openml','data_id':data_id,'details':json.loads(json.dumps(bunch.details,default=str)),
                'target_names':list(bunch.target_names),'loader_version':version('scikit-learn')}
    with TemporaryDirectory() as temp:
        path=Path(temp)/f'openml-{data_id}.parquet'
        _snapshot(bunch.frame,path,max_rows,max_bytes)
        return _workspace(workspace).import_files(path,dataset_id=dataset_id,max_bytes=max_bytes,provenance=provenance)


def import_url(workspace,url,*,filename=None,dataset_id=None,max_bytes=512*1024**2):
    _limits(1,max_bytes)
    parsed=urlsplit(url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Supply an HTTP(S) data-file URL without embedded credentials')
    filename=filename or Path(parsed.path).name
    validate_filename(filename)
    if Path(filename).suffix.lower() not in {'.csv','.parquet','.json','.jsonl','.ndjson','.tsv'}:
        raise ValueError('Use a direct CSV, TSV, Parquet, JSON or JSONL link and filename')
    provenance={'provider':'url','url':urlunsplit((parsed.scheme,parsed.netloc,parsed.path,'','')),
                'query_omitted':bool(parsed.query)}
    with TemporaryDirectory() as temp:
        path=Path(temp)/filename
        with urlopen(Request(url,headers={'User-Agent':'ETML-Workbench'}),timeout=60) as response,path.open('wb') as out:
            size=0
            while chunk:=response.read(1024**2):
                size+=len(chunk)
                if size>max_bytes: raise ValueError('Download exceeds max_bytes')
                out.write(chunk)
        return _workspace(workspace).import_files(path,dataset_id=dataset_id,max_bytes=max_bytes,provenance=provenance)


def import_huggingface(workspace,repository,*,config=None,split='train',splits=None,revision=None,
                       columns=None,dataset_id=None,max_rows=200_000,max_bytes=512*1024**2):
    """splits maps local train/validation/test roles to Hub split names. Limits are total."""
    _limits(max_rows,max_bytes)
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError('Install Hugging Face support: pip install -e ".[huggingface]"') from exc
    if splits is not None and ('train' not in splits or set(splits)-{'train','validation','test'}):
        raise ValueError('Hugging Face splits must map train and optional validation/test roles')
    roles=splits or {'data':split}
    if not all(isinstance(v,str) and v for v in roles.values()): raise ValueError('Split names must be strings')
    resolved=HfApi().dataset_info(repository,revision=revision).sha
    provenance={'provider':'huggingface','repository':repository,'config':config,'revision':resolved,
                'requested_revision':revision,'splits':roles,'columns':columns,'version':version('datasets')}
    with TemporaryDirectory() as temp:
        paths={}; total_rows=0; total_memory=0
        for role,remote_split in roles.items():
            stream=load_dataset(repository,name=config,split=remote_split,revision=resolved,streaming=True)
            if columns: stream=stream.select_columns(columns)
            path=Path(temp)/(role+'.parquet'); writer=None
            declared=stream.features.arrow_schema if getattr(stream,'features',None) is not None else None
            iterator=iter(stream.iter(batch_size=1000))
            try:
                for batch in iterator:
                    frame=pd.DataFrame(batch)
                    if not len(frame): continue
                    total_rows+=len(frame)
                    total_memory+=int(frame.memory_usage(deep=True).sum())
                    if total_rows>max_rows or total_memory>max_bytes:
                        raise ValueError('Hub snapshot exceeds limits; choose fewer rows upstream or raise limits')
                    table=pa.Table.from_pandas(frame,preserve_index=False)
                    if any(pa.types.is_nested(field.type) or pa.types.is_binary(field.type) for field in table.schema):
                        raise ValueError('Choose scalar tabular columns; nested/media columns need an explicit feature transform')
                    if writer is None: writer=pq.ParquetWriter(path,declared or table.schema)
                    writer.write_table(table.cast(writer.schema))
            finally:
                if writer: writer.close()
                close=getattr(iterator,'close',None)
                if close: close()
            if not path.exists(): raise ValueError(f'Empty Hub split: {remote_split}')
            paths[role]=path
        workspace=_workspace(workspace)
        if splits:
            return workspace.import_presplit(paths['train'],validation=paths.get('validation'),test=paths.get('test'),
                name=repository,dataset_id=dataset_id,max_bytes=max_bytes,provenance=provenance)
        return workspace.import_files(paths['data'],name=repository,dataset_id=dataset_id,max_bytes=max_bytes,provenance=provenance)
