"""Inspect and predict from the exact data/model artifacts recorded at training."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from data.manifest import resolve_inside,sha256_file,write_json,validate_name
from training.artifacts import verify_artifacts,staged_directory,seal
from .project_models import ProjectModels


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def row_at(path,number):
    """One-based positional row; bounded reads, independent of dataframe index labels."""
    from eda_tool.loader import open_dataset
    from preprocessing.transforms.base import batches
    if type(number) is not int or number<1: raise ValueError('Row number must be at least 1')
    index=number-1
    with batches(lambda:open_dataset(path).iter_batches(batch_size=1000)) as iterator:
        for frame in iterator:
            if index<len(frame): return json.loads(frame.iloc[[index]].to_json(orient='records',date_format='iso'))[0]
            index-=len(frame)
    raise ValueError('Row number exceeds this split’s row count')


class ModelInspection:
    def __init__(self,project,name):
        self.project=project; self.store=ProjectModels(project); self.name=validate_name(name)
        self.record=read_json(resolve_inside(self.store.root,self.name)/'model.json')
        self.bundle=self.store.bundle(self.name)
        self.manifest=verify_artifacts(self.bundle)
        self.kind=self.manifest['kind']

    def data_root(self):
        if self.kind=='model_bundle':
            origin=self.manifest['metadata']; ws=self.project.workspace
            ws.get_task_run(origin['dataset_id'],origin['task_id'],origin['preparation_run'],verify=True)
            dataset=ws.get(origin['dataset_id'])
            path=resolve_inside(dataset.directory,f"tasks/{origin['task_id']}/runs/{origin['preparation_run']}")
            if sha256_file(path/'manifest.json')!=origin['preparation_manifest_sha256']:
                raise ValueError('The training split manifest has changed')
            return path
        if self.kind=='specialized_model':
            config=read_json(self.bundle/'config.json')
            # Prefer the recorded original path; relocated projects can find the same input id locally.
            original=Path(config['input']); path=original if original.exists() else self.project.directory/'inputs'/original.name
            verify_artifacts(path,'model_input')
            if sha256_file(path/'manifest.json')!=config['input_manifest_sha256']:
                raise ValueError('Model input snapshot differs from training')
            return path
        if self.kind=='exploration_model': return self.bundle
        raise ValueError('Unsupported model bundle')

    def splits(self):
        import pyarrow.parquet as pq
        root=self.data_root(); result={}
        for split in ('test','validation','train'):
            path=root/f'splits/{split}/part-00000.parquet' if self.kind=='model_bundle' else root/f'splits/{split}.parquet'
            if self.kind=='exploration_model':
                if split!='train': continue
                path=root/'input.parquet'
            if path.exists():
                with pq.ParquetFile(path) as file: count=file.metadata.num_rows
                if count: result[split]={'path':path,'rows':count}
        return result

    def details(self):
        metrics=read_json(self.bundle/'metrics.json')
        result={'name':self.name,'model':self.record['model'],'status':self.record['status'],
                'bundle':str(self.bundle),'created_at':self.record.get('created_at'),'metrics':metrics,
                'fitting_split':'all input rows' if self.kind=='exploration_model' else 'train',
                'environment':read_json(self.bundle/'environment.json')}
        if self.kind=='model_bundle':
            origin=self.manifest['metadata']; result['training_origin']=origin
            result['configuration']=read_json(self.bundle/'model_config.json')
            result['schema']=read_json(self.bundle/'schema.json')
            prep=read_json(self.data_root()/'manifest.json')
            result['split_details']=prep.get('metadata',prep)
        else:
            result['configuration']=read_json(self.bundle/'config.json')
            result['training_origin']=result['configuration'].get('input_metadata',{'protocol':'Full-input exploration'})
        return result

    def predict_row(self,number=1,split=None):
        sources=self.splits()
        split=split or next(iter(sources),None)
        if split not in sources: raise ValueError('This model has no rows in the requested split')
        values=row_at(sources[split]['path'],number)
        base={'split':split,'row':number,'values':values}
        if self.kind=='model_bundle':
            from prediction.example import predict_example
            result=predict_example(self.bundle,values)
        elif self.kind=='exploration_model':
            # These assignments were actually fitted; do not pretend transductive models predict new rows.
            assignment=row_at(self.bundle/'assignments.csv',number)
            result={'prediction':{k:v for k,v in assignment.items() if k!='source_row'},'status':'stored_assignment'}
        else:
            from training.specialized import predict_specialized
            config=read_json(self.bundle/'config.json'); modality=config['input_metadata']['modality']
            columns=config['input_metadata']['columns']; root=self.data_root()
            if modality=='image':
                result=predict_specialized(self.bundle,{'image':str(resolve_inside(root,values[columns['image']]))})
            elif modality=='text':
                result=predict_specialized(self.bundle,{'text':values[columns['text']]})
                if 'text' in result: result['prediction']=result['text']
            elif modality=='time_series':
                result=self._series_row(root,config,split,number)
            else:
                result=predict_specialized(self.bundle,{'user':values[columns['user']]})
                result['prediction']=result['recommendations']
                result['observed_item']=values[columns['item']]
                result['actual']=values.get(columns.get('weight'),1)
                result['actual_meaning']='Observed positive interaction weight; ranking scores are not predicted ratings'
            if 'target' in columns:
                result['actual']=values.get(columns['target'])
                if config['task']=='classification' and result.get('actual') is not None:
                    result['correct']=result.get('prediction')==result['actual']
        return {**base,**result}

    def _series_row(self,root,config,split,number):
        from training.specialized import predict_specialized
        value=config['input_metadata']['columns']['value']
        # Only preceding observations are used as model inputs.
        preceding=[]
        for part in ('train','validation','test'):
            path=root/f'splits/{part}.parquet'
            if not path.exists(): continue
            frame=pd.read_parquet(path,columns=[value])
            if part==split:
                actual=float(frame.iloc[number-1][value]); preceding.extend(frame.iloc[:number-1][value].tolist()); break
            preceding.extend(frame[value].tolist())
        if config['model']=='statsmodels:arima':
            from statsmodels.tsa.arima.model import ARIMA
            with np.load(self.bundle/'arima.npz',allow_pickle=False) as arrays:
                fitted=ARIMA(arrays['training'],order=tuple(config['options']['order'])).filter(arrays['parameters'])
            position=len(preceding)
            prediction=float(np.asarray(fitted.predict(start=position,end=position))[0])
            protocol='In-sample fitted prediction' if split=='train' else 'Fixed-origin forecast from training end'
        else:
            window=config['options']['window']
            if len(preceding)<window:
                raise ValueError(f'This row needs {window} prior observations; choose a later row or held-out split')
            prediction=predict_specialized(self.bundle,{'history':preceding[-window:]})['forecast'][0]
            protocol='One-step prediction using observed preceding history'
        return {'prediction':prediction,'actual':actual,'residual':actual-prediction,'protocol':protocol}

    def evaluate_test(self,max_rows=200000,max_bytes=512*1024**2):
        if self.kind!='model_bundle': raise ValueError('Held-out test evaluation currently supports supervised tabular models')
        from training import evaluate_test
        origin=self.manifest['metadata']
        report=evaluate_test(self.project.workspace,origin['dataset_id'],origin['task_id'],origin['training_run'],
                             candidate=origin['candidate'],max_rows=max_rows,max_bytes=max_bytes)
        self.record=self.store.update(self.name,test_evaluation=report)
        return report

    def export(self,destination,*,include_data=False,max_bytes=2*1024**3):
        """Export the model, inspection metadata, and optional exact associated data.

        This snapshot export is distinct from the existing tabular replay package.
        It also works for older models whose training source code differs.
        """
        destination=Path(destination).expanduser().resolve(); root=self.data_root() if include_data else None
        for source in (self.bundle,root):
            if source and (destination==source or destination.is_relative_to(source)):
                raise ValueError('Choose an export destination outside model/data artifacts')
        files=[(p,Path('bundle')/p.relative_to(self.bundle)) for p in self.bundle.rglob('*') if p.is_file()]
        if include_data and root!=self.bundle:
            files.extend((p,Path('data')/p.relative_to(root)) for p in root.rglob('*') if p.is_file())
        if sum(p.stat().st_size for p,_ in files)>max_bytes: raise ValueError('Export exceeds byte budget')
        with staged_directory(destination) as out:
            for source,relative in files:
                target=out/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
            write_json(out/'model-details.json',self.details())
            write_json(out/'project-model.json',self.record)
            (out/'README.md').write_text('Model export\n\nThe prediction bundle is in bundle/. Model details and metrics are saved alongside it. '
                +('data/ preserves the associated input/split artifacts (including available test data). ' if include_data and root!=self.bundle else '')
                +'This is a data/model snapshot, not an automatic replay package. Use the bundle prediction command appropriate to its model family.\n',encoding='utf-8')
            seal(out,'project_model_export',{'name':self.name,'includes_data':include_data})
        return destination
