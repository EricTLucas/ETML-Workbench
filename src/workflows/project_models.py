"""Project model names point to immutable training bundles, not run winners."""
from pathlib import Path
import json
from data.manifest import validate_name,write_json,utc_now,resolve_inside


class ProjectModels:
    def __init__(self,project):
        self.project=project
        self.root=project.directory/'models'

    def list(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.root.glob('*/model.json'))]

    def default_name(self):
        n=max([int(r['name'][5:]) for r in self.list() if r['name'].startswith('model') and r['name'][5:].isdigit()]+[0])+1
        while (self.root/f'model{n}').exists(): n+=1
        return f'model{n}'

    def create(self,key,*,name=None,params=None,input_reference=None):
        name=validate_name(name or self.default_name())
        path=resolve_inside(self.root,name)
        path.mkdir(parents=True,exist_ok=False)
        record={'name':name,'model':key,'params':params or {},'input':input_reference,
                'created_at':utc_now(),'status':'configured'}
        write_json(path/'model.json',record)
        return record

    def update(self,name,**values):
        path=resolve_inside(self.root,validate_name(name))/'model.json'
        record=json.loads(path.read_text(encoding='utf-8')); record.update(values)
        write_json(path,record,overwrite=True)
        return record

    def bundle(self,name):
        path=resolve_inside(self.root,validate_name(name))/'model.json'
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['status']!='complete': raise ValueError('This model has not completed training')
        return resolve_inside(self.project.directory,record['bundle'])

    def predict(self,name,values):
        from training.artifacts import verify_artifacts
        path=self.bundle(name); kind=verify_artifacts(path)['kind']
        if kind=='specialized_model':
            from training.specialized import predict_specialized
            return predict_specialized(path,values)
        if kind=='exploration_model':
            from training.exploration import predict_exploration
            return predict_exploration(path,values)
        from prediction.example import predict_example
        return predict_example(path,values)


def train_named_tabular(project,service,key,params,*,name=None,max_rows=200000,max_bytes=512*1024**2,progress=None):
    from models import ModelConfig
    from training import train_models
    state=service.state(); split=state.get('split')
    if not split: raise ValueError('Prepare and split the project dataset first')
    store=ProjectModels(project)
    record=store.create(key,name=name,params=params,input_reference={'dataset':service.dataset.dataset_id,**split})
    from .model_lifecycle import train_saved
    trained=train_saved(project,record['name'],max_rows=max_rows,max_bytes=max_bytes,progress=progress)
    return store.list(),trained['metrics']
