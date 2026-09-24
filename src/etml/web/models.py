"""Model browser operations backed by the same services as the project CLI."""
from pathlib import Path
import json
import uuid
import zipfile
from dataclasses import asdict

from data.manifest import resolve_inside, validate_name
from models.catalog import CATALOG, suggest
from workflows.project_models import ProjectModels
from workflows.model_lifecycle import train_saved, saved_record, checkpoint_path
from workflows.model_inspection import ModelInspection


class ModelService:
    def __init__(self, service):
        self.service = service

    def overview(self, project_name):
        project = self.service.store.get(project_name)
        store = ProjectModels(project)
        inputs = []
        from data.model_inputs import load_model_input
        for path in sorted((project.directory/'inputs').glob('input-*')):
            _, info = load_model_input(path)
            inputs.append({'id': path.name, 'modality': info['modality'], 'task': info.get('task')})
        return {'models': store.list(), 'default_name': store.default_name(), 'inputs': inputs,
                'catalog': [dict(e.to_dict(), recommended_tasks=[t for t in e.tasks if e in suggest(t,e.modality)]) for e in CATALOG.values()]}

    def details(self, project_name, name):
        project = self.service.store.get(project_name)
        record = saved_record(project, name)
        entry = CATALOG[record['model']]
        result = {'record': record, 'entry': entry.to_dict(),
                  'can_resume': record['model'] in {'pytorch:mlp','tensorflow:mlp'} and bool(checkpoint_path(record)),
                  'details': None, 'splits': {}, 'custom_fields': {}}
        if record['status'] != 'complete':
            return result
        inspection = ModelInspection(project, name)
        result['details'] = inspection.details()
        result['kind'] = inspection.kind
        result['splits'] = {key: {'rows': value['rows']} for key,value in inspection.splits().items()}
        if inspection.kind == 'model_bundle':
            schema = result['details']['schema']
            result['custom_fields'] = {'values': {c: None for c in schema['raw_columns']}, 'dtypes': schema.get('raw_dtypes',{})}
            result['custom_target'] = {'name': inspection.manifest['metadata']['target'],
                'task': inspection.manifest['metadata']['task_type'],
                'classes': json.loads((inspection.bundle/'labels.json').read_text(encoding='utf-8'))['classes']}
        elif inspection.kind == 'exploration_model':
            result['custom_fields'] = {'values': {c: None for c in record.get('features',[])}, 'dtypes': {}}
        else:
            examples = {'image': {'image': ''}, 'text': {'text': ''},
                        'time_series': {'history': [], 'horizon': 1}, 'recommendation': {'user': '', 'k': 10}}
            if record['model'] == 'statsmodels:arima':
                examples['time_series'] = {'horizon': 1}
            elif entry.modality == 'time_series':
                extra = json.loads((inspection.bundle/'model_details.json').read_text(encoding='utf-8'))
                examples['time_series']['history'] = extra.get('tail', [])
            if record['model'] in {'transformers:gpt','transformers:t5'}:
                examples['text']['max_new_tokens'] = 32
            result['custom_fields'] = {'values': examples[entry.modality], 'dtypes': {}}
        result['custom_supported'] = record['model'] not in {'sklearn:hierarchical','sklearn:dbscan','sklearn:tsne'}
        # History is recorded by epoch-aware backends and remains available after reload.
        history = inspection.bundle/'history.json'
        if history.exists():
            result['history'] = json.loads(history.read_text(encoding='utf-8'))
            from training.curves import learning_curves
            result['learning_curves'] = learning_curves(result['history'])
        return result

    def train(self, project_name, payload, progress):
        project = self.service.store.get(project_name)
        store = ProjectModels(project)
        action = payload.get('action','new')
        name = payload.get('name') or store.default_name()
        from models.dependencies import ensure_dependencies
        if action not in {'new','retry','resume'}: raise ValueError('Unknown training action')
        key = payload.get('key') if action == 'new' else saved_record(project, name)['model']
        if key not in CATALOG: raise ValueError('Choose a model from the catalog')
        ensure_dependencies(key, progress)
        if action == 'new':
            key = payload['key']
            if key not in CATALOG:
                raise ValueError('Choose a model from the catalog')
            entry = CATALOG[key]
            params = payload.get('params',{})
            if not isinstance(params,dict):
                raise ValueError('Hyperparameters must be an object')
            params = {**entry.defaults, **params}
            task = payload.get('task')
            if task not in entry.tasks:
                raise ValueError('The selected model does not support this task')
            progress({'stage':'preparing','model':key,'message':'Preparing model input'})
            if entry.modality == 'tabular' and task in {'classification','regression'}:
                prep = self.service.preparation(project_name, payload.get('dataset_id'))
                state = prep.state()
                if not state.get('split'):
                    raise ValueError('Prepare and split the dataset in Data first')
                if prep.task().task_type != task:
                    raise ValueError('Model task must match the prepared dataset')
                reference = {'dataset': prep.dataset.dataset_id, **state['split']}
            elif entry.modality == 'tabular':
                source = Path(payload['source']).expanduser().resolve()
                features = payload.get('features')
                if not features or not isinstance(features,list):
                    raise ValueError('Select the feature columns for exploration')
                reference = str(source)
            else:
                existing = payload.get('input_id')
                if existing:
                    from data.model_inputs import load_model_input
                    path = resolve_inside(project.directory/'inputs', validate_name(existing))
                    _, info = load_model_input(path)
                    if info['modality'] != entry.modality or info['task'] != task:
                        raise ValueError('Saved input does not match the model task')
                else:
                    from data.model_inputs import prepare_model_input
                    spec = dict(payload['input_spec'], modality=entry.modality, task=task)
                    path = prepare_model_input(project, spec, max_bytes=self.service.max_bytes)
                reference = str(path)
            store.create(key, name=name, params=params, input_reference=reference)
            if entry.modality == 'tabular' and task not in {'classification','regression'}:
                store.update(name, features=features)
        elif action not in {'retry','resume'}:
            raise ValueError('Unknown training action')
        additional = payload.get('additional_epochs') if action == 'resume' else None
        if action == 'resume' and (type(additional) is not int or additional < 1):
            raise ValueError('Additional epochs must be a positive integer')
        record = train_saved(project, name, additional_epochs=additional,
                             max_bytes=self.service.max_bytes, progress=progress)
        progress({'stage':'evaluating','message':'Computing final results'})
        warning = None
        inspection = ModelInspection(project,name)
        if inspection.kind == 'model_bundle' and 'test' in inspection.splits():
            try:
                inspection.evaluate_test(max_bytes=self.service.max_bytes)
            except (ValueError, OSError) as exc:
                warning = 'Model saved. Test metrics are unavailable: '+str(exc)
        return {'name': name, 'warning': warning}

    def predict(self, project_name, name, payload):
        project = self.service.store.get(project_name)
        from models.dependencies import ensure_dependencies
        ensure_dependencies(saved_record(project,name)['model'])
        inspection = ModelInspection(project,name)
        if payload.get('mode') == 'row':
            return inspection.predict_row(payload.get('row',1),payload.get('split'))
        if payload.get('mode') != 'custom':
            raise ValueError('Choose dataset row or custom input')
        return ProjectModels(project).predict(name,payload['values'])

    def evaluate(self, project_name, name):
        return ModelInspection(self.service.store.get(project_name),name).evaluate_test(max_bytes=self.service.max_bytes)

    def export(self, project_name, name, payload):
        project = self.service.store.get(project_name)
        inspection = ModelInspection(project,name)
        export_id = 'export-'+uuid.uuid4().hex
        root = project.directory/'exports'/export_id
        folder = inspection.export(root/'snapshot', include_data=bool(payload.get('include_data')), max_bytes=self.service.max_bytes)
        archive = root/'model.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
            for file in folder.rglob('*'):
                if file.is_file():
                    output.write(file,file.relative_to(folder).as_posix())
        return {'id': export_id, 'name': name+'.zip', 'directory': str(root),
                'url': '/downloads/'+project_name+'/'+export_id}

    def compare(self, project_name):
        import math
        groups = {}
        unavailable = []
        project=self.service.store.get(project_name)
        for record in ProjectModels(project).list():
            metrics = (record.get('test_evaluation') or {}).get('metrics', {})
            evaluation_split='test'
            if not metrics and record['status']=='complete' and isinstance(record.get('input'),dict):
                inspection=ModelInspection(project,record['name'])
                if inspection.kind=='model_bundle' and 'test' not in inspection.splits():
                    metrics=inspection.details()['metrics'].get('training_metrics',{})
                    evaluation_split='train'
            values = {k:v for k,v in metrics.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)}
            reference = record.get('input')
            if record['status'] != 'complete' or not values or not isinstance(reference,dict):
                unavailable.append(record['name']); continue
            key = json.dumps(reference,sort_keys=True)
            group = groups.setdefault(key, {'reference':reference,'evaluation_split':evaluation_split,'models':[]})
            group['models'].append({'name':record['name'],'model':record['model'],'metrics':values,'evaluation_split':evaluation_split})
        return {'groups':list(groups.values()),'unavailable':unavailable}

    def predict_csv(self, project_name, name, source):
        import pandas as pd
        from prediction import Predictor
        from models.dependencies import ensure_dependencies
        project=self.service.store.get(project_name)
        inspection=ModelInspection(project,name)
        if inspection.kind != 'model_bundle':
            raise ValueError('CSV target prediction currently supports supervised tabular models.')
        ensure_dependencies(inspection.record['model'])
        predictor=Predictor.load(inspection.bundle)
        target=inspection.manifest['metadata']['target']
        export_id='predictions-'+uuid.uuid4().hex
        root=project.directory/'exports'/export_id
        root.mkdir(parents=True)
        path=root/'predictions.csv'
        rows=excluded=0
        try:
            with pd.read_csv(source,chunksize=10000) as reader, pd.read_csv(source,chunksize=10000,dtype=str,keep_default_na=False) as raw_reader:
                for frame, raw in zip(reader,raw_reader):
                    frame=frame.reset_index(drop=True)
                    prediction=predictor.predict(frame,probabilities=False)
                    frame=raw.reset_index(drop=True)
                    def free_column(base):
                        candidate=base; i=2
                        while candidate in frame: candidate=base+'_'+str(i); i+=1
                        return candidate
                    if target in frame: frame[free_column(target+'_actual')]=frame[target]
                    frame[target]=prediction['prediction'].to_numpy()
                    frame.to_csv(path,index=False,mode='w' if rows==0 else 'a',header=rows==0)
                    rows+=len(frame)
                    excluded+=int((prediction['status']!='predicted').sum())
            if not rows: raise ValueError('The CSV has no data rows.')
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {'url':'/downloads/'+project_name+'/'+export_id+'/predictions.csv',
                'id':export_id,'columns':frame.columns.tolist(),
                'name':name+'-predictions.csv','rows':rows,'excluded':excluded,'target':target}
