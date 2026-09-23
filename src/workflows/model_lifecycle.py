"""Retry saved configurations and advance resumable models without losing old runs."""
import json
import uuid
from pathlib import Path
from data.manifest import resolve_inside,validate_name
from .project_models import ProjectModels


def saved_record(project,name):
    store=ProjectModels(project)
    return json.loads((resolve_inside(store.root,validate_name(name))/'model.json').read_text(encoding='utf-8'))


def checkpoint_path(record):
    return record.get('latest_checkpoint') or record.get('metrics',{}).get('checkpoint')


def train_saved(project,name,*,additional_epochs=None,features=None,max_rows=200000,max_bytes=512*1024**2,progress=None):
    from models import ModelConfig
    from models.catalog import CATALOG
    from training import train_models
    store=ProjectModels(project); original=saved_record(project,name)
    key=original['model']; entry=CATALOG[key]; reference=original['input']
    if additional_epochs is None and original['status']=='complete':
        raise ValueError('This model is already trained; create a new model or continue from a checkpoint')
    if additional_epochs is not None:
        if type(additional_epochs) is not int or additional_epochs<1:
            raise ValueError('Additional epochs must be a positive integer')
        checkpoint=checkpoint_path(original)
        if key not in {'pytorch:mlp','tensorflow:mlp'} or not checkpoint:
            raise ValueError('This model has no supported resumable checkpoint')
        from training.checkpoints import load_checkpoint
        context=load_checkpoint(checkpoint)
        if context['config']['backend']+':'+context['config']['algorithm']!=key:
            raise ValueError('Checkpoint model differs from the saved configuration')
        expected=dict(context['config']['params']); requested=dict(original['params'])
        expected.pop('epochs',None); requested.pop('epochs',None)
        if expected!=requested: raise ValueError('Saved settings differ from checkpoint settings; retry from scratch instead')
        origin=context['origin']
        if not isinstance(reference,dict) or (origin['dataset_id'],origin['task_id'],origin['preparation_run'])!=(reference['dataset'],reference['task_id'],reference['run_id']):
            raise ValueError('Checkpoint input differs from this model’s recorded split')
        state=json.loads((Path(checkpoint)/'state.json').read_text(encoding='utf-8'))
        epochs=state['epoch']+additional_epochs
    else: checkpoint=None
    if entry.modality=='tabular' and not isinstance(reference,dict):
        features=features or original.get('features')
        if not features: raise ValueError('This older exploration record needs explicit feature columns before retrying')
    latest=None
    def report(event):
        nonlocal latest
        if event.get('checkpoint'):
            latest=event['checkpoint']; store.update(name,latest_checkpoint=latest)
        if progress: progress(event)
    store.update(name,status='training',error=None)
    try:
        if isinstance(reference,dict) and entry.modality=='tabular':
            if additional_epochs is not None:
                from training.checkpoints import resume_training
                result=resume_training(project.workspace,checkpoint,epochs=epochs,reset_patience=state['stopped_early'],
                    max_rows=max_rows,max_bytes=max_bytes,progress=report,label=name)
            else:
                result=train_models(project.workspace,reference['dataset'],reference['task_id'],reference['run_id'],
                    configs=[ModelConfig(*key.split(':'),params=original['params'])],max_rows=max_rows,max_bytes=max_bytes,progress=report,label=name)
            candidate=next(r for r in result.leaderboard if r['model']['backend']+':'+r['model']['algorithm']==key)
            bundle=result.directory/'candidates'/candidate['candidate']
            changes={'training_run':result.run_id,'metrics':candidate,'params':candidate['model']['params'],
                     'latest_checkpoint':candidate.get('checkpoint')}
        else:
            bundle=store.root/name/('bundle-'+uuid.uuid4().hex[:12])
            if entry.modality=='tabular':
                from training.exploration import train_exploration
                result=train_exploration(reference,key,bundle,features=features,params=original['params'],max_rows=max_rows,max_bytes=max_bytes)
            else:
                from training.specialized import train_specialized
                result=train_specialized(reference,key,bundle,params=original['params'],max_bytes=max_bytes,progress=report)
            changes={'metrics':result,'features':features}
        history=list(original.get('history',[]))
        history.append({k:v for k,v in original.items() if k!='history'})
        return store.update(name,status='complete',bundle=bundle.relative_to(project.directory).as_posix(),
            history=history,error=None,test_evaluation=None,**changes)
    except BaseException as exc:
        # A failed continuation leaves the last successful model usable.
        changes={'status':original['status'] if original.get('bundle') else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                 'error':str(exc)}
        if latest: changes['latest_checkpoint']=latest
        store.update(name,**changes)
        raise
