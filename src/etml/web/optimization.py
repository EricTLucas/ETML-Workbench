"""Persisted, sequential model experiments based on immutable preparation artifacts."""
from pathlib import Path
import json
import time
import uuid
from data.manifest import write_json
from workflows.model_inspection import ModelInspection
from workflows.project_models import ProjectModels
from workflows.model_lifecycle import train_saved
from training.control import check_cancelled, TrainingCancelled


def experiments(project):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((project.directory/'optimizations').glob('opt-*/experiment.json'))]


def run_experiment(models, project_name, name, payload, progress):
    from preprocessing import Recipe
    from tasks import TaskConfig
    from splitting import SplitConfig
    from workflows.preparation import prepare_task
    from models.dependencies import ensure_dependencies
    project=models.service.store.get(project_name);store=ProjectModels(project)
    inspection=ModelInspection(project,name)
    if inspection.kind!='model_bundle':raise ValueError('Optimization currently supports supervised tabular models.')
    original=inspection.record;reference=dict(original['input']);prep_root=inspection.data_root()
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    config=read(prep_root/'split_config.json');mode=payload.get('mode')
    source_manifest=project.workspace.get_task_run(reference['dataset'],reference['task_id'],reference['run_id'],verify=True)
    metadata=source_manifest.metadata
    if metadata.get('source_version','raw')!='raw':raise ValueError('Optimization requires a raw-data preparation run.')
    if mode=='seeds':
        count=payload.get('count',5);start=payload.get('start_seed',43)
        if type(count) is not int or not 1<=count<=50:raise ValueError('Choose between 1 and 50 split trials.')
        if type(start) is not int or start<0 or start+count>2**32:raise ValueError('Choose a nonnegative starting seed below 2^32.')
        if config['strategy']=='chronological':raise ValueError('Chronological splits do not change with random seeds. Use hyperparameter trials.')
        dataset=project.workspace.get(reference['dataset'])
        if 'validation' in dataset.manifest.split_files:raise ValueError('Uploaded validation rows are fixed. Use hyperparameter trials, or prepare data without a fixed validation upload.')
        if dataset.manifest.split_files and reference.get('validation_fraction',.2)<=0:raise ValueError('This presplit input has no randomly selected validation rows to vary.')
        trials=[{'seed':start+i,'overrides':{}} for i in range(count)]
    elif mode=='parameters':
        variants=payload.get('variants')
        if not isinstance(variants,list) or not 1<=len(variants)<=50 or any(not isinstance(v,dict) for v in variants):
            raise ValueError('Enter a JSON list of 1–50 hyperparameter objects.')
        trials=[{'seed':config['seed'],'overrides':v} for v in variants]
    else:raise ValueError('Choose split seeds or hyperparameter trials.')
    task=TaskConfig.from_dict(read(prep_root/'task.json'))
    recipe=Recipe.load(prep_root/'preprocessing/recipe.json')
    ensure_dependencies(original['model'],progress)
    experiment_id='opt-'+uuid.uuid4().hex;path=project.directory/'optimizations'/experiment_id/'experiment.json'
    experiment={'id':experiment_id,'source_model':name,'model':original['model'],'mode':mode,'status':'running',
                'reference':reference,'split_config':config,'recipe_fingerprint':recipe.fingerprint,
                'trials':[],'requested_trials':len(trials)}
    def save():write_json(path,experiment,overwrite=path.exists())
    save()
    try:
        for index,spec in enumerate(trials):
            check_cancelled();started=time.perf_counter()
            trial={'number':index+1,'seed':spec['seed'],'params':{**original['params'],**spec['overrides']},'status':'preparing'}
            experiment['trials'].append(trial);save()
            try:
                progress({'stage':'trial','message':f'Trial {index+1} of {len(trials)}','trial':index+1,'total':len(trials)})
                trial_reference=dict(reference)
                if mode=='seeds':
                    split_config=SplitConfig(**{**config,'seed':spec['seed']})
                    prepared=prepare_task(project.workspace,reference['dataset'],task,split_config=split_config,recipe=recipe,
                        validation_fraction=reference.get('validation_fraction',.2),loader_options=metadata.get('loader_options',{}),progress=progress)
                    trial_reference.update(run_id=prepared.run_id,task_id=prepared.task_id,
                        path=prepared.directory.relative_to(project.workspace.get(reference['dataset']).directory).as_posix(),
                        config=split_config.to_dict(),split_counts=prepared.split_counts,prepared_counts=prepared.prepared_counts)
                check_cancelled()
                trial['reference']=trial_reference
                model_name=store.default_name();trial['name']=model_name
                store.create(original['model'],name=model_name,params=trial['params'],input_reference=trial_reference)
                store.update(model_name,optimization_id=experiment_id)
                trial['status']='training';save()
                train_saved(project,model_name,max_bytes=models.service.max_bytes,progress=progress)
                check_cancelled()
                result=ModelInspection(project,model_name);detail=result.details();metrics=detail['metrics']
                trial['fit_seconds']=metrics.get('fit_seconds')
                trial['training_metrics']=metrics.get('training_metrics',{})
                trial['validation_metrics']=metrics.get('validation',{})
                trial['test_metrics']={}
                if 'test' in result.splits():
                    try:trial['test_metrics']=result.evaluate_test(max_bytes=models.service.max_bytes)['metrics']
                    except ValueError as exc:trial['evaluation_note']=str(exc)
                trial['status']='complete'
            except TrainingCancelled:
                trial['status']='cancelled';raise
            except Exception as exc:
                trial.update(status='failed',error=str(exc))
            finally:
                trial['elapsed_seconds']=time.perf_counter()-started;save()
            progress({'stage':'trial_complete','message':f'Trial {index+1} of {len(trials)}: {trial["status"]}', 'trial':index+1})
        experiment['status']='complete'
    except TrainingCancelled:
        experiment['status']='cancelled';raise
    except Exception:
        experiment['status']='failed';raise
    finally:save()
    return {'experiment_id':experiment_id,'name':name}
