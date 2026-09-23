from pathlib import Path
import json
from data import DatasetWorkspace
from data.manifest import resolve_inside, validate_name
from .artifacts import verify_artifacts, require_files
from .evaluation import validate_metric


def _task_root(workspace,dataset_id,task_id):
    ws = workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)
    return resolve_inside(ws.get(dataset_id).directory,f'tasks/{validate_name(task_id)}')


def latest_preparation(workspace,dataset_id,task_id):
    root = _task_root(workspace,dataset_id,task_id)/'runs'
    paths = [p for p in root.glob('run-*') if (p/'manifest.json').is_file()]
    if not paths:
        raise ValueError('No prepared runs; first run: workbench tasks prepare DATASET TASK')
    return max(paths,key=lambda p:((p/'manifest.json').stat().st_mtime_ns,p.name)).name


def training_runs(workspace,dataset_id,task_id):
    root = _task_root(workspace,dataset_id,task_id)/'training'
    result = []
    for path in root.glob('train-*'):
        if not (path/'manifest.json').exists():
            continue
        manifest = verify_artifacts(path,'training_run')
        require_files(manifest,['selection.json'])
        metadata = manifest['metadata']
        if metadata['dataset_id']!=dataset_id or metadata['task_id']!=task_id:
            raise ValueError('Training run identity mismatch')
        selection = json.loads((path/'selection.json').read_text(encoding='utf-8'))
        result.append({'run_id':path.name,'directory':str(path),'created_at':manifest.get('created_at'),
                       'mtime_ns':(path/'manifest.json').stat().st_mtime_ns,**metadata,**selection})
    return sorted(result,key=lambda row:(row['created_at'] or '',row['mtime_ns'],row['run_id']),reverse=True)


def resolve_run(workspace,dataset_id,task_id,run_id='latest'):
    runs = training_runs(workspace,dataset_id,task_id)
    if not runs:
        raise ValueError('No completed training runs for this task')
    if run_id=='latest':
        return runs[0]
    return next((r for r in runs if r['run_id']==run_id),None) or _missing(run_id)


def _missing(run_id):
    raise ValueError(f'Unknown training run: {run_id}')


def compare_runs(workspace,dataset_id,task_id,*,preparation_run=None,metric=None):
    runs = training_runs(workspace,dataset_id,task_id)
    if not runs:
        raise ValueError('No completed training runs to compare')
    preparation_run = preparation_run or runs[0]['preparation_run']
    chosen = [r for r in runs if r['preparation_run']==preparation_run]
    if not chosen:
        raise ValueError('No training runs for this preparation')
    if not any(r.get('selection_split')=='validation' for r in chosen):
        raise ValueError('No validation scores are available; training-only runs are not ranked')
    chosen=[r for r in chosen if r.get('selection_split')=='validation']
    if len({r['preparation_manifest_sha256'] for r in chosen})!=1:
        raise ValueError('Preparation changed between runs; these scores are not comparable')
    ws = workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)
    task_type = ws.get_task(dataset_id,task_id).task_type
    metric = metric or chosen[0]['metric']
    direction = validate_metric(metric,task_type)
    rows = []
    for run in chosen:
        for candidate in run['leaderboard']:
            score = candidate['validation'].get(metric)
            rows.append({**candidate,'run_id':run['run_id'],'label':run.get('label'),
                         'score':score,'bundle':str(Path(run['directory'])/'candidates'/candidate['candidate'])})
    rows.sort(key=lambda r:(r['score'] is None,0 if r['score'] is None else
                          (-r['score'] if direction=='max' else r['score']),r['run_id'],r['candidate']))
    return {'preparation_run':preparation_run,'metric':metric,'direction':direction,
            'selection_split':'validation','leaderboard':rows,
            'excluded_runs_from_other_preparations':len(runs)-len(chosen)}
