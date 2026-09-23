"""Project prediction, results, details and export without opaque run IDs."""
import json
from workflows.project_models import ProjectModels
from workflows.model_inspection import ModelInspection


def _print_values(values,indent='  '):
    for key,value in values.items():
        formatted=f'{value:.6g}' if isinstance(value,float) else json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value)
        print(indent+str(key)+': '+formatted)


def show_metrics(inspection):
    from workflows.model_inspection import read_json
    metrics=read_json(inspection.bundle/'metrics.json')
    print('\nResults — '+inspection.name)
    if metrics.get('validation'):
        print('Validation metrics:'); _print_values(metrics['validation'])
        if metrics.get('validation_details'):
            _print_values(metrics['validation_details'])
    elif inspection.kind=='model_bundle':
        print('No validation set. No validation accuracy, model ranking, or validation early stopping.')
    if metrics.get('training_metrics'):
        print('Training metrics (in-sample; not a held-out accuracy estimate):')
        _print_values(metrics['training_metrics'])
        if not metrics.get('validation') and metrics.get('training_details'): _print_values(metrics['training_details'])
    if inspection.kind=='exploration_model': _print_values(metrics)
    for key in ('baseline','protocol','fit_seconds','training_rows','validation_rows','best_epoch','epochs_completed'):
        if key in metrics and metrics[key] is not None: _print_values({key:metrics[key]})
    stored=inspection.record.get('metrics',{})
    if stored.get('baseline_score') is not None:
        _print_values({'validation_baseline_score':stored['baseline_score'],
                       'improvement_over_baseline':stored.get('improvement_over_baseline')})
    if inspection.record.get('test_evaluation'):
        print('Saved test metrics (held-out):'); _print_values(inspection.record['test_evaluation']['metrics'])


def run(project,args,*,name=None,choose=None,ask=None):
    from .project_cli import _choose,_ask
    choose=choose or _choose; ask=ask or _ask
    from workflows.model_lifecycle import saved_record,train_saved,checkpoint_path
    def new_model():
        from .model_wizard import run as model_run
        return model_run(project,args,choose=choose,ask=ask,start_new=True)
    def train_more(record,additional=None):
        def progress(event):
            if event.get('stage')=='epoch':
                label='validation_loss' if 'validation_loss' in event else 'train_loss'
                print(f"Epoch {event['epoch']}: {label.replace('_',' ')} {event[label]:.6g}")
        features=None
        if not isinstance(record['input'],dict) and record['model'].startswith('sklearn:') and not record.get('features'):
            features=[c.strip() for c in ask('Original feature columns, comma separated: ').split(',') if c.strip()]
        return train_saved(project,record['name'],additional_epochs=additional,features=features,
            max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2,progress=progress)
    while True:
        records=ProjectModels(project).list()
        if name is None:
            index=choose('Saved models',[r['name']+' — '+r['model']+' ('+r['status']+')' for r in records]+['Finish','Create a new model'])
            if index==len(records): return 0
            if index==len(records)+1: return new_model()
            name=records[index]['name']
        record=saved_record(project,name)
        if record['status']!='complete':
            print(f"\n{name}: {record['status']} — a saved configuration, not a completed model.")
            if record.get('error'): print('Last error: '+record['error'])
            from models.catalog import CATALOG
            entry=CATALOG.get(record['model'])
            if entry and entry.extra: print(f'Optional dependency: python -m pip install -e ".[{entry.extra}]"')
            actions=['Retry training with saved settings','View saved settings','Choose another model','Create a new model','Finish']
            if checkpoint_path(record): actions.append('Continue from saved checkpoint')
            action=choose('Saved configuration',actions)
            if action==4: return 0
            if action==3: return new_model()
            if action==2: name=None; continue
            if action==1: print(json.dumps(record,indent=2)); continue
            try:
                additional=int(ask('Additional epochs [10]: ') or '10') if action==5 else None
                train_more(record,additional)
            except (ValueError,TypeError,OSError,ImportError,RuntimeError,KeyError) as exc:
                print('Training did not finish: '+str(exc))
            continue
        inspection=ModelInspection(project,name)
        show_metrics(inspection)
        while True:
            record=saved_record(project,name)
            choices=['Predict a dataset row','Enter a custom example (JSON)',
                'Model details and split information','Export model / data','Show metrics again','Choose another model','Finish']
            if inspection.kind=='model_bundle': choices.append('Evaluate held-out test split')
            new_index=len(choices); choices.append('Create a new model')
            resume_index=None
            if record['model'] in {'pytorch:mlp','tensorflow:mlp'} and checkpoint_path(record):
                resume_index=len(choices); choices.append('Continue training (additional epochs)')
            action=choose('\nPrediction and model tools',choices)
            try:
                if action==new_index: return new_model()
                if resume_index is not None and action==resume_index:
                    from pathlib import Path
                    state=json.loads((Path(checkpoint_path(record))/'state.json').read_text(encoding='utf-8'))
                    print(f"Continue from the last completed checkpoint: epoch {state['epoch']} (optimizer state included).")
                    if state['stopped_early']: print('Continuing resets the previous early-stopping patience counter.')
                    additional=int(ask('Additional epochs [10]: ') or '10')
                    record=train_more(record,additional)
                    inspection=ModelInspection(project,name); show_metrics(inspection); continue
                if action==7 and inspection.kind=='model_bundle':
                    report=inspection.evaluate_test(max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2)
                    print('Test metrics (held-out):'); _print_values(report['metrics']); continue
                if action==6: return 0
                if action==5: name=None; break
                if action==4: show_metrics(inspection); continue
                if action==2:
                    print(json.dumps(inspection.details(),indent=2,ensure_ascii=False)); continue
                if action==3:
                    include=choose('Export contents',['Model bundle and details','Model, details and associated data (includes available test data)'])==1
                    destination=ask('New export folder: ').strip('"')
                    if not destination: print('No destination entered.'); continue
                    path=inspection.export(destination,include_data=include,max_bytes=max(1,args.max_memory_mb)*1024**2)
                    print('Export saved: '+str(path)); continue
                if action==1:
                    values=json.loads(ask('Raw input values as a JSON object: '))
                    result=inspection.store.predict(name,values)
                    print(json.dumps(result,indent=2,ensure_ascii=False)); continue
                sources=inspection.splits()
                if not sources: print('No saved rows are available.'); continue
                splits=list(sources)
                split=splits[choose('Dataset split',[f'{s} ({sources[s]["rows"]} rows)' for s in splits])] if len(splits)>1 else splits[0]
                raw=ask(f'Row number, 1–{sources[split]["rows"]} [1]: ')
                number=int(raw or '1')
                result=inspection.predict_row(number,split)
                print(f'\n{split.capitalize()} row {number}')
                print('Values:'); _print_values(result['values'])
                print('Prediction: '+json.dumps(result.get('prediction'),ensure_ascii=False))
                print('True value: '+json.dumps(result['actual'],ensure_ascii=False) if 'actual' in result else 'True value: unavailable (no observed target for this example).')
                for key in ('status','correct','residual','probabilities','protocol','actual_meaning','observed_item'):
                    if key in result: _print_values({key:result[key]})
            except (ValueError,TypeError,OSError,ImportError,RuntimeError,KeyError) as exc:
                print('Could not complete that action: '+str(exc))
