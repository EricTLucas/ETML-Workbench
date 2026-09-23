"""Human-sized model selection and settings flow."""
import json
from pathlib import Path
from models.catalog import CATALOG,describe,suggest
from workflows.project_models import ProjectModels,train_named_tabular


def _settings(entry,choose,ask,task=None):
    options=dict(entry.defaults)
    print('\nCommon settings: '+json.dumps(options))
    action=choose('Training settings',['Use these defaults','Edit settings one at a time','Enter parameter overrides as JSON'])
    if action==2:
        updates=json.loads(ask('JSON object: '))
        if not isinstance(updates,dict): raise ValueError('Expected a JSON object')
        return {**options,**updates}
    if action==1:
        for key,default in list(options.items()):
            choices={'optimizer':['adam','adamw','sgd','rmsprop'],'regularizer':['none','l1','l2'],
                     'loss':['auto','cross_entropy'] if (task or entry.tasks[0])=='classification' or entry.modality=='text' else ['auto','mse','mae','huber']}
            if key in choices:
                print('Common default: '+str(default))
                options[key]=choices[key][choose(key,choices[key])]
            else:
                value=ask(f'{key} [{json.dumps(default)}] (Enter keeps it): ')
                if value:
                    try: options[key]=json.loads(value)
                    except json.JSONDecodeError:
                        if isinstance(default,str): options[key]=value
                        else: raise ValueError(f'{key} requires a JSON number, list, boolean or null') from None
    return options


def _select(entries,task,modality,choose):
    recommended={e.key for e in suggest(task,modality)}
    entries=sorted(entries,key=lambda e:(e.key not in recommended,e.name))
    print('\nSuggestions are starting points based on task/input type, not measured winners.')
    while True:
        entry=entries[choose('Choose a model',[e.name+(' — suggested' if e.key in recommended else '') for e in entries])]
        print('\n'+describe(entry.key))
        action=choose('Next',['Use this model','See another model'])
        if action==0: return entry


def _input(project,entry,task,args,choose,ask):
    from data.model_inputs import prepare_model_input,load_model_input
    existing=[]
    for path in (project.directory/'inputs').glob('input-*'):
        _,info=load_model_input(path)
        if info['modality']==entry.modality and info.get('task')==task: existing.append(path)
    action=choose('Model input',['Import a new input']+[p.name for p in existing])
    if action: return existing[action-1]
    roles={'image':['image','target'],'text':['text']+(['target'] if task!='language_modeling' else []),
           'time_series':['time','value'],'recommendation':['user','item']}
    print('Use a CSV/parquet table. Images use a path column; the image files are copied into the project.')
    source=choose('Input source',['Local table','Hugging Face dataset','Load an input configuration JSON'])
    if source==2:
        spec=json.loads(Path(ask('Configuration path: ')).read_text(encoding='utf-8'))
        if spec.get('modality')!=entry.modality or spec.get('task',task)!=task:
            raise ValueError('Input configuration must match the selected model modality/task')
        spec['task']=task
    else:
        spec={'modality':entry.modality,'task':task}
        if source==0:
            spec['train']=ask('Training/source table path: ').strip('"')
            val=ask('Presplit validation table (Enter to split automatically): ').strip('"')
            if val:
                spec['validation']=val
                test=ask('Presplit test table (optional): ').strip('"')
                if test: spec['test']=test
        else:
            spec['source']={'provider':'huggingface','dataset':ask('Hub dataset (OWNER/NAME): '),
                'config':ask('Configuration (optional): ') or None,'split':ask('Split [train]: ') or 'train',
                'revision':ask('Revision/commit (optional): ') or None}
        spec['columns']={role:ask(f'Column for {role}: ') for role in roles[entry.modality]}
        if entry.modality=='recommendation':
            weight=ask('Positive interaction weight column (Enter for weight=1): ')
            if weight: spec['columns']['weight']=weight
        if not spec.get('validation'):
            fractions=ask('Train, validation, test fractions [0.7, 0.15, 0.15]: ')
            if fractions: spec['fractions']=[float(v.strip()) for v in fractions.split(',')]
            print('Chronological split.' if entry.modality=='time_series' else 'Seeded split; recommendations split unique user–item pairs.')
    path=prepare_model_input(project,spec,max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2)
    print('Input saved: '+str(path))
    return path


def run(project,args,*,choose=None,ask=None,start_new=False):
    from .project_cli import _choose,_ask
    choose=choose or _choose; ask=ask or _ask
    store=ProjectModels(project); records=store.list()
    if records:
        print('\nSaved models:')
        for record in records: print(f"  {record['name']} — {record['model']} ({record['status']})")
    action=0 if start_new else choose('\nModels',['Create and train a model','Finish for now','Inspect, retry or continue a saved model'])
    if action==1: return 0
    if action==2:
        from .prediction_wizard import run as prediction_run
        return prediction_run(project,args,choose=choose,ask=ask)
    modality=['tabular','tabular','image','text','time_series','recommendation'][choice:=choose('What are you modeling?',
        ['Tabular prediction (classification/regression)','Tabular exploration (clusters, projections, anomalies)',
         'Images','Text','Time series','Recommendations'])]
    service=None; task=None
    if choice==0:
        from workflows.project_preparation import ProjectPreparation
        service=ProjectPreparation(project,getattr(args,'dataset_id',None))
        if not service.state().get('split'):
            print('Prepare and split the tabular dataset first: workbench '+project.name+' --continue')
            return 0
        task=service.task().task_type
    elif choice==1: task=['clustering','projection','anomaly'][choose('Exploration task',['Find clusters','Reduce dimensions','Find anomalies'])]
    elif choice==2: task='classification'
    elif choice==3: task=['classification','language_modeling','text_to_text'][choose('Text task',['Classify text (BERT)','Continue/generate text (GPT-2)','Map input text to target text (T5)'])]
    elif choice==4: task='forecasting'
    else: task='recommendation'
    entry=_select([e for e in CATALOG.values() if e.modality==modality and task in e.tasks],task,modality,choose)
    options=_settings(entry,choose,ask,task)
    name=ask(f'Model name [{store.default_name()}]: ') or store.default_name()
    source=None; features=None
    if modality!='tabular': source=_input(project,entry,task,args,choose,ask)
    elif choice==1:
        source=ask('CSV/parquet file to explore: ').strip('"')
        print('All selected rows will be fitted for exploration. Exclude labels, IDs and any held-out test rows.')
        selected=ask('Feature columns, comma separated (required): ')
        features=[s.strip() for s in selected.split(',') if s.strip()]
        if not features: raise ValueError('Select feature columns explicitly')
    if choose(f'Train {name} using {entry.name}?',['Train now','Keep data and return'])==1: return 0
    def progress(event):
        if event.get('stage')=='epoch':
            label='validation_loss' if 'validation_loss' in event else 'train_loss'
            print(f"Epoch {event['epoch']}: {label.replace('_',' ')} {event[label]:.5g}")
        elif event.get('stage')=='training': print('Training '+event.get('model','model')+'...')
    if choice==0:
        _,result=train_named_tabular(project,service,entry.key,options,name=name,max_rows=args.max_rows,
            max_bytes=args.max_memory_mb*1024**2,progress=progress)
    else:
        record=store.create(entry.key,name=name,params=options,input_reference=str(source))
        if features: store.update(name,features=features)
        destination=store.root/name/'bundle'
        try:
            store.update(name,status='training')
            if choice==1:
                from training.exploration import train_exploration
                result=train_exploration(source,entry.key,destination,features=features,params=options,
                    max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2)
            else:
                from training.specialized import train_specialized
                result=train_specialized(source,entry.key,destination,params=options,max_bytes=args.max_memory_mb*1024**2,progress=progress)
            store.update(name,status='complete',bundle=destination.relative_to(project.directory).as_posix(),metrics=result)
        except BaseException as exc:
            store.update(name,status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
            raise
    print('\nSaved model: '+name)
    from .prediction_wizard import run as prediction_run
    return prediction_run(project,args,name=name,choose=choose,ask=ask)
