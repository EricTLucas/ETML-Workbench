import argparse
import json
from pathlib import Path
import sys
from .display import command, emit


def parser():
    root = argparse.ArgumentParser(prog='workbench models',description='Train, tune, compare, resume, predict and export')
    commands = root.add_subparsers(dest='command',required=True)
    catalog = commands.add_parser('list',help='Available models and capabilities')
    train = commands.add_parser('train',help='Compare candidates; preparation defaults to latest')
    search = commands.add_parser('search',help='Budgeted random or grid search')
    for p in (train,search):
        p.add_argument('dataset_id'); p.add_argument('task_id')
        p.add_argument('preparation_run',nargs='?',default='latest')
        p.add_argument('--categorical',nargs='+',default=[])
        p.add_argument('--metric'); p.add_argument('--label')
        p.add_argument('--max-features',type=int,default=50_000)
        p.add_argument('--params',default='{}',help='JSON fixed parameters; single model only')
        p.add_argument('--epochs',type=int,help='Total epochs for neural models')
        p.add_argument('--batch-size',type=int,help='Neural training batch size')
        p.add_argument('--learning-rate',type=float)
        p.add_argument('--patience',type=int,help='Neural validation-loss stopping patience')
        p.add_argument('--hidden-sizes',type=int,nargs='+')
        p.add_argument('--early-stopping-rounds',type=int,help='XGBoost validation stopping patience')
        p.add_argument('--seed',type=int,default=42)
    candidates = train.add_mutually_exclusive_group()
    candidates.add_argument('--model',action='append',help='BACKEND:ALGORITHM; repeat for candidates')
    candidates.add_argument('--config',help='JSON list of model configurations')
    search.add_argument('--model',default='sklearn:random_forest')
    search.add_argument('--space',help='JSON file mapping parameters to lists of choices')
    search.add_argument('--method',choices=['grid','random'],default='random')
    search.add_argument('--trials',type=int,default=10)
    search.add_argument('--max-trials',type=int,default=100)
    resume = commands.add_parser('resume',help='Continue a neural checkpoint to a total epoch target')
    resume.add_argument('checkpoint'); resume.add_argument('--epochs',type=int,required=True)
    resume.add_argument('--reset-patience',action='store_true'); resume.add_argument('--label')
    test = commands.add_parser('test',help='Evaluate held-out test data after tuning is finished')
    test.add_argument('dataset_id'); test.add_argument('task_id')
    test.add_argument('training_run',nargs='?',default='latest')
    test.add_argument('--candidate',help='Defaults to the run winner')
    runs = commands.add_parser('runs',help='List completed experiments')
    compare = commands.add_parser('compare',help='Rank validation scores on one preparation')
    history = commands.add_parser('history',help='Inspect epoch/boosting history and optionally plot it')
    checkpoints = commands.add_parser('checkpoints',help='Find resumable neural checkpoints')
    for p in (runs,compare,history,checkpoints):
        p.add_argument('dataset_id'); p.add_argument('task_id')
    compare.add_argument('--preparation-run'); compare.add_argument('--metric')
    history.add_argument('training_run',nargs='?',default='latest')
    history.add_argument('--candidate'); history.add_argument('--plot',help='New PNG for learning curves')
    predict = commands.add_parser('predict')
    predict.add_argument('input'); predict.add_argument('--output',required=True)
    predict.add_argument('--batch-size',type=int,default=50_000)
    predict.add_argument('--no-probabilities',action='store_true'); predict.add_argument('--strict',action='store_true')
    predict.add_argument('--loader-options',default='{}')
    export = commands.add_parser('export')
    export.add_argument('--output',required=True)
    export.add_argument('--format',choices=['bundle','native','onnx'],default='bundle')
    export.add_argument('--sample-data',help='Raw sample for ONNX parity verification')
    for p in (predict,export):
        selector = p.add_mutually_exclusive_group(required=True)
        selector.add_argument('--bundle'); selector.add_argument('--dataset',dest='dataset_id')
        p.add_argument('--task',dest='task_id'); p.add_argument('--run',default='latest')
        p.add_argument('--candidate')
    for p in (train,search,resume,test):
        p.add_argument('--max-rows',type=int,default=200_000)
        p.add_argument('--max-memory-mb',type=int,default=512)
    for p in (catalog,train,search,resume,test,runs,compare,history,checkpoints,predict,export):
        p.add_argument('--workspace',default='datasets')
        p.add_argument('--json',action='store_true'); p.add_argument('--quiet',action='store_true')
    return root


def _object(value):
    result = json.loads(value)
    if not isinstance(result,dict): raise ValueError('Expected a JSON object')
    return result


def _configs(args):
    from models import ModelConfig
    neural_keys = ('epochs','batch_size','learning_rate','patience','hidden_sizes')
    if getattr(args,'config',None):
        if args.params!='{}' or args.categorical or any(getattr(args,k) is not None for k in (*neural_keys,'early_stopping_rounds')):
            raise ValueError('Put model settings inside --config when using a configuration file')
        values = json.loads(Path(args.config).read_text(encoding='utf-8'))
        if not isinstance(values,list): raise ValueError('Configuration must be a list of dictionaries')
        return [ModelConfig(**v) for v in values]
    names = [args.model] if args.command=='search' else args.model or ['sklearn:linear','sklearn:random_forest']
    params = _object(args.params)
    if params and len(names)!=1: raise ValueError('--params requires one --model; use --config for multiple parameter sets')
    result = []
    for name in names:
        parts = name.split(':')
        if len(parts)!=2: raise ValueError('Expected BACKEND:ALGORITHM')
        values = dict(params)
        if parts[0] in {'pytorch','tensorflow'}:
            for key in neural_keys:
                if getattr(args,key) is not None: values[key]=getattr(args,key)
        elif parts[0]=='xgboost':
            if args.early_stopping_rounds is not None:
                values['early_stopping_rounds']=args.early_stopping_rounds
            if args.learning_rate is not None:
                values['learning_rate']=args.learning_rate
        result.append(ModelConfig(*parts,params=values,seed=args.seed,categorical_columns=tuple(args.categorical)))
    if any(getattr(args,k) is not None for k in neural_keys if k!='learning_rate') and not any(c.backend in {'pytorch','tensorflow'} for c in result):
        raise ValueError('Epoch/batch/patience flags require a neural model')
    if args.learning_rate is not None and not any(c.backend in {'pytorch','tensorflow','xgboost'} for c in result):
        raise ValueError('--learning-rate requires a neural model or XGBoost')
    if args.early_stopping_rounds is not None and not any(c.backend=='xgboost' for c in result):
        raise ValueError('--early-stopping-rounds requires XGBoost')
    return result


def _bundle(args):
    if args.bundle:
        if args.task_id or args.candidate or args.run!='latest':
            raise ValueError('Use either --bundle or dataset/task/run selection')
        return Path(args.bundle)
    if not args.task_id: raise ValueError('--dataset requires --task')
    from training.experiments import resolve_run
    from data.manifest import resolve_inside, validate_name
    run = resolve_run(args.workspace,args.dataset_id,args.task_id,args.run)
    return resolve_inside(Path(run['directory']),'candidates/'+validate_name(args.candidate or run['winner']))


def _training_steps(args,result):
    from training.artifacts import verify_artifacts
    origin = verify_artifacts(result.directory,'training_run')['metadata']
    dataset,task = origin['dataset_id'],origin['task_id']
    common = ['--workspace',args.workspace]
    steps = [('Compare validation scores',command('workbench','models','compare',dataset,task,
              '--preparation-run',origin['preparation_run'],*common)),
             ('Inspect the winning model history',command('workbench','models','history',dataset,task,result.run_id,*common)),
             ('When tuning is finished, evaluate the test set',command('workbench','models','test',dataset,task,result.run_id,*common)),
             ('Export the prediction bundle',command('workbench','models','export','--bundle',result.bundle,
              '--output',str(Path('exports')/result.run_id),*common))]
    for row in result.leaderboard:
        if row.get('checkpoint'):
            steps.append((f"Plot {row['candidate']} learning curves",command('workbench','models','history',dataset,task,
                result.run_id,'--candidate',row['candidate'],'--plot',str(Path('plots')/(result.run_id+'-'+row['candidate']+'.png')),*common)))
            parts = ['workbench','models','resume',row['checkpoint'],'--epochs',row['training']['epoch']+10,*common]
            if row['training'].get('stopped_early'): parts.append('--reset-patience')
            steps.append((f"Continue {row['candidate']} for up to 10 more epochs",command(*parts)))
    return steps


def main(argv=None):
    args = parser().parse_args(argv)
    latest_checkpoint = None
    try:
        from models import available_models
        from training.experiments import latest_preparation,training_runs,compare_runs,resolve_run
        steps,table = [],None
        def progress(event):
            nonlocal latest_checkpoint
            if event.get('checkpoint'): latest_checkpoint=event['checkpoint']
            if not args.quiet:
                if event['stage']=='epoch':
                    print(f"{event['candidate']} epoch {event['epoch']}: train loss {event['train_loss']:.6g}, validation loss {event['validation_loss']:.6g}",file=sys.stderr)
                elif event['stage']=='training':
                    print(f"Training {event['candidate']}: {event['model']}",file=sys.stderr)
                elif event['stage']=='boosting_complete':
                    print(f"Boosting complete: {event['boosting_rounds']} rounds; best iteration {event['best_iteration']}",file=sys.stderr)
        if args.command=='list':
            payload = {'models':available_models()}
            table = (['Model','Epochs','Resume','ONNX'],[[name,v['epochs'],v['resumable'],v['onnx_model_only']] for name,v in payload['models'].items()])
        elif args.command in {'train','search','resume'}:
            from training import train_models
            kwargs = {'max_rows':args.max_rows,'max_bytes':args.max_memory_mb*1024**2,'progress':progress,'label':args.label}
            if args.command=='resume':
                from training.checkpoints import resume_training
                result = resume_training(args.workspace,args.checkpoint,epochs=args.epochs,reset_patience=args.reset_patience,**kwargs)
            else:
                prep = latest_preparation(args.workspace,args.dataset_id,args.task_id) if args.preparation_run=='latest' else args.preparation_run
                configs = _configs(args)
                kwargs.update(metric=args.metric,max_features=args.max_features)
                if args.command=='search':
                    from training.search import search_models
                    defaults = {'sklearn':{'n_estimators':[100,200],'max_depth':[None,6,12],'min_samples_leaf':[1,3]},
                        'xgboost':{'n_estimators':[100,300],'max_depth':[3,6],'learning_rate':[0.03,0.1]},
                        'pytorch':{'hidden_sizes':[[32],[64,32]],'learning_rate':[0.001,0.01]},
                        'tensorflow':{'hidden_sizes':[[32],[64,32]],'learning_rate':[0.001,0.01]}}
                    if not args.space and configs[0].backend=='sklearn' and configs[0].algorithm!='random_forest':
                        raise ValueError('Supply --space for this model')
                    space = json.loads(Path(args.space).read_text(encoding='utf-8')) if args.space else defaults.get(configs[0].backend)
                    result = search_models(args.workspace,args.dataset_id,args.task_id,prep,base=configs[0],space=space,
                        method=args.method,trials=args.trials,seed=args.seed,max_trials=args.max_trials,**kwargs)
                else:
                    result = train_models(args.workspace,args.dataset_id,args.task_id,prep,configs=configs,**kwargs)
            payload = {'status':'ok','training_run':result.run_id,'directory':str(result.directory),
                       'preparation_run':json.loads((result.directory/'manifest.json').read_text(encoding='utf-8'))['metadata']['preparation_run'],
                       'winner':result.winner,'bundle':str(result.bundle),'metric':result.metric,
                       'leaderboard':result.leaderboard,'test_used_for_selection':False}
            table = (['Candidate','Model','Validation '+result.metric],[[r['candidate'],
                r['model']['backend']+':'+r['model']['algorithm'],f"{r['validation'][result.metric]:.6g}"] for r in result.leaderboard])
            steps = _training_steps(args,result)
        elif args.command=='runs':
            payload = {'runs':training_runs(args.workspace,args.dataset_id,args.task_id)}
            table = (['Run','Label','Preparation','Winner'],[[r['run_id'],r.get('label') or '',r['preparation_run'],r['winner']] for r in payload['runs']])
        elif args.command=='compare':
            payload = compare_runs(args.workspace,args.dataset_id,args.task_id,preparation_run=args.preparation_run,metric=args.metric)
            table = (['Run','Candidate','Model',payload['metric']],[[r['run_id'],r['candidate'],
                r['model']['backend']+':'+r['model']['algorithm'],f"{r['score']:.6g}" if r['score'] is not None else 'undefined'] for r in payload['leaderboard']])
            best = next((r for r in payload['leaderboard'] if r['score'] is not None),None)
            if best:
                steps = [('When tuning is finished, test this candidate',command('workbench','models','test',args.dataset_id,args.task_id,
                    best['run_id'],'--candidate',best['candidate'],'--workspace',args.workspace))]
        elif args.command=='checkpoints':
            from training.experiments import _task_root
            from training.checkpoints import load_checkpoint
            rows = []
            for path in (_task_root(args.workspace,args.dataset_id,args.task_id)/'checkpoints').glob('train-*/candidate-*/epoch-*'):
                if not (path/'manifest.json').exists(): continue
                context = load_checkpoint(path)
                state = json.loads((path/'state.json').read_text(encoding='utf-8'))
                rows.append({'path':str(path),'epoch':state['epoch'],'backend':context['config']['backend'],'stopped_early':state['stopped_early']})
            rows.sort(key=lambda r:Path(r['path']).stat().st_mtime_ns,reverse=True)
            payload = {'checkpoints':rows}
            table = (['Epoch','Backend','Checkpoint'],[[r['epoch'],r['backend'],r['path']] for r in rows])
            if rows:
                first = rows[0]
                extra = ['--reset-patience'] if first['stopped_early'] else []
                steps = [('Resume the latest checkpoint',command('workbench','models','resume',first['path'],'--epochs',first['epoch']+10,'--workspace',args.workspace,*extra))]
        elif args.command=='history':
            from training.history import read_history, plot_history
            run = resolve_run(args.workspace,args.dataset_id,args.task_id,args.training_run)
            payload = read_history(run,args.candidate)
            if args.plot: payload['plot']=str(plot_history(payload,args.plot))
            rows = payload['rows']
            table = (list(rows[0]),[list(r.values()) for r in rows]) if rows else (['History'],[['This model has no epoch/round history.']])
        elif args.command=='test':
            from training import evaluate_test
            run = resolve_run(args.workspace,args.dataset_id,args.task_id,args.training_run)
            payload = evaluate_test(args.workspace,args.dataset_id,args.task_id,run['run_id'],candidate=args.candidate,
                max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2)
            bundle = Path(run['directory'])/'candidates'/payload['candidate']
            steps = [('Export this model',command('workbench','models','export','--bundle',bundle,
                     '--output',str(Path('exports')/run['run_id']/payload['candidate'])))]
        elif args.command=='predict':
            from workflows.prediction import predict_file
            payload = predict_file(_bundle(args),args.input,args.output,batch_size=args.batch_size,
                loader_options=_object(args.loader_options),probabilities=not args.no_probabilities,strict=args.strict)
        else:
            from exporting import export_bundle, export_native, export_onnx
            bundle = _bundle(args)
            if args.format=='onnx':
                if not args.sample_data: raise ValueError('ONNX export requires --sample-data for prediction parity verification')
                from eda_tool.loader import open_dataset
                from preprocessing.transforms.base import batches
                import pandas as pd
                source = open_dataset(args.sample_data)
                with batches(lambda:source.iter_batches(batch_size=256)) as iterator:
                    frame = next(iterator,pd.DataFrame(columns=list(source.schema())))
                path = export_onnx(bundle,args.output,sample_data=frame)
            else:
                path = (export_bundle if args.format=='bundle' else export_native)(bundle,args.output)
            payload = {'status':'ok','output':str(path),'format':args.format,'raw_input_preprocessing_included':args.format=='bundle'}
        emit(payload,args,steps=steps,table=table)
        return 0
    except KeyboardInterrupt:
        print('Training interrupted. Completed epoch checkpoints are preserved.',file=sys.stderr)
        if latest_checkpoint:
            print('Last checkpoint: '+latest_checkpoint,file=sys.stderr)
            state = json.loads((Path(latest_checkpoint)/'state.json').read_text(encoding='utf-8'))
            extra = ['--reset-patience'] if state['stopped_early'] else []
            print(command('workbench','models','resume',latest_checkpoint,'--epochs',max(state['epoch']+1,getattr(args,'epochs',None) or 30),
                          '--workspace',args.workspace,*extra),file=sys.stderr)
        return 130
    except (ValueError,TypeError,OSError,ImportError,ArithmeticError,KeyError,RuntimeError,AssertionError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        if latest_checkpoint: print('Last recoverable checkpoint: '+latest_checkpoint,file=sys.stderr)
        return 2
