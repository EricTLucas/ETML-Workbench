import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog='workbench models',description='Train, compare, predict and export models')
    commands = root.add_subparsers(dest='command',required=True)
    catalog = commands.add_parser('list')
    train = commands.add_parser('train')
    train.add_argument('dataset_id'); train.add_argument('task_id'); train.add_argument('preparation_run')
    candidates = train.add_mutually_exclusive_group()
    candidates.add_argument('--model',action='append',help='BACKEND:ALGORITHM; repeat for multiple candidates')
    candidates.add_argument('--config',help='JSON list of ModelConfig dictionaries, including hyperparameters')
    train.add_argument('--categorical',nargs='+',default=[])
    train.add_argument('--metric')
    train.add_argument('--max-features',type=int,default=50_000)
    test = commands.add_parser('test',help='Evaluate only the validation-selected winner on held-out test data')
    test.add_argument('dataset_id'); test.add_argument('task_id'); test.add_argument('training_run')
    for command in (train,test):
        command.add_argument('--workspace',default='datasets')
        command.add_argument('--max-rows',type=int,default=200_000)
        command.add_argument('--max-memory-mb',type=int,default=512)
    predict = commands.add_parser('predict')
    predict.add_argument('input'); predict.add_argument('--bundle',required=True)
    predict.add_argument('--output',required=True); predict.add_argument('--batch-size',type=int,default=50_000)
    predict.add_argument('--no-probabilities',action='store_true'); predict.add_argument('--strict',action='store_true')
    predict.add_argument('--loader-options',default='{}')
    export = commands.add_parser('export')
    export.add_argument('--bundle',required=True); export.add_argument('--output',required=True)
    export.add_argument('--format',choices=['bundle','native','onnx'],default='bundle')
    export.add_argument('--sample-data',help='Raw example dataset required for ONNX prediction parity checking')
    for command in (catalog,train,test,predict,export):
        command.add_argument('--json',action='store_true')
        command.add_argument('--quiet',action='store_true')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from models import ModelConfig, available_models
        if args.command=='list':
            payload = {'models':available_models()}
        elif args.command=='train':
            from training import train_models
            if args.config:
                if args.categorical:
                    raise ValueError('Set categorical_columns inside each configuration when using --config')
                values = json.loads(Path(args.config).read_text(encoding='utf-8'))
                if not isinstance(values,list):
                    raise ValueError('Model configuration file must contain a JSON list')
                configs = [ModelConfig(**value) for value in values]
            else:
                configs = []
                for value in args.model or ['sklearn:linear','sklearn:random_forest']:
                    parts = value.split(':')
                    if len(parts)!=2:
                        raise ValueError('Expected BACKEND:ALGORITHM')
                    configs.append(ModelConfig(parts[0],parts[1],categorical_columns=tuple(args.categorical)))
            def progress(event):
                if not args.quiet:
                    print(f"Training {event['candidate']}: {event['model']}",file=sys.stderr)
            result = train_models(args.workspace,args.dataset_id,args.task_id,args.preparation_run,
                                   configs=configs,metric=args.metric,max_rows=args.max_rows,
                                   max_bytes=args.max_memory_mb*1024**2,max_features=args.max_features,progress=progress)
            payload = {'status':'ok','training_run':result.run_id,'directory':str(result.directory),
                       'winner':result.winner,'bundle':str(result.bundle),'metric':result.metric,
                       'leaderboard':result.leaderboard,'test_used_for_selection':False}
        elif args.command=='test':
            from training import evaluate_test
            payload = evaluate_test(args.workspace,args.dataset_id,args.task_id,args.training_run,
                                     max_rows=args.max_rows,max_bytes=args.max_memory_mb*1024**2)
        elif args.command=='predict':
            from workflows.prediction import predict_file
            options = json.loads(args.loader_options)
            if not isinstance(options,dict):
                raise ValueError('--loader-options must be a JSON object')
            payload = predict_file(args.bundle,args.input,args.output,batch_size=args.batch_size,
                                    loader_options=options,probabilities=not args.no_probabilities,strict=args.strict)
        else:
            from exporting import export_bundle, export_native, export_onnx
            if args.format=='onnx':
                if not args.sample_data:
                    raise ValueError('ONNX export requires --sample-data for prediction parity verification')
                from eda_tool.loader import open_dataset
                from preprocessing.transforms.base import batches
                import pandas as pd
                source = open_dataset(args.sample_data)
                with batches(lambda:source.iter_batches(batch_size=256)) as iterator:
                    frame = next(iterator,pd.DataFrame(columns=list(source.schema())))
                path = export_onnx(args.bundle,args.output,sample_data=frame)
            else:
                path = (export_bundle if args.format=='bundle' else export_native)(args.bundle,args.output)
            payload = {'status':'ok','output':str(path),'format':args.format,
                       'raw_input_preprocessing_included':args.format=='bundle'}
        print(json.dumps(payload,indent=None if args.json else 2,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,TypeError,OSError,ImportError,ArithmeticError,KeyError,RuntimeError,AssertionError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
