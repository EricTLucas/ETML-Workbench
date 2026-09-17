import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog='workbench tasks', description='Define prediction tasks and prepare leakage-safe splits')
    commands = root.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('dataset_id'); create.add_argument('task_id')
    create.add_argument('--target', required=True)
    create.add_argument('--type', dest='task_type', required=True, choices=['classification','regression'])
    create.add_argument('--exclude', nargs='+', default=[])
    create.add_argument('--missing-target', choices=['error','drop'], default='error')
    show = commands.add_parser('show')
    show.add_argument('dataset_id'); show.add_argument('task_id')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('dataset_id'); prepare.add_argument('task_id')
    prepare.add_argument('--strategy', choices=['random','stratified','group','chronological'])
    prepare.add_argument('--train', type=float, default=.7)
    prepare.add_argument('--validation', type=float, default=.15)
    prepare.add_argument('--test', type=float, default=.15)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--group-column'); prepare.add_argument('--time-column'); prepare.add_argument('--time-format')
    prepare.add_argument('--max-classes', type=int, default=10_000)
    prepare.add_argument('--batch-size', type=int, default=50_000)
    recipes = prepare.add_mutually_exclusive_group()
    recipes.add_argument('--recipe', help='Reviewed proposal filename inside dataset recipes/')
    recipes.add_argument('--recipe-file', help='Approved raw Recipe JSON file')
    prepare.add_argument('--source-version', default='raw')
    prepare.add_argument('--allow-processed-source', action='store_true', help='Confirm the processed source used only fixed cleanup, not fitted preprocessing')
    prepare.add_argument('--loader-options', default='{}')
    inspect = commands.add_parser('inspect-run')
    inspect.add_argument('dataset_id'); inspect.add_argument('task_id'); inspect.add_argument('run_id')
    inspect.add_argument('--verify', action='store_true')
    for command in (create,show,prepare,inspect):
        command.add_argument('--workspace', default='datasets')
        command.add_argument('--json', action='store_true')
        command.add_argument('--quiet', action='store_true')
    return root


def main(argv=None):
    from data import DatasetWorkspace
    from tasks import TaskConfig
    from splitting import SplitConfig
    from preprocessing import Recipe
    from workflows.preparation import prepare_task
    from workflows.preprocessing import PreprocessingWorkflow
    args = parser().parse_args(argv)
    try:
        workspace = DatasetWorkspace(args.workspace)
        if args.command == 'create':
            config = TaskConfig(args.task_id,args.target,args.task_type,tuple(args.exclude),args.missing_target)
            path = workspace.create_task(args.dataset_id,config)
            payload = {'status':'ok','task':config.to_dict(),'directory':str(path)}
        elif args.command == 'show':
            payload = workspace.get_task(args.dataset_id,args.task_id).to_dict()
        elif args.command == 'inspect-run':
            payload = asdict(workspace.get_task_run(args.dataset_id,args.task_id,args.run_id,verify=args.verify))
        else:
            task = workspace.get_task(args.dataset_id,args.task_id)
            strategy = args.strategy or ('stratified' if task.task_type == 'classification' else 'random')
            config = SplitConfig(strategy,args.train,args.validation,args.test,args.seed,
                                 args.group_column,args.time_column,args.time_format,args.max_classes)
            loader_options = json.loads(args.loader_options)
            if not isinstance(loader_options,dict):
                raise ValueError('--loader-options requires a JSON object')
            recipe = Recipe.load(args.recipe_file) if args.recipe_file else Recipe()
            if args.recipe:
                proposal = PreprocessingWorkflow(workspace).load_proposal(args.dataset_id,args.recipe)
                if proposal.source_version != args.source_version:
                    raise ValueError('Proposal source version differs from the task preparation source')
                recipe = proposal.recipe
                if args.loader_options == '{}':
                    loader_options = proposal.loader_options
            def progress(event):
                if not args.quiet:
                    print(f"{event['stage']}: {event['rows']:,} rows",file=sys.stderr)
            result = prepare_task(workspace,args.dataset_id,task,split_config=config,recipe=recipe,
                                  source_version=args.source_version,allow_processed_source=args.allow_processed_source,
                                  loader_options=loader_options,batch_size=args.batch_size,progress=progress)
            payload = {'status':'ok','dataset_id':result.dataset_id,'task_id':result.task_id,
                       'run_id':result.run_id,'directory':str(result.directory),
                       'split_counts':result.split_counts,'prepared_counts':result.prepared_counts,
                       'features':list(result.feature_columns),'target':result.target,
                       'fitted':str(result.directory/'preprocessing'/'fitted.json')}
        print(json.dumps(payload,indent=None if args.json else 2,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,TypeError,OSError,ImportError,ArithmeticError,KeyError,RuntimeError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
