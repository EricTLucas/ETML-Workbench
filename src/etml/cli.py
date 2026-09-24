from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from . import __version__


def _common(parser):
    parser.add_argument('--workspace', default='datasets', help='Dataset storage directory')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress on stderr')


def _profile_options(parser):
    parser.add_argument('--batch-size', type=int, default=50_000)
    parser.add_argument('--sample-size', type=int, default=10_000)
    parser.add_argument('--role', action='append', default=[], metavar='COLUMN=ROLE')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--save-sample', action='store_true')


def parser():
    root = argparse.ArgumentParser(prog='workbench', description='ETML Workbench: run without arguments to choose a project',
        epilog='Start: workbench | Import: workbench PROJECT -upload FILE | Library: workbench sklearn | Project options: workbench PROJECT --help')
    root.add_argument('--version', action='version', version=f'ETML Workbench {__version__}')
    groups = root.add_subparsers(dest='group', required=True)
    groups.add_parser('projects', help='List saved projects', add_help=False)
    for name in ('library','sklearn','huggingface','openml'):
        groups.add_parser(name, help='Dataset library instructions', add_help=False)
    groups.add_parser('ui',help='Open the project data workbench',add_help=False)
    groups.add_parser('models', help='Training, prediction and model export', add_help=False)
    groups.add_parser('tasks', help='Prediction tasks and train/validation/test preparation', add_help=False)
    groups.add_parser('eda', help='Existing EDA analyze, plot, charts and report commands', add_help=False)
    data = groups.add_parser('datasets', help='Import and inspect preserved datasets')
    commands = data.add_subparsers(dest='command', required=True)
    from .source_cli import COMMANDS
    for name in sorted(COMMANDS):
        commands.add_parser(name,help='Import a dataset source',add_help=False)
    imp = commands.add_parser('import', help='Preserve original files; does not run EDA')
    imp.add_argument('files', nargs='+')
    imp.add_argument('--name')
    imp.add_argument('--id', dest='dataset_id')
    imp.add_argument('--max-bytes', type=int)
    listing = commands.add_parser('list')
    show = commands.add_parser('show')
    show.add_argument('dataset_id')
    show.add_argument('--verify', action='store_true')
    for command in (imp, listing, show):
        _common(command)
    pre = groups.add_parser('preprocess', help='Propose, review, preview and execute recipes')
    commands = pre.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Run EDA and save unapproved suggestions')
    prepare.add_argument('dataset_id')
    prepare.add_argument('--source-version', default='raw')
    prepare.add_argument('--protect', nargs='+', default=[])
    prepare.add_argument('--loader-options', default='{}', help='JSON loader options, including dtype overrides')
    _profile_options(prepare)
    show_recipe = commands.add_parser('show', help='Inspect or export an editable recipe')
    review = commands.add_parser('review', help='Save an edited and/or explicitly approved recipe revision')
    preview = commands.add_parser('preview', help='Preview changes without writing processed data')
    execute = commands.add_parser('execute', help='Execute an approved recipe into a new version')
    for command in (show_recipe, review, preview, execute):
        command.add_argument('dataset_id')
        command.add_argument('--recipe', required=True, help='Saved recipe filename shown by prepare/review')
    show_recipe.add_argument('--output', help='Write raw recipe JSON for editing; must be a new file')
    review.add_argument('--from-file', help='Edited raw recipe JSON exported by preprocess show')
    approval = review.add_mutually_exclusive_group()
    approval.add_argument('--approve-all', action='store_true')
    approval.add_argument('--approve', action='append', default=[], metavar='STEP_ID')
    review.add_argument('--reject', action='append', default=[], metavar='STEP_ID')
    preview.add_argument('--max-rows', type=int, default=20)
    preview.add_argument('--fitted', help='Saved FittedRecipe JSON')
    preview.add_argument('--output', help='Save the preview JSON to a new file')
    fitting = execute.add_mutually_exclusive_group()
    fitting.add_argument('--fit-data', nargs='+', help='Explicit fitting dataset files; use training data for ML')
    fitting.add_argument('--fit-on-source', action='store_true', help='Explicitly fit on this entire source version')
    fitting.add_argument('--fitted', help='Previously fitted FittedRecipe JSON')
    execute.add_argument('--fit-loader-options', default='{}', help='JSON loader options for --fit-data')
    execute.add_argument('--no-profile', action='store_true', help='Skip EDA after successful processing')
    _profile_options(execute)
    export_fit = commands.add_parser('export-fitted', help='Export learned transformations for reuse')
    export_fit.add_argument('dataset_id')
    export_fit.add_argument('version')
    export_fit.add_argument('--output', required=True)
    catalog = commands.add_parser('transforms', help='List available transformation operations')
    for command in (prepare, show_recipe, review, preview, execute, export_fit, catalog):
        _common(command)
    return root


def _object(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def _roles(values):
    result = {}
    for value in values:
        if '=' not in value:
            raise ValueError('Expected COLUMN=ROLE')
        column, role = value.split('=', 1)
        if not column or column in result:
            raise ValueError('Role column must be nonempty and specified once')
        result[column] = role
    return result


def _emit(payload, args):
    from .display import emit, command
    steps = []
    common = ['--workspace',args.workspace]
    if args.group=='datasets' and args.command=='import':
        steps = [('Inspect data and propose preprocessing',command('workbench','preprocess','prepare',
                  payload['dataset_id'],*common))]
    elif args.group=='preprocess' and args.command in {'prepare','review'}:
        steps = [('Inspect the saved recipe',command('workbench','preprocess','show',payload['dataset_id'],
                  '--recipe',payload['recipe'],*common))]
        if args.command=='prepare':
            steps.append(('After reviewing the suggestions, approve the recipe',command('workbench','preprocess','review',
                payload['dataset_id'],'--recipe',payload['recipe'],'--approve-all',*common)))
        else:
            steps.append(('Define a prediction task (replace YOUR_TARGET and choose classification/regression)',
                command('workbench','tasks','create',payload['dataset_id'],'prediction','--target','YOUR_TARGET',
                        '--type','classification',*common)))
            steps.append(('After creating that task, fit this recipe on the training split',
                command('workbench','tasks','prepare',payload['dataset_id'],'prediction','--recipe',payload['recipe'],*common)))
    emit(payload,args,steps=steps)


def _proposal(proposal):
    return {'status': 'ok', 'dataset_id': proposal.dataset_id, 'source_version': proposal.source_version,
            'recipe': proposal.recipe_path.name, 'recipe_path': str(proposal.recipe_path),
            'profile_directory': str(proposal.profile_directory), 'steps': proposal.recipe.to_dict()['steps']}


def _dispatch(args):
    from data import DatasetWorkspace
    from data.manifest import resolve_inside, write_json
    from workflows import PreprocessingWorkflow
    from preprocessing import Recipe, FittedRecipe, available_transforms

    workspace = DatasetWorkspace(args.workspace)
    workflow = PreprocessingWorkflow(workspace)
    if args.group == 'datasets':
        if args.command == 'import':
            dataset = workspace.import_files(args.files, name=args.name, dataset_id=args.dataset_id,
                                             max_bytes=args.max_bytes)
            _emit({'status': 'ok', 'dataset_id': dataset.dataset_id, 'directory': str(dataset.directory),
                   'raw_files': [str(p) for p in dataset.raw_files]}, args)
        elif args.command == 'list':
            _emit({'datasets': [{'dataset_id': d.dataset_id, 'name': d.manifest.name,
                                  'directory': str(d.directory)} for d in workspace.list_datasets()]}, args)
        else:
            from dataclasses import asdict
            dataset = workspace.get(args.dataset_id, verify=args.verify)
            versions = workspace.list_versions(args.dataset_id)
            if args.verify:
                for version in versions:
                    workspace.get_version(args.dataset_id, version, verify=True)
            _emit({'manifest': asdict(dataset.manifest), 'versions': versions,
                   'verified': args.verify}, args)
        return 0

    if args.command == 'transforms':
        _emit({'transforms': available_transforms()}, args)
        return 0
    if args.command == 'export-fitted':
        path = workflow.load_fitted(args.dataset_id, args.version).save(args.output)
        _emit({'status': 'ok', 'output': str(path.resolve())}, args)
        return 0

    def progress(event):
        if not args.quiet:
            message = event.message if hasattr(event, 'message') else (
                f"Processing: {event['input_rows']} input rows, {event['output_rows']} output rows")
            print(message, file=sys.stderr)

    if args.command in {'prepare', 'execute'}:
        from eda_tool.profiler import ProfileConfig
        config = ProfileConfig(batch_size=args.batch_size, sample_size=args.sample_size, roles=_roles(args.role))
    if args.command == 'prepare':
        proposal = workflow.prepare(args.dataset_id, source_version=args.source_version, profile_config=config,
                                     loader_options=_object(args.loader_options), protected_columns=args.protect,
                                     generate_summary=args.summary, export_html=args.html,
                                     save_sample=args.save_sample, progress=progress)
        _emit(_proposal(proposal), args)
        return 0

    proposal = workflow.load_proposal(args.dataset_id, args.recipe)
    if args.command == 'show':
        if args.output:
            proposal.recipe.save(args.output)
        _emit(_proposal(proposal), args)
    elif args.command == 'review':
        recipe = Recipe.load(args.from_file) if args.from_file else proposal.recipe
        known = {s.step_id for s in recipe.steps}
        approve, reject = set(args.approve), set(args.reject)
        if (approve | reject)-known or approve & reject:
            raise ValueError('Unknown or conflicting approval/rejection step IDs')
        if not (args.from_file or args.approve_all or approve or reject):
            raise ValueError('Specify --from-file, --approve-all, --approve, or --reject')
        steps = tuple(s.reject() if s.step_id in reject else s.approve()
                      if (s.step_id in approve or args.approve_all and s.enabled and s.status != 'rejected') else s
                      for s in recipe.steps)
        reviewed = workflow.save_review(proposal, replace(recipe, steps=steps))
        _emit(_proposal(reviewed), args)
    elif args.command == 'preview':
        fitted = FittedRecipe.load(args.fitted) if args.fitted else None
        result = workflow.preview(proposal, fitted=fitted, max_rows=args.max_rows)
        payload = {'changes': result.changes, 'metadata': result.metadata,
                   'before': json.loads(result.before.to_json(orient='split', date_format='iso')),
                   'after': json.loads(result.after.to_json(orient='split', date_format='iso'))}
        if args.output:
            write_json(Path(args.output), payload)
        _emit(payload, args)
    elif args.command == 'execute':
        if args.no_profile and (args.summary or args.html or args.save_sample):
            raise ValueError('--no-profile cannot be combined with --summary, --html, or --save-sample')
        fitted = FittedRecipe.load(args.fitted) if args.fitted else None
        fit_source = args.fit_data
        fit_options = _object(args.fit_loader_options)
        if args.fit_on_source:
            dataset = workspace.get(args.dataset_id)
            if proposal.source_version == 'raw':
                fit_source = list(dataset.raw_files)
            else:
                manifest = workspace.get_version(args.dataset_id, proposal.source_version)
                root = resolve_inside(dataset.directory, 'processed/'+proposal.source_version)
                fit_source = [resolve_inside(root, f.path) for f in manifest.files]
            fit_options = proposal.loader_options
        result = workflow.execute(proposal, fitted=fitted, fit_source=fit_source,
                                   fit_loader_options=fit_options, batch_size=args.batch_size,
                                   profile_after=not args.no_profile, profile_config=config,
                                   generate_summary=args.summary, export_html=args.html,
                                   save_sample=args.save_sample, progress=progress, profile_progress=progress)
        _emit({'status': 'processed_profile_failed' if result.profile_error else 'ok',
               'dataset_id': result.execution.dataset_id, 'version': result.execution.version,
               'directory': str(result.execution.directory), 'input_rows': result.execution.input_rows,
               'output_rows': result.execution.output_rows,
               'profile_directory': str(result.profile_directory) if result.profile_directory else None,
               'report_path': str(result.report_path) if result.report_path else None,
               'profile_error': result.profile_error}, args)
        return 3 if result.profile_error else 0
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        known = {'ui','datasets','preprocess','tasks','models','eda','projects','library','sklearn','huggingface','openml'}
        if not argv or argv[0] in {'--projects-dir','--no-open','--no-eda','--no-continue','--continue','--status','--models','--predict-model','--predict'} or (argv[0] not in known and not argv[0].startswith('-')):
            from .project_cli import main as project_main
            return project_main(argv)
        if argv[0] in {'library','sklearn','huggingface','openml'}:
            from .project_cli import library
            if len(argv)>1 and argv[1:] != ['--help']:
                raise ValueError('Use workbench PROJECT --'+argv[0]+' DATASET to import into a project')
            library(None if argv[0]=='library' else argv[0])
            return 0
        if argv[0]=='projects':
            import os
            from data.projects import ProjectStore
            listing = argparse.ArgumentParser(prog='workbench projects')
            listing.add_argument('--projects-dir',default=os.environ.get('ETML_PROJECTS_DIR','projects'))
            options = listing.parse_args(argv[1:])
            projects = ProjectStore(options.projects_dir).list()
            print('Projects' if projects else 'No projects yet. Run workbench to create one.')
            for project in projects:
                print('  '+project.name)
            return 0
        if argv and argv[0]=='ui':
            from .ui import launch
            return launch(argv[1:])
        if len(argv)>1 and argv[0]=='datasets':
            from .source_cli import COMMANDS,main as source_main
            if argv[1] in COMMANDS: return source_main(argv[1:])
        if argv and argv[0] == 'models':
            from .model_cli import main as model_main
            return model_main(argv[1:])
        if argv and argv[0] == 'tasks':
            from .task_cli import main as task_main
            return task_main(argv[1:])
        if argv and argv[0] == 'eda':
            from eda_tool.cli import main as eda_main
            return eda_main(argv)
        args = parser().parse_args(argv)
        return _dispatch(args)
    except (ValueError, TypeError, OSError, ImportError, ArithmeticError, KeyError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
