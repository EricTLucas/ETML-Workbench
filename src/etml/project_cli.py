"""Small project-first onboarding flow; advanced commands stay independently usable."""
import argparse
import os
import sys
from pathlib import Path
import uuid
import webbrowser
from .display import command


def parser():
    p = argparse.ArgumentParser(prog='workbench PROJECT', description='Open a project or import its data')
    p.add_argument('project', nargs='?')
    source = p.add_mutually_exclusive_group()
    source.add_argument('-upload', '--upload', metavar='FILE')
    source.add_argument('--sklearn', metavar='DATASET')
    source.add_argument('--huggingface', metavar='OWNER/DATASET')
    source.add_argument('--openml', type=int, metavar='DATA_ID')
    source.add_argument('--url', metavar='URL')
    p.add_argument('--test', help='Optional test file alongside --upload training data')
    p.add_argument('--validation', help='Optional validation file alongside --upload')
    p.add_argument('--config'); p.add_argument('--split', default='train'); p.add_argument('--revision')
    p.add_argument('--columns', nargs='+')
    p.add_argument('--no-open', action='store_true', help='Save EDA without opening a browser')
    p.add_argument('--no-eda', action='store_true', help='Import and preview only')
    p.add_argument('--eda', action='store_true', help='Generate EDA for the current dataset')
    flow = p.add_mutually_exclusive_group()
    flow.add_argument('--continue', dest='continue_flow', action='store_true', help='Review preprocessing and save splits interactively')
    flow.add_argument('--no-continue', '--status', dest='no_continue', action='store_true', help='Show/import data without entering the preparation wizard')
    p.add_argument('--dataset-id', help='Use a previous dataset within this project')
    p.add_argument('--models', action='store_true', help='Choose, name and train a project model')
    p.add_argument('--predict', action='store_true', help='Open saved-model prediction, details and export')
    p.add_argument('--predict-model', metavar='NAME', help='Predict with a saved project model')
    p.add_argument('--values', help='JSON input object for --predict-model')
    p.add_argument('--batch-size', type=int, default=50_000)
    p.add_argument('--sample-size', type=int, default=10_000)
    p.add_argument('--max-rows', type=int, default=200_000)
    p.add_argument('--max-memory-mb', type=int, default=512)
    p.add_argument('--projects-dir', default=os.environ.get('ETML_PROJECTS_DIR', 'projects'))
    return p


def library(provider=None):
    print('Dataset library\n')
    if provider in (None, 'sklearn'):
        print('sklearn — bundled datasets, no download required')
        print('  iris           Flower classification (150 rows)')
        print('  wine           Wine classification')
        print('  breast_cancer  Binary classification')
        print('  digits         Handwritten digits as tabular pixel features')
        print('  diabetes       Regression')
        print('\n  workbench my-project --sklearn iris\n')
    if provider in (None, 'huggingface'):
        print('Hugging Face — scalar tabular datasets from the Hub')
        print('  Install: python -m pip install -e ".[huggingface]"')
        print('  Example: workbench my-project --huggingface scikit-learn/iris')
        print('  Choose another: --huggingface OWNER/DATASET --split train')
        print('  Optional: --config NAME --revision COMMIT --columns COLUMN TARGET\n')
    if provider in (None, 'openml'):
        print('OpenML — choose a numeric dataset ID')
        print('  Example: workbench my-project --openml 61  (Iris)\n')
    print('Replace my-project with your project name. Imports automatically generate EDA.')


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        raise ValueError('Interactive input ended. Saved work is preserved. Use workbench PROJECT --continue to resume or --status to inspect.') from None


def _choose(title, choices):
    print(title)
    for i, label in enumerate(choices, 1):
        print(f'  {i}. {label}')
    while True:
        value = _ask('Choose a number (q to cancel): ')
        if value.lower() in {'q', 'quit', 'exit'}:
            raise KeyboardInterrupt
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return int(value)-1
        print('Please choose one of the listed numbers.')


def _next(project, args):
    print('\nNext: preprocessing and data split')
    print('  '+command('workbench', project.name, '--continue', '--projects-dir', args.projects_dir,
                       *(['--dataset-id',args.dataset_id] if args.dataset_id else [])))


def _continue(project, args, dataset_id=None):
    if not dataset_id and not args.dataset_id and project.current_dataset() is None:
        if not args.no_continue and (args.continue_flow or sys.stdin.isatty()):
            from .model_wizard import run
            return run(project,args)
        print('Next: '+command('workbench',project.name,'--models','--projects-dir',args.projects_dir))
        return 0
    if not args.no_continue and (args.continue_flow or sys.stdin.isatty()):
        from .preparation_wizard import run
        return run(project,args,dataset_id=dataset_id or args.dataset_id,show_saved=not getattr(args,'_artifacts_shown',False))
    _next(project,args)
    return 0


def _select_source(project, args):
    selection = _choose('\nChoose data', ['Upload a local file', 'Dataset library', 'Do this later'])
    if selection == 2:
        print('\nImport when ready: '+command('workbench', project.name, '-upload', 'FILE_PATH',
                                             '--projects-dir', args.projects_dir))
        return False
    if selection == 0:
        args.upload = _ask('File path: ').strip('"').strip("'")
        if not args.upload:
            raise ValueError('A file path is required')
    else:
        provider = _choose('\nDataset provider', ['sklearn (built in)', 'Hugging Face', 'OpenML'])
        if provider == 0:
            from data.sources import SKLEARN_DATASETS
            names = list(SKLEARN_DATASETS)
            args.sklearn = names[_choose('\nChoose a dataset', names)]
        elif provider == 1:
            library('huggingface')
            args.huggingface = _ask('Hub dataset (OWNER/DATASET): ')
            args.config = _ask('Configuration (Enter for default): ') or None
            args.split = _ask('Split (Enter for train): ') or 'train'
        else:
            library('openml')
            args.openml = int(_ask('OpenML data ID: '))
    return True


def _source(dataset):
    from data.manifest import resolve_inside
    if dataset.manifest.split_files:
        return resolve_inside(dataset.directory, dataset.manifest.split_files['train'])
    return list(dataset.raw_files)


def _preview(dataset):
    from eda_tool.loader import open_dataset
    from preprocessing.transforms.base import batches
    with batches(lambda: open_dataset(_source(dataset)).iter_batches(batch_size=5)) as iterator:
        frame = next(iterator, None)
    if frame is not None:
        print('\nData preview (first 5 training/source rows):')
        print(frame.head(5).to_string(index=False, max_cols=8, max_colwidth=24, line_width=120))
        print(f'{len(frame.columns)} columns. Raw data preserved.')


def _analyze(project, dataset, args):
    from eda_tool.workflows import run_eda
    from eda_tool.profiler import ProfileConfig
    print('\nGenerating EDA...')
    output = dataset.profiles_dir / ('overview-'+uuid.uuid4().hex[:12])
    try:
        result = run_eda(_source(dataset), output_dir=output, export_html=True,
                         generate_summary=True, save_sample=True,
                         profile_config=ProfileConfig(batch_size=args.batch_size, sample_size=args.sample_size),
                         title=project.name+' — Data overview')
        try:
            report = result.html_path
        finally:
            result.close()
        print('EDA ready: '+str(report))
        if not args.no_open:
            try:
                opened = webbrowser.open(Path(report).resolve().as_uri())
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                print('Could not open a browser. Open the report path above.')
        return 0
    except (ValueError, TypeError, OSError, ImportError, ArithmeticError, KeyError, RuntimeError) as exc:
        print(f'Your data is saved, but EDA could not finish: {exc}')
        print('Retry: '+command('workbench', project.name, '--eda', '--projects-dir', args.projects_dir))
        return 3


def _status(project, args):
    dataset = project.workspace.get(args.dataset_id) if args.dataset_id else project.current_dataset()
    print('\nProject: '+project.name)
    from workflows.project_models import ProjectModels
    models=ProjectModels(project).list()
    inputs=sorted((project.directory/'inputs').glob('input-*/input.json'))
    for path in inputs:
        import json
        info=json.loads(path.read_text(encoding='utf-8'))
        print(f"  Input {path.parent.name}: {info['modality']}; splits {info['split_counts']}")
    for model in models:
        print(f"  Model {model['name']}: {model['model']} ({model['status']})")
    if dataset is None:
        print('No tabular dataset.' if inputs or models else 'No dataset yet.')
        return bool(inputs or models)
    print('Current dataset: '+dataset.manifest.name)
    try:
        _preview(dataset)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        print(f'Preview unavailable: {exc}')
    reports = sorted(dataset.profiles_dir.glob('overview-*/report.html'), key=lambda p:p.stat().st_mtime_ns)
    if reports:
        print('EDA: '+str(reports[-1]))
    from workflows.project_preparation import ProjectPreparation
    from .preparation_wizard import show_artifacts
    show_artifacts(ProjectPreparation(project,dataset.dataset_id,batch_size=args.batch_size))
    args._artifacts_shown=True
    older = [d for d in project.datasets() if d.dataset_id!=dataset.dataset_id]
    if older:
        print('\nOther saved datasets:')
        for item in older:
            print('  '+item.manifest.name+': '+command('workbench',project.name,'--dataset-id',item.dataset_id,
                  '--projects-dir',args.projects_dir))
    return True


def main(argv=None):
    args = parser().parse_args(argv)
    from data.projects import ProjectStore
    store = ProjectStore(args.projects_dir)
    try:
        has_source = any((args.upload, args.sklearn, args.huggingface, args.openml is not None, args.url))
        if (args.models or args.predict_model or args.predict) and (has_source or args.eda):
            raise ValueError('Use model commands separately from imports/EDA')
        if sum(bool(v) for v in (args.models,args.predict_model,args.predict))>1:
            raise ValueError('Choose --models, --predict, or --predict-model')
        if bool(args.values) != bool(args.predict_model):
            raise ValueError('--predict-model and --values are required together')
        if has_source and args.dataset_id:
            raise ValueError('--dataset-id selects existing data; do not combine it with an import')
        if (args.test or args.validation) and not args.upload:
            raise ValueError('--test/--validation require --upload for the training file')
        if args.eda and (args.no_eda or has_source):
            raise ValueError('Use --eda alone to analyze existing data; imports analyze automatically')
        if min(args.batch_size, args.sample_size, args.max_rows, args.max_memory_mb) < 1:
            raise ValueError('Batch, sample, row, and memory limits must be positive')
        if not args.project:
            print('ETML Workbench\n')
            choice = _choose('Projects', ['New project', 'Existing project'])
            if choice == 0:
                name = _ask('Project name (letters, numbers, hyphens; no spaces): ')
                project = store.create(name)
            else:
                projects = store.list()
                if not projects:
                    print('No projects yet. Run workbench and choose New project.')
                    return 0
                project = projects[_choose('\nExisting projects', [p.name for p in projects])]
        else:
            try:
                project = store.get(args.project)
            except FileNotFoundError:
                if has_source:
                    project = store.create(args.project)
                else:
                    raise ValueError('Project not found. Run workbench to create one, or supply --upload/--sklearn.') from None
        if args.predict_model:
            import json
            from workflows.project_models import ProjectModels
            values=json.loads(args.values)
            if not isinstance(values,dict): raise ValueError('--values must be a JSON object')
            print(json.dumps(ProjectModels(project).predict(args.predict_model,values),indent=2))
            return 0
        if args.models:
            from .model_wizard import run
            return run(project,args)
        if args.predict:
            from .prediction_wizard import run
            return run(project,args)
        if not has_source:
            if args.eda:
                dataset = project.workspace.get(args.dataset_id) if args.dataset_id else project.current_dataset()
                if dataset is None:
                    raise ValueError('Import a dataset first')
                _preview(dataset)
                return _analyze(project, dataset, args)
            if _status(project, args):
                if not args.no_continue and not args.continue_flow and sys.stdin.isatty():
                    action=_choose('\nProject actions',['Continue preprocessing / splits','Choose and train a model',
                        'Predict, inspect or export models','Finish'])
                    if action==3: return 0
                    if action==1:
                        from .model_wizard import run
                        return run(project,args)
                    if action==2:
                        from .prediction_wizard import run
                        return run(project,args)
                return _continue(project,args)
            if not _select_source(project, args):
                return 0
        from data.sources import import_sklearn, import_huggingface, import_openml, import_url
        options = {'dataset_id':'data-'+uuid.uuid4().hex[:12], 'max_bytes':args.max_memory_mb*1024**2}
        ws = project.workspace
        print('\nImporting data...')
        if args.upload:
            if args.test or args.validation:
                dataset = ws.import_presplit(args.upload, test=args.test, validation=args.validation, **options)
            else:
                dataset = ws.import_files(args.upload, **options)
        elif args.sklearn:
            dataset = import_sklearn(ws, args.sklearn, max_rows=args.max_rows, **options)
        elif args.huggingface:
            dataset = import_huggingface(ws, args.huggingface, config=args.config, split=args.split,
                        revision=args.revision, columns=args.columns, max_rows=args.max_rows, **options)
        elif args.openml is not None:
            dataset = import_openml(ws, args.openml, max_rows=args.max_rows, **options)
        else:
            dataset = import_url(ws, args.url, **options)
        print('Project: '+project.name)
        try:
            _preview(dataset)
        except (ValueError, TypeError, OSError, ImportError) as exc:
            print(f'Data saved; preview unavailable: {exc}')
        status = 0 if args.no_eda else _analyze(project, dataset, args)
        if status == 0:
            return _continue(project,args,dataset.dataset_id)
        return status
    except KeyboardInterrupt:
        print('\nCancelled. Saved projects and data are preserved.')
        return 130
    except (ValueError, TypeError, OSError, ImportError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}')
        return 2
