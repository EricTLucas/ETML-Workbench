"""Browser operations sharing the CLI's persistent project workflows."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
import json
import uuid
from time import monotonic
from numbers import Integral

from data.projects import ProjectStore
from data import sources
from data.manifest import resolve_inside
from eda_tool.loader import open_dataset
from preprocessing import Recipe
from preprocessing.transforms.base import batches
from preprocessing.transforms import build_transform
from splitting import SplitConfig
from workflows.project_preparation import ProjectPreparation


def table(frame):
    # pandas handles timestamps, numpy scalars and missing values consistently.
    return {'columns': list(frame.columns), 'rows': json.loads(frame.to_json(orient='values', date_format='iso')),
            'indexes': [str(i+1) if isinstance(i, Integral) else str(i) for i in frame.index]}


class Service:
    def __init__(self, root, *, max_bytes=512*1024**2):
        self.store = ProjectStore(root)
        self.max_bytes = max_bytes
        # One worker also serializes matplotlib, which has process-global state.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='etml-web')
        self.lock = Lock()
        self.jobs, self.busy, self.previews = {}, {}, {}

    def close(self):
        self.pool.shutdown(wait=True)

    def submit(self, name, label, operation, *, with_progress=False):
        self.store.get(name)
        with self.lock:
            if name in self.busy:
                raise ValueError('This project has an operation in progress. Wait for it to finish.')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'id': job_id, 'project': name, 'label': label, 'status': 'queued'}
            self.busy[name] = job_id
        def run():
            started = monotonic()
            def report(event):
                # Keep real backend events bounded; never invent percentage progress.
                with self.lock:
                    job = self.jobs[job_id]
                    job['elapsed_seconds'] = monotonic()-started
                    job.setdefault('events', []).append(dict(event))
                    job['events'] = job['events'][-200:]
            with self.lock:
                self.jobs[job_id]['status'] = 'running'
            try:
                result = operation(report) if with_progress else operation()
                with self.lock:
                    self.jobs[job_id].update(status='done', result=result)
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(status='failed', error=str(exc))
            finally:
                with self.lock:
                    self.busy.pop(name, None)
                    # Bound finished job metadata while keeping active jobs.
                    for key in list(self.jobs)[:-100]:
                        if self.jobs[key]['status'] in {'done', 'failed'}:
                            self.jobs.pop(key, None)
        self.pool.submit(run)
        return {'job_id': job_id}

    def job(self, job_id):
        with self.lock:
            return dict(self.jobs[job_id])

    def preparation(self, name, dataset_id=None):
        return ProjectPreparation(self.store.get(name), dataset_id)

    def report(self, dataset):
        reports = sorted(dataset.profiles_dir.rglob('*.html'), key=lambda p: p.stat().st_mtime)
        return reports[-1].relative_to(dataset.profiles_dir).as_posix() if reports else None

    def overview(self, name, dataset_id=None):
        project = self.store.get(name)
        datasets = project.datasets()
        dataset = project.workspace.get(dataset_id) if dataset_id else project.current_dataset()
        result = {'name': name, 'directory': str(project.directory), 'projects_dir': str(self.store.root),
                  'datasets': [{'id': d.dataset_id, 'name': d.manifest.name} for d in datasets],
                  'dataset': None, 'busy': self.busy.get(name)}
        if dataset is None:
            return result
        prep = self.preparation(name, dataset.dataset_id)
        with batches(lambda: open_dataset(prep.source()).iter_batches(batch_size=10)) as iterator:
            first = next(iterator, None)
        state = prep.state()
        result['dataset'] = {'id': dataset.dataset_id, 'name': dataset.manifest.name,
            'sources': [str(p) for p in prep.source()],
            'columns': {str(k): str(v) for k, v in prep.schema().items()},
            'preview': table(first.head(10)) if first is not None else {'columns': [], 'rows': [], 'indexes': []},
            'report': self.report(dataset), 'state': state,
            'presplit': dict(dataset.manifest.split_files),
            'provenance': dataset.manifest.provenance,
            'recipe': prep.recipe().to_dict() if state.get('recipe') else state.get('proposal'),
            'artifacts': [{'label': label, 'path': str(path)} for label, path in prep.artifacts()]}
        return result

    def analyze(self, name, dataset_id):
        from eda_tool.workflows import run_eda
        from eda_tool.profiler import ProfileConfig
        prep = self.preparation(name, dataset_id)
        result = run_eda(prep.source(), output_dir=prep.dataset.profiles_dir/('overview-'+uuid.uuid4().hex[:12]),
                         export_html=True, generate_summary=True, save_sample=True,
                         profile_config=ProfileConfig(batch_size=50_000, sample_size=5_000),
                         title=name+' — Data overview')
        try:
            return {'report': result.html_path.relative_to(prep.dataset.profiles_dir).as_posix()}
        finally:
            result.close()

    def import_data(self, name, payload):
        workspace = self.store.get(name).workspace
        source = payload.get('source', 'path')
        kwargs = {'max_bytes': self.max_bytes}
        if source == 'path':
            path = Path(payload['path']).expanduser()
            # Validate that the loader supports the input before publishing raw files.
            open_dataset(path).schema()
            dataset = workspace.import_files(path, **kwargs)
        elif source == 'sklearn':
            dataset = sources.import_sklearn(workspace, payload['name'], **kwargs)
        elif source == 'huggingface':
            dataset = sources.import_huggingface(workspace, payload['repository'],
                config=payload.get('config') or None, split=payload.get('split') or 'train',
                revision=payload.get('revision') or None, **kwargs)
        elif source == 'openml':
            dataset = sources.import_openml(workspace, int(payload['data_id']), **kwargs)
        elif source == 'url':
            dataset = sources.import_url(workspace, payload['url'], **kwargs)
        else:
            raise ValueError('Unknown data source')
        result = {'dataset_id': dataset.dataset_id}
        if payload.get('analyze', True):
            try:
                result.update(self.analyze(name, dataset.dataset_id))
            except Exception as exc:
                result['warning'] = 'Data saved, but the EDA could not finish: '+str(exc)
        return result

    def suggest(self, name, payload):
        prep = self.preparation(name, payload['dataset_id'])
        previous = prep.state()
        try:
            prep.configure(payload['target'], payload['task_type'], payload.get('excluded', ()),
                           payload.get('missing_target', 'error'))
            recipe = prep.suggest().to_dict()
            state = prep.state()
            state['proposal'] = recipe
            prep._save(state)
        except Exception:
            if previous:
                prep._save(previous)
            raise
        self.previews.pop(name, None)
        return {'recipe': recipe}

    def preview(self, name, payload):
        prep = self.preparation(name, payload['dataset_id'])
        recipe = Recipe.from_dict(payload['recipe']).approve_all()
        for step in recipe.active_steps:
            build_transform(step)
        previous = prep.state()
        try:
            prep.save_recipe(recipe)
            fitted, before, after = prep.fit_preview()
        except Exception:
            if previous:
                prep._save(previous)
            raise
        self.previews.pop(name, None)
        token = uuid.uuid4().hex
        self.previews[name] = (token, prep.dataset.dataset_id, prep.state()['recipe'], fitted)
        return {'token': token, 'before': table(before), 'after': table(after),
                'affected_columns': sorted({c for s in recipe.active_steps for c in s.columns}),
                'recipe': recipe.to_dict(), 'preview_info': before.attrs['preview']}

    def save(self, name, payload):
        cached = self.previews.get(name)
        if not cached or cached[0] != payload.get('token') or cached[1] != payload['dataset_id']:
            raise ValueError('Preview the current recipe again before saving.')
        prep = self.preparation(name, payload['dataset_id'])
        if prep.state().get('recipe') != cached[2]:
            raise ValueError('The recipe changed. Preview it again before saving.')
        result = prep.save_processed(cached[3])
        self.previews.pop(name, None)
        return result

    def split(self, name, payload):
        prep = self.preparation(name, payload['dataset_id'])
        prep.split(SplitConfig(**payload['config']), validation_fraction=payload.get('validation_fraction', .2))
        return prep.state()['split']
