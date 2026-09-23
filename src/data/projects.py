"""Project identity and isolated dataset workspaces."""
from dataclasses import dataclass
from pathlib import Path
import json
from .manifest import validate_name, resolve_inside, write_json, utc_now
from .workspace import DatasetWorkspace, _publish_directory
from tempfile import TemporaryDirectory

RESERVED = {'datasets', 'preprocess', 'tasks', 'models', 'eda', 'ui', 'projects',
            'sklearn', 'huggingface', 'openml', 'library'}


@dataclass(frozen=True)
class Project:
    directory: Path
    name: str

    @property
    def workspace(self):
        return DatasetWorkspace(self.directory / 'datasets')

    def datasets(self):
        return sorted(self.workspace.list_datasets(), key=lambda d: d.manifest.created_at)

    def current_dataset(self):
        datasets = self.datasets()
        return datasets[-1] if datasets else None


class ProjectStore:
    def __init__(self, root='projects'):
        self.root = Path(root).expanduser().resolve()

    def get(self, name):
        path = resolve_inside(self.root, validate_name(name))
        value = json.loads((path / 'project.json').read_text(encoding='utf-8'))
        if value.get('format') != 'etml-project' or value.get('name') != name:
            raise ValueError('Invalid project manifest')
        return Project(path, name)

    def create(self, name):
        validate_name(name)
        if name.lower() in RESERVED:
            raise ValueError('That name is reserved for a workbench command; choose another project name')
        path = resolve_inside(self.root, name)
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f'Project {name!r} already exists; select Existing project')
        with TemporaryDirectory(prefix='.project-', dir=self.root) as temp:
            staging = Path(temp) / name
            staging.mkdir()
            write_json(staging / 'project.json', {'format':'etml-project', 'version':1,
                       'name':name, 'created_at':utc_now()})
            (staging / 'datasets').mkdir()
            _publish_directory(staging, path)
        return self.get(name)

    def list(self):
        if not self.root.exists():
            return []
        return [self.get(path.name) for path in sorted(self.root.iterdir(), key=lambda p:p.name.lower())
                if path.is_dir() and (path / 'project.json').is_file()]
