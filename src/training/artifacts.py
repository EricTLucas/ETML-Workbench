from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import platform

from data.manifest import FileRecord, resolve_inside, write_json
from data.workspace import _publish_directory


@contextmanager
def staged_directory(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.model-',dir=destination.parent) as temp:
        root = Path(temp)
        yield root
        _publish_directory(root,destination)


def environment():
    result = {'python':platform.python_version()}
    for package in ('scikit-learn','skops','numpy','pandas','scipy','xgboost','eda_tool'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def seal(directory, kind, metadata):
    root = Path(directory)
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            records.append(asdict(FileRecord.from_file(root,path)))
    payload = {'format_version':1,'kind':kind,'metadata':metadata,'files':records}
    write_json(root/'manifest.json',payload)
    return payload


def verify_artifacts(directory, kind=None):
    root = Path(directory).resolve()
    payload = json.loads(resolve_inside(root,'manifest.json').read_text(encoding='utf-8'))
    if payload.get('format_version') != 1 or (kind and payload.get('kind') != kind):
        raise ValueError('Unsupported artifact manifest')
    names = set()
    for entry in payload['files']:
        record = FileRecord(**entry)
        if record.path.casefold() in names:
            raise ValueError('Duplicate artifact path')
        names.add(record.path.casefold())
        record.verify(root)
    return payload


def require_files(manifest, paths):
    present = {item['path'] for item in manifest['files']}
    if not set(paths) <= present:
        raise ValueError('Model bundle is missing required verified artifacts')
