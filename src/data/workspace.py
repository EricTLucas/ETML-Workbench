"""Local dataset storage. No pandas dependency, profiling, or cleaning decisions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import BinaryIO, Iterator
import uuid

from .manifest import (
    CHUNK_SIZE, DatasetManifest, FileRecord, ProcessedManifest, load_manifest,
    resolve_inside, sha256_file, utc_now, validate_name, validate_filename, write_json,
)


def _publish_directory(staging: Path, destination: Path) -> None:
    # Windows scanners may briefly hold a freshly closed file in this directory.
    # Only retry sharing/access errors; never overwrite a published destination.
    for attempt in range(6):
        if destination.exists():
            raise FileExistsError(destination)
        try:
            staging.rename(destination)
            return
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


@dataclass(frozen=True)
class Dataset:
    directory: Path
    manifest: DatasetManifest

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    @property
    def raw_files(self) -> tuple[Path, ...]:
        return tuple(resolve_inside(self.directory, f.path) for f in self.manifest.files)

    @property
    def profiles_dir(self) -> Path:
        return resolve_inside(self.directory, 'profiles')

    @property
    def recipes_dir(self) -> Path:
        return resolve_inside(self.directory, 'recipes')


@dataclass
class VersionDraft:
    """Temporary destination for an executor; valid only inside the context."""
    version: str
    directory: Path
    schema: dict[str, str]

    @property
    def data_dir(self) -> Path:
        return self.directory / 'data'


class DatasetWorkspace:
    def __init__(self, root: str | Path = 'datasets', *, chunk_size: int = CHUNK_SIZE):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.root = Path(root).expanduser().resolve()
        self.chunk_size = chunk_size
        # Reading/constructing a workspace does not create directories.

    def _directory(self, dataset_id: str) -> Path:
        return resolve_inside(self.root, validate_name(dataset_id))

    @contextmanager
    def _lock(self, dataset_id: str):
        self.root.mkdir(parents=True, exist_ok=True)
        locks = resolve_inside(self.root, 'locks')
        locks.mkdir(exist_ok=True)
        lock = resolve_inside(locks, validate_name(dataset_id) + '.lock')
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(f'Dataset {dataset_id!r} is being written; retry after that operation finishes') from exc
        try:
            yield
        finally:
            lock.rmdir()

    @staticmethod
    def _new_id(name: str) -> str:
        stem = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:70] or 'dataset'
        return stem + '-' + uuid.uuid4().hex[:12]

    def _copy(self, stream: BinaryIO, destination: Path, relative: str, max_bytes: int | None) -> FileRecord:
        digest, size = hashlib.sha256(), 0
        with destination.open('xb') as output:
            while True:
                block = stream.read(self.chunk_size)
                if not isinstance(block, (bytes, bytearray)):
                    raise TypeError('Uploads must be binary streams, opened with rb')
                if not block:
                    break
                size += len(block)
                if max_bytes is not None and size > max_bytes:
                    raise ValueError('Import exceeds max_bytes')
                output.write(block)
                digest.update(block)
        return FileRecord(relative, size, digest.hexdigest())

    def _import(self, sources, *, name, dataset_id, max_bytes) -> Dataset:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be a nonempty display name')
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError('max_bytes must be a nonnegative integer or None')
        dataset_id = validate_name(dataset_id or self._new_id(name))
        if dataset_id.casefold() == 'locks':
            raise ValueError('Dataset ID locks is reserved')
        destination = self._directory(dataset_id)
        names = [validate_filename(filename) for filename, _ in sources]
        if not names or len({n.casefold() for n in names}) != len(names):
            raise ValueError('Provide files with distinct portable filenames')
        with self._lock(dataset_id):
            if destination.exists():
                raise FileExistsError(f'Dataset already exists: {dataset_id}')
            with TemporaryDirectory(prefix='.import-', dir=self.root) as temp:
                staging = Path(temp)
                for folder in ('raw', 'profiles', 'recipes', 'processed'):
                    (staging / folder).mkdir()
                records, total = [], 0
                for filename, source in sources:
                    remaining = None if max_bytes is None else max_bytes-total
                    target = staging / 'raw' / filename
                    if isinstance(source, Path):
                        with source.open('rb') as stream:
                            record = self._copy(stream, target, 'raw/'+filename, remaining)
                    else:
                        record = self._copy(source, target, 'raw/'+filename, remaining)
                    records.append(record)
                    total += record.size_bytes
                manifest = DatasetManifest(dataset_id, name, utc_now(), tuple(records))
                manifest.save(staging / 'manifest.json')
                # Publish only after every file and the manifest were written.
                if destination.exists():
                    raise FileExistsError(destination)
                _publish_directory(staging, destination)
        return Dataset(destination, manifest)

    def import_files(self, paths, *, name: str | None = None, dataset_id: str | None = None,
                     max_bytes: int | None = None) -> Dataset:
        """Copy one file or an explicit list of files, bounded by chunk_size."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(p).expanduser().resolve() for p in paths]
        if not paths or any(not p.is_file() for p in paths):
            raise ValueError('Provide one or more existing regular files')
        return self._import([(p.name, p) for p in paths], name=name or paths[0].stem,
                            dataset_id=dataset_id, max_bytes=max_bytes)

    def import_upload(self, stream: BinaryIO, *, filename: str, name: str | None = None,
                      dataset_id: str | None = None, max_bytes: int | None = None) -> Dataset:
        """Copy bytes from the current stream position. Does not seek or close it.

        Rewind a previously read upload yourself before calling this method.
        filename must be a portable basename, not an untrusted client path.
        """
        validate_filename(filename)
        return self._import([(filename, stream)], name=name or Path(filename).stem,
                            dataset_id=dataset_id, max_bytes=max_bytes)

    def get(self, dataset_id: str, *, verify: bool = False) -> Dataset:
        directory = self._directory(dataset_id)
        manifest = load_manifest(resolve_inside(directory, 'manifest.json'))
        if not isinstance(manifest, DatasetManifest) or manifest.dataset_id != dataset_id:
            raise ValueError('Dataset manifest does not match its directory')
        for record in manifest.files:
            resolve_inside(directory, record.path)
            if verify:
                record.verify(directory)
        return Dataset(directory, manifest)

    def list_datasets(self) -> list[Dataset]:
        if not self.root.exists():
            return []
        return [self.get(p.name) for p in sorted(self.root.iterdir())
                if p.is_dir() and not p.name.startswith('.') and p.name != 'locks'
                and (p / 'manifest.json').is_file()]

    def set_schema(self, dataset_id: str, schema: dict[str, str]) -> Dataset:
        """Attach a known schema; this does not inspect or convert raw values."""
        with self._lock(dataset_id):
            dataset = self.get(dataset_id)
            manifest = replace(dataset.manifest, schema=dict(schema))
            manifest.save(dataset.directory / 'manifest.json', overwrite=True)
        return Dataset(dataset.directory, manifest)

    def get_version(self, dataset_id: str, version: str, *, verify: bool = False) -> ProcessedManifest:
        if not re.fullmatch(r'v[1-9][0-9]*', version):
            raise ValueError('Expected a version such as v1')
        dataset = self.get(dataset_id)
        root = resolve_inside(dataset.directory, 'processed/'+version)
        manifest = load_manifest(resolve_inside(root, 'manifest.json'))
        if not isinstance(manifest, ProcessedManifest) or manifest.dataset_id != dataset_id or manifest.version != version:
            raise ValueError('Processed manifest does not match its directory')
        for record in (*manifest.files, manifest.recipe):
            resolve_inside(root, record.path)
            if verify:
                record.verify(root)
        return manifest

    def list_versions(self, dataset_id: str) -> list[str]:
        dataset = self.get(dataset_id)
        root = resolve_inside(dataset.directory, 'processed')
        versions = sorted((p.name for p in root.iterdir() if re.fullmatch(r'v[1-9][0-9]*', p.name)),
                          key=lambda v: int(v[1:]))
        for version in versions:
            self.get_version(dataset_id, version)
        return versions

    @contextmanager
    def processed_version(self, dataset_id: str, *, recipe: dict,
                          source_version: str = 'raw', schema: dict[str, str] | None = None) -> Iterator[VersionDraft]:
        """Publish a processed version only if the caller finishes successfully.

        This stores caller-produced files; it does not execute or approve recipes.
        The dataset write lock is held until the context exits. Close output writers
        before leaving the context. On an exception the temporary output is removed.
        """
        if not isinstance(recipe, dict):
            raise ValueError('recipe must be a JSON-serializable dictionary')
        with self._lock(dataset_id):
            dataset = self.get(dataset_id)
            if source_version == 'raw':
                source = dataset.directory / 'manifest.json'
            else:
                self.get_version(dataset_id, source_version)
                source = resolve_inside(dataset.directory, f'processed/{source_version}/manifest.json')
            source_digest = sha256_file(source)
            versions = self.list_versions(dataset_id)
            version = 'v'+str(max((int(v[1:]) for v in versions), default=0)+1)
            processed = resolve_inside(dataset.directory, 'processed')
            with TemporaryDirectory(prefix='.version-', dir=processed) as temp:
                staging = Path(temp)
                (staging / 'data').mkdir()
                write_json(staging / 'recipe.json', recipe)
                recipe_record = FileRecord.from_file(staging, staging / 'recipe.json')
                draft = VersionDraft(version, staging, dict(schema or {}))
                yield draft
                recipe_record.verify(staging)
                files = []
                for path in sorted(draft.data_dir.rglob('*')):
                    resolve_inside(staging, path.relative_to(staging).as_posix())
                    if path.is_file():
                        files.append(FileRecord.from_file(staging, path))
                manifest = ProcessedManifest(dataset_id, version, source_version, source_digest,
                                             utc_now(), tuple(files), recipe_record, dict(draft.schema))
                manifest.save(staging / 'manifest.json')
                target = resolve_inside(processed, version)
                if target.exists():
                    raise FileExistsError(target)
                _publish_directory(staging, target)
