"""Versioned, JSON-only metadata for local dataset storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

FORMAT_VERSION = 1
CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_name(value: str) -> str:
    """Portable single path component; reject traversal and Windows devices."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value):
        raise ValueError('Names must be 1-120 letters, digits, dots, underscores or hyphens')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)),
                *(f'LPT{i}' for i in range(10))}
    if value.endswith('.') or value.split('.')[0].upper() in reserved:
        raise ValueError(f'Nonportable name: {value!r}')
    return value


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Artifact paths must be relative POSIX paths')
    parts = value.split('/')
    if PurePosixPath(value).is_absolute() or any(p in {'', '.', '..'} for p in parts):
        raise ValueError('Invalid relative artifact path')
    for part in parts:
        validate_filename(part)
    return value


def validate_filename(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200 or value in {'.', '..'}:
        raise ValueError('Invalid filename')
    if any(ord(c) < 32 or c in '<>:"/\\|?*' for c in value) or value.endswith((' ', '.')):
        raise ValueError('Filename must be a portable basename')
    # Reuse device-name rejection without restricting spaces or Unicode in names.
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)),
                *(f'LPT{i}' for i in range(10))}
    if value.split('.')[0].upper() in reserved:
        raise ValueError('Reserved filename')
    return value


def resolve_inside(root: Path, relative: str) -> Path:
    relative_path(relative)
    root = Path(root).resolve()
    candidate = root.joinpath(*relative.split('/'))
    # Reject links even when their current targets happen to be inside the root.
    cursor = root
    for part in relative.split('/'):
        cursor = cursor / part
        if cursor.is_symlink() or (hasattr(cursor, 'is_junction') and cursor.is_junction()):
            raise ValueError(f'Linked artifact paths are unsupported: {relative}')
    if not candidate.resolve().is_relative_to(root):
        raise ValueError('Artifact escapes its workspace')
    return candidate


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, payload: dict, *, overwrite: bool = False) -> Path:
    """Write valid JSON. Replacement is atomic; callers serialize shared updates."""
    path = Path(path)
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
        return path
    fd, temp = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return path


def _schema(value: dict) -> None:
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError('schema must map column names to strings')


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self):
        relative_path(self.path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError('Invalid file size')
        if not isinstance(self.sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.sha256):
            raise ValueError('Invalid SHA-256 digest')

    @classmethod
    def from_file(cls, root: Path, path: Path) -> FileRecord:
        relative = Path(path).relative_to(root).as_posix()
        checked = resolve_inside(root, relative)
        return cls(relative, checked.stat().st_size, sha256_file(checked))

    def verify(self, root: Path) -> None:
        path = resolve_inside(root, self.path)
        if not path.is_file() or path.stat().st_size != self.size_bytes or sha256_file(path) != self.sha256:
            raise ValueError(f'Artifact missing or changed: {self.path}')


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    name: str
    created_at: str
    files: tuple[FileRecord, ...]
    schema: dict[str, str] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    kind: str = 'dataset'

    def __post_init__(self):
        validate_name(self.dataset_id)
        _validate_common(self, 'dataset')
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Dataset display name must be nonempty')
        if not self.files or any(not f.path.startswith('raw/') for f in self.files):
            raise ValueError('Dataset files must live under raw/')

    def save(self, path: Path, *, overwrite: bool = False) -> Path:
        return write_json(path, asdict(self), overwrite=overwrite)


@dataclass(frozen=True)
class ProcessedManifest:
    dataset_id: str
    version: str
    source_version: str
    source_manifest_sha256: str
    created_at: str
    files: tuple[FileRecord, ...]
    recipe: FileRecord
    schema: dict[str, str] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    kind: str = 'processed'
    status: str = 'complete'

    def __post_init__(self):
        validate_name(self.dataset_id)
        _validate_common(self, 'processed')
        if not re.fullmatch(r'v[1-9][0-9]*', self.version):
            raise ValueError('Invalid processed version')
        if self.source_version != 'raw' and not re.fullmatch(r'v[1-9][0-9]*', self.source_version):
            raise ValueError('Invalid source version')
        if not re.fullmatch(r'[0-9a-f]{64}', self.source_manifest_sha256):
            raise ValueError('Invalid source manifest digest')
        if self.status != 'complete' or not self.files or any(not f.path.startswith('data/') for f in self.files):
            raise ValueError('Processed manifests require completed data artifacts')
        if not isinstance(self.recipe, FileRecord) or self.recipe.path != 'recipe.json':
            raise ValueError('Processed recipe must be recipe.json')

    def save(self, path: Path) -> Path:
        return write_json(path, asdict(self))


def _validate_common(manifest, kind):
    if type(manifest.format_version) is not int or manifest.format_version != FORMAT_VERSION or manifest.kind != kind:
        raise ValueError('Unsupported manifest format')
    if not isinstance(manifest.created_at, str) or datetime.fromisoformat(manifest.created_at).tzinfo is None:
        raise ValueError('created_at must be an ISO timestamp with a timezone')
    _schema(manifest.schema)
    if not all(isinstance(f, FileRecord) for f in manifest.files):
        raise ValueError('Invalid artifact records')
    paths = [f.path.casefold() for f in manifest.files]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate artifact paths')


def load_manifest(path: Path) -> DatasetManifest | ProcessedManifest:
    """Read metadata only. File-content verification is a separate explicit step."""
    with Path(path).open(encoding='utf-8') as stream:
        payload = json.load(stream)
    try:
        payload['files'] = tuple(FileRecord(**item) for item in payload['files'])
        if payload.get('kind') == 'dataset':
            return DatasetManifest(**payload)
        if payload.get('kind') == 'processed':
            payload['recipe'] = FileRecord(**payload['recipe'])
            return ProcessedManifest(**payload)
        raise ValueError('Unknown manifest kind')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed manifest') from exc
