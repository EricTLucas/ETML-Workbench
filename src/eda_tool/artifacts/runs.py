"""Saved EDA runs: portable summaries, explicit sample persistence, integrity checks."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import pandas as pd
from ..profiler.base import SectionResult
from . import codec


FORMAT_VERSION = 1


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.drive:
        raise ValueError('Run artifact paths must be relative and stay inside the run directory')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Run artifact escapes the run directory')
    return path


@dataclass
class SavedChart:
    kind: str
    path: Path
    title: str
    metadata: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)


@dataclass
class SavedRun:
    profile: dict
    manifest: dict
    output_dir: Path
    charts: list


def persisted_profile(profile):
    """Remove full row material while retaining aggregate statistics and warnings.

    Aggregates may still contain category values, example values, or top words;
    this is NOT anonymization. No mutation of the caller's profile occurs.
    """
    result = {}
    for key, value in profile.items():
        data = value.data
        if key == 'interactions':
            data = {**data, 'sample': None}
        elif key == 'summary':
            data = {**data, 'sample_values_first': None, 'sample_values_last': None}
        result[key] = SectionResult(value.name, data, dict(value.metadata))
    return result


def write_run(directory, profile, charts, manifest, *, save_sample=False, html=None):
    """Write a new run into the caller's private staging directory."""
    directory = Path(directory)
    payload = {'format': 'eda-profile', 'version': FORMAT_VERSION, 'sections': persisted_profile(profile)}
    codec.dump(directory/'profile.json', payload)
    files = ['profile.json']
    manifest = {**manifest, 'format': 'workbench-eda-run', 'version': FORMAT_VERSION,
                'profile': 'profile.json', 'sample': None, 'charts': [], 'html': None,
                'row_previews_saved': False}
    if save_sample:
        sample = profile.get('interactions')
        sample = sample.data.get('sample') if sample is not None else None
        if sample is None:
            raise ValueError('No profiler sample available; enable ProfileConfig.interactions to save a sample')
        sample.to_parquet(directory/'sample.parquet', index=False)
        files.append('sample.parquet')
        manifest['sample'] = 'sample.parquet'
        manifest['sample_rows'] = len(sample)
    for i, chart in enumerate(charts, 1):
        # Numbered names avoid collisions, path injection and duplicate requests.
        relative = f'charts/{i:03d}.png'
        chart.save(directory/relative)
        title = chart.figure.texts[0].get_text() if chart.figure.texts else chart.kind
        record = {'kind': chart.kind, 'title': title, 'metadata': chart.metadata, 'request': chart.request}
        meta_path = f'charts/{i:03d}.json'
        codec.dump(directory/meta_path, record)
        files.extend([relative, meta_path])
        manifest['charts'].append({'image': relative, 'metadata': meta_path})
    if html is not None:
        (directory/'report.html').write_text(html, encoding='utf-8')
        files.append('report.html')
        manifest['html'] = 'report.html'
    manifest['files'] = {name: digest(directory/name) for name in files}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return manifest


def load_run(directory, *, verify=True):
    """Load without revisiting the original dataset. Never deserializes pickle."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'workbench-eda-run' or manifest.get('version') != FORMAT_VERSION:
        raise ValueError('Unsupported saved-run format/version')
    required = [manifest['profile']]
    if manifest.get('sample'): required.append(manifest['sample'])
    for chart in manifest.get('charts', []):
        required.extend([chart['image'], chart['metadata']])
    if manifest.get('html'): required.append(manifest['html'])
    for name in required:
        path = inside(directory, name)
        expected = manifest.get('files', {}).get(name)
        if not expected:
            raise ValueError(f'No integrity entry for {name}')
        if verify and digest(path) != expected:
            raise ValueError(f'Integrity check failed for {name}')
    payload = codec.load(inside(directory, manifest['profile']))
    if payload.get('format') != 'eda-profile' or payload.get('version') != FORMAT_VERSION:
        raise ValueError('Unsupported profile format/version')
    profile = payload['sections']
    if manifest.get('sample'):
        sample = pd.read_parquet(inside(directory, manifest['sample']))
        if len(sample) != manifest.get('sample_rows'):
            raise ValueError('Sample row count does not match manifest')
        profile['interactions'].data['sample'] = sample
    charts = []
    for entry in manifest.get('charts', []):
        record = codec.load(inside(directory, entry['metadata']))
        charts.append(SavedChart(record['kind'], inside(directory, entry['image']), record['title'],
                                 record['metadata'], record['request']))
    return SavedRun(profile, manifest, directory, charts)
