from pathlib import Path
import shutil
from data.manifest import resolve_inside
from training.artifacts import staged_directory, verify_artifacts


def export_bundle(source, destination):
    from prediction import Predictor
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Export destination must be outside the source bundle')
    manifest = verify_artifacts(source,'model_bundle')
    Predictor.load(source)
    with staged_directory(destination) as staging:
        for record in manifest['files']:
            src = resolve_inside(source,record['path'])
            target = resolve_inside(staging,record['path'])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,target)
        shutil.copyfile(source/'manifest.json',staging/'manifest.json')
        Predictor.load(staging)
    return destination
