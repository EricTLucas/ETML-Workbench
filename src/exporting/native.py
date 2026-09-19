from pathlib import Path
import json
import shutil
from data.manifest import resolve_inside, write_json
from training.artifacts import staged_directory, verify_artifacts, seal


def export_native(source,destination):
    from prediction import Predictor
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Export destination must be outside the source bundle')
    manifest = verify_artifacts(source,'model_bundle')
    predictor = Predictor.load(source)
    filename = manifest['metadata']['model_file']
    schema = json.loads((source/'schema.json').read_text(encoding='utf-8'))
    with staged_directory(destination) as staging:
        target = resolve_inside(staging,filename)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(resolve_inside(source,filename),target)
        for name in ('labels.json','model_config.json','environment.json'):
            shutil.copyfile(source/name,staging/name)
        for name in ('architecture.json','history.json'):
            if (source/name).exists():
                shutil.copyfile(source/name,staging/name)
        write_json(staging/'input_schema.json',{'input':'encoded float32 model feature matrix, not raw rows',
                                               'features':schema['encoded_feature_names'],
                                               'preprocessing_included':False,
                                               'classification_output':'integer codes decoded using labels.json',
                                               'xgboost_iteration_range': [0,predictor.adapter.estimator.best_iteration+1]
                                               if hasattr(predictor.adapter,'estimator') and hasattr(predictor.adapter.estimator,'best_iteration') else None})
        seal(staging,'native_model',{'model_file':filename,'preprocessing_included':False})
    return destination
