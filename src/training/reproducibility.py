"""Portable replay uses frozen prepared data, learned preprocessing, code and package pins."""
from pathlib import Path
from importlib import util,metadata
import hashlib
import json
import os
import platform
import shutil
import sys
from data.manifest import write_json,resolve_inside,sha256_file
from .artifacts import seal,staged_directory,verify_artifacts

PACKAGES=('data','eda_tool','etml','preprocessing','workflows','tasks','splitting','models','training','prediction','exporting')
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS','TF_ENABLE_ONEDNN_OPTS')


def source_files():
    for package in PACKAGES:
        root=Path(util.find_spec(package).origin).parent
        for path in sorted(root.rglob('*.py')):
            if '__pycache__' not in path.parts:
                yield package+'/'+path.relative_to(root).as_posix(),path


def source_digest():
    digest=hashlib.sha256()
    for name,path in source_files():
        digest.update(name.encode()); digest.update(b'\0'); digest.update(path.read_bytes())
    return digest.hexdigest()


def requirements(config):
    from packaging.requirements import Requirement
    roots=['numpy','pandas','scipy','pyarrow','scikit-learn','skops','matplotlib','wordcloud','packaging']
    if config.backend in {'xgboost','tensorflow','pytorch'}:
        roots.append('torch' if config.backend=='pytorch' else config.backend)
    found={}
    while roots:
        name=roots.pop()
        distribution=metadata.distribution(name)
        canonical=distribution.metadata['Name']
        if canonical.lower() in found: continue
        found[canonical.lower()]=(canonical,distribution.version)
        for text in distribution.requires or ():
            requirement=Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra':''}):
                roots.append(requirement.name)
    return '\n'.join(f'{name}=={version}' for name,version in sorted(found.values()))+'\n'


def write_setup(directory,config,origin):
    directory=Path(directory)
    report={'format_version':1,'python':platform.python_version(),'platform':platform.platform(),
            'machine':platform.machine(),'processor':platform.processor(),'seed':config.seed,
            'source_sha256':source_digest(),'thread_environment':{key:os.environ.get(key) for key in THREAD_ENV},
            'config':config.to_dict(),'preparation_manifest_sha256':origin['preparation_manifest_sha256'],
            'resumed_from':origin.get('resumed_from'),
            'guarantee':'Saved bundle reuses exact learned parameters. Retraining is verified by prediction parity; bitwise equality across hardware is not guaranteed.'}
    write_json(directory/'reproducibility.json',report)
    (directory/'requirements.txt').write_text(requirements(config),encoding='utf-8')
    (directory/'SETUP.md').write_text(f'''# Reuse this model

Use Python {report['python']} and a new virtual environment on the recorded platform:

```text
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\python -m pip install -r requirements.txt
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

Install the matching ETML Workbench source in that environment (`python -m pip install --no-deps -e /path/to/workbench`).
Then run `workbench models predict-one --bundle /path/to/this/bundle --values '{{"feature": 123}}'`, using your actual input columns.
Use `workbench models input-template --bundle /path/to/this/bundle` to get those columns.

The bundle contains fitted preprocessing, encoder, class mapping and exact trained parameters.
`reproducibility.json` records the configuration, seed, code fingerprint, platform and preparation fingerprint.
`requirements.txt` pins the active dependency versions (platform-specific).

For retraining, run `workbench models setup --bundle /path/to/this/bundle --output replay-package --workspace /path/to/datasets`
while the original prepared data and, for resumed runs, the original checkpoint are available.
That explicit export includes training/validation data and a source snapshot. It does not include held-out test rows.
Follow the generated package's SETUP.md. Hardware/library differences can change results; replay reports parity instead of promising bitwise retraining equality.
''',encoding='utf-8')


def export_reproduction(source,destination,*,workspace='datasets',max_bytes=2*1024**3):
    from data import DatasetWorkspace
    from prediction import Predictor
    from .checkpoints import load_checkpoint
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination==source or destination.is_relative_to(source): raise ValueError('Choose an output outside the bundle')
    if type(max_bytes) is not int or max_bytes<1: raise ValueError('max_bytes must be positive')
    predictor=Predictor.load(source)
    manifest=verify_artifacts(source,'model_bundle')
    if not (source/'reproducibility.json').exists():
        raise ValueError('This older bundle predates reproducibility metadata; train it again with this version')
    report=json.loads((source/'reproducibility.json').read_text(encoding='utf-8'))
    if source_digest()!=report['source_sha256']:
        raise ValueError('Workbench code differs from training. Use the original source before exporting a replay package.')
    ws=workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)
    origin=predictor.metadata
    ws.get_task_run(origin['dataset_id'],origin['task_id'],origin['preparation_run'],verify=True)
    prep=resolve_inside(ws.get(origin['dataset_id']).directory,
        f"tasks/{origin['task_id']}/runs/{origin['preparation_run']}")
    if sha256_file(prep/'manifest.json')!=origin['preparation_manifest_sha256']:
        raise ValueError('Prepared data differs from training')
    checkpoint=origin.get('resumed_from')
    if checkpoint: load_checkpoint(checkpoint)
    files=[(source/record['path'],'bundle/'+record['path']) for record in manifest['files']]
    files.append((source/'manifest.json','bundle/manifest.json'))
    files += [(prep/f'prepared/{name}/part-00000.parquet',f'data/{name}.parquet') for name in ('train','validation')]
    files += [(path,'source/'+name) for name,path in source_files()]
    if checkpoint:
        check_manifest=verify_artifacts(checkpoint,'training_checkpoint')
        files += [(Path(checkpoint)/record['path'],'checkpoint/'+record['path']) for record in check_manifest['files']]
        files.append((Path(checkpoint)/'manifest.json','checkpoint/manifest.json'))
    if sum(path.stat().st_size for path,_ in files)>max_bytes: raise ValueError('Replay package exceeds max_bytes')
    with staged_directory(destination) as staging:
        for path,relative in files:
            target=resolve_inside(staging,relative); target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
        shutil.copyfile(source/'requirements.txt',staging/'requirements.txt')
        entry="""from pathlib import Path
import json,os,sys
root=Path(__file__).resolve().parent
report=json.loads((root/'bundle/reproducibility.json').read_text(encoding='utf-8'))
for name,value in report['thread_environment'].items():
    if value is None: os.environ.pop(name,None)
    else: os.environ[name]=value
sys.path.insert(0,str(root/'source'))
from training.reproducibility import replay
result=replay(root)
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['matches'] else 1)
"""
        (staging/'reproduce.py').write_text(entry,encoding='utf-8')
        (staging/'SETUP.md').write_text(f'''# Reproduce this model

This package includes training/validation rows, the original model, fitted preprocessing and matching ETML source.
Keep it private if the dataset is private. Held-out test rows are not included.

Use Python {report['python']} on {report['platform']}. Package pins are specific to that environment.

Windows PowerShell:
```text
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python reproduce.py
```

macOS/Linux (a compatible platform may need different package wheels):
```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

No workbench install is needed: the script uses the included source snapshot.
The replay reuses frozen train-fitted preprocessing, trains on the saved training rows, and uses validation for early stopping.
For a resumed model it restores the included optimizer checkpoint.
It writes a new complete model bundle to `retrained/` and compares validation predictions/probabilities to the original.
`matches` uses rtol=1e-5 and atol=1e-6; `exact_predictions` reports exact array equality.
A hardware change can affect floating-point results. A mismatch exits with code 1; it is never silently called an exact reproduction.
The destination must not exist; move a previous retrained folder before replaying again.
''',encoding='utf-8')
        seal(staging,'reproduction_package',{'source_model_sha256':sha256_file(source/'manifest.json'),
                                            'contains_training_data':True,'test_data_included':False})
    return destination


def replay(package,*,output=None):
    import numpy as np
    import pandas as pd
    from models import ModelConfig,create_model
    from prediction import Predictor
    from .features import encode
    from .runner import _encode_target
    root=Path(package).resolve()
    verify_artifacts(root,'reproduction_package')
    bundle=root/'bundle'
    original=Predictor.load(bundle)
    settings=json.loads((bundle/'model_config.json').read_text(encoding='utf-8'))
    config=ModelConfig(**settings['config'])
    report=json.loads((bundle/'reproducibility.json').read_text(encoding='utf-8'))
    if platform.python_version()!=report['python']: raise ValueError('Use the Python version recorded in SETUP.md')
    for line in (root/'requirements.txt').read_text(encoding='utf-8').splitlines():
        name,expected=line.split('==',1)
        if metadata.version(name)!=expected: raise ValueError(f'Install pinned {name}=={expected} before replaying')
    train,val=pd.read_parquet(root/'data/train.parquet'),pd.read_parquet(root/'data/validation.parquet')
    schema=original.schema['feature_schema']; target=original.metadata['target']
    x_train,x_val=encode(original.encoder,train,schema),encode(original.encoder,val,schema)
    y_train,y_val=(_encode_target(train[target],original.classes),_encode_target(val[target],original.classes)) if original.task_type=='classification' else (train[target].to_numpy(dtype=float),val[target].to_numpy(dtype=float))
    adapter=create_model(config,original.task_type)
    resume=root/'checkpoint' if (root/'checkpoint').exists() else None
    # A resumed early-stopped run was explicitly continued by the original user.
    adapter.fit_validation(x_train,y_train,validation_data=(x_val,y_val),resume=resume,
                           reset_patience=original.metadata.get('reset_patience',False))
    expected,actual=original.adapter.predict(x_val),adapter.predict(x_val)
    exact=bool(np.array_equal(expected,actual)); matches=bool(np.allclose(expected,actual,rtol=1e-5,atol=1e-6))
    difference=float(np.max(np.abs(expected-actual)))
    if original.task_type=='classification':
        left,right=original.adapter.predict_proba(x_val),adapter.predict_proba(x_val)
        exact=exact and bool(np.array_equal(left,right))
        matches=matches and bool(np.allclose(left,right,rtol=1e-5,atol=1e-6))
        difference=max(difference,float(np.max(np.abs(left-right))))
    result={'matches':matches,'exact_predictions':exact,'max_absolute_difference':difference,
            'validation_rows':len(val),'rtol':1e-5,'atol':1e-6}
    destination=Path(output).resolve() if output else root/'retrained'
    with staged_directory(destination) as staging:
        old=verify_artifacts(bundle,'model_bundle')
        for item in old['files']:
            if item['path'] in {old['metadata']['model_file'],'architecture.json'}: continue
            target_path=resolve_inside(staging,item['path']); target_path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(bundle/item['path'],target_path)
        adapter.save(staging)
        write_json(staging/'replay_parity.json',result)
        seal(staging,'model_bundle',{**old['metadata'],'reproduced_from':sha256_file(bundle/'manifest.json')})
    return {**result,'bundle':str(destination)}
