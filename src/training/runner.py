from dataclasses import dataclass
from pathlib import Path
import json
import uuid
from time import perf_counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data import DatasetWorkspace
from data.manifest import resolve_inside, write_json, sha256_file
from models import ModelConfig, create_model
from models.adapters.sklearn import dump_safe, load_safe
from preprocessing import FittedRecipe
from splitting.strategies import scalar_key
from .artifacts import environment, seal, staged_directory, verify_artifacts
from .features import infer_schema, fit_encoder, encode
from .evaluation import evaluate_predictions, validate_metric
from .checkpoints import checkpoint_writer, load_checkpoint


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    directory: Path
    winner: str
    bundle: Path
    metric: str
    leaderboard: list


def _load_frame(path, max_rows, max_bytes):
    file = pq.ParquetFile(path)
    try:
        metadata = file.metadata
        estimate = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        if metadata.num_rows > max_rows or estimate > max_bytes:
            raise ValueError('Split exceeds in-memory training limits; increase limits explicitly or use a future streaming backend')
        frame = file.read().to_pandas()
    finally:
        file.close()
    if frame.memory_usage(deep=True).sum() > max_bytes:
        raise ValueError('Loaded split exceeds max_bytes')
    return frame


def _encode_target(values, classes):
    mapping = {scalar_key(value):i for i,value in enumerate(classes)}
    try:
        return np.asarray([mapping[scalar_key(value)] for value in values],dtype=np.int64)
    except KeyError as exc:
        raise ValueError('Evaluation data contains a target class absent from training') from exc


def _reference_rows(preparation, columns):
    file = pq.ParquetFile(preparation/'splits/train/part-00000.parquet')
    try:
        batch = next(file.iter_batches(batch_size=64,columns=list(columns)),None)
        return batch.to_pandas() if batch is not None else pd.DataFrame(columns=columns)
    finally:
        file.close()


def _write_bundle(directory, adapter, encoder, fitted, config, task_type, classes, schema,
                  metrics, origin, reference, reference_matrix):
    from prediction import Predictor
    directory.mkdir()
    model_path = adapter.save(directory)
    dump_safe(encoder,directory/'encoder.skops')
    fitted.save(directory/'preprocessing.json')
    write_json(directory/'model_config.json',{'config':config.to_dict(),'task_type':task_type})
    write_json(directory/'schema.json',schema)
    write_json(directory/'labels.json',{'classes':classes,'probability_columns':{f'probability_{i}':v for i,v in enumerate(classes)}})
    write_json(directory/'metrics.json',metrics)
    write_json(directory/'environment.json',environment())
    from .reproducibility import write_setup
    write_setup(directory,config,origin)
    write_json(directory/'history.json',{'history':getattr(adapter,'history',[]),
                                         'summary':getattr(adapter,'training_summary',{})})
    metadata = {**origin,'model_file':model_path.relative_to(directory).as_posix()}
    original = Predictor(adapter,encoder,fitted,schema,classes,task_type).predict(reference)
    # Verify persisted components before committing the final bundle manifest.
    from models.adapters.sklearn import load_safe
    restored_predictor = Predictor(create_model(config,task_type).load(directory),load_safe(directory/'encoder.skops'),
                         FittedRecipe.load(directory/'preprocessing.json'),schema,classes,task_type)
    restored = restored_predictor.predict(reference)
    np.testing.assert_allclose(adapter.predict(reference_matrix),
                               restored_predictor.adapter.predict(reference_matrix),rtol=1e-6,atol=1e-7)
    pd.testing.assert_frame_equal(original,restored,check_exact=False,rtol=1e-6,atol=1e-7)
    write_json(directory/'roundtrip.json',{'verified':True,'rows':len(reference),'rtol':1e-6,'atol':1e-7})
    seal(directory,'model_bundle',metadata)
    Predictor.load(directory)


def train_models(workspace, dataset_id, task_id, preparation_run, *, configs=None, metric=None,
                 max_rows=200_000, max_bytes=512*1024**2, max_features=50_000, progress=None,
                 resume=None, reset_patience=False, experiment=None, label=None):
    for value in (max_rows,max_bytes,max_features):
        if type(value) is not int or value < 1:
            raise ValueError('Training limits must be positive integers')
    workspace = workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)
    manifest = workspace.get_task_run(dataset_id,task_id,preparation_run,verify=True)
    metadata = manifest.metadata
    task_type, target = metadata['task_type'],metadata['target']
    metric = metric or ('balanced_accuracy' if task_type=='classification' else 'rmse')
    direction = validate_metric(metric,task_type)
    dataset = workspace.get(dataset_id)
    preparation = resolve_inside(dataset.directory,f'tasks/{task_id}/runs/{preparation_run}')
    preparation_digest = sha256_file(preparation/'manifest.json')
    resumed = load_checkpoint(resume) if resume else None
    if resumed and (resumed['origin']['preparation_manifest_sha256']!=preparation_digest
                    or resumed['origin']['dataset_id']!=dataset_id or resumed['origin']['task_id']!=task_id):
        raise ValueError('Checkpoint preparation data or task does not match')
    features = metadata['feature_columns']
    train = _load_frame(preparation/'prepared/train/part-00000.parquet',max_rows,max_bytes)
    validation = _load_frame(preparation/'prepared/validation/part-00000.parquet',max_rows,max_bytes)
    if train.empty or validation.empty:
        raise ValueError('Training and validation must both be nonempty; test data is not used for selection')
    if train.memory_usage(deep=True).sum()+validation.memory_usage(deep=True).sum() > max_bytes:
        raise ValueError('Combined training and validation frames exceed max_bytes')
    fitted = FittedRecipe.load(preparation/'preprocessing/fitted.json')
    classes = []
    if task_type=='classification':
        classes = sorted([v.item() if hasattr(v,'item') else v for v in train[target].unique()],key=scalar_key)
        if len(classes)<2:
            raise ValueError('Classification needs at least two training classes')
        y_train,y_val = _encode_target(train[target],classes),_encode_target(validation[target],classes)
    else:
        y_train,y_val = train[target].to_numpy(dtype=float),validation[target].to_numpy(dtype=float)
        if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
            raise ValueError('Targets must be finite and observed')
    configs = list(configs) if configs is not None else [ModelConfig(algorithm='linear'),ModelConfig(algorithm='random_forest')]
    if not configs or not all(isinstance(c,ModelConfig) for c in configs):
        raise ValueError('Supply ModelConfig candidates')
    if resumed:
        if len(configs)!=1:
            raise ValueError('Resume exactly one neural candidate')
        original, requested = ModelConfig(**resumed['config']).to_dict(),configs[0].to_dict()
        original['params'].pop('epochs',None); requested['params'].pop('epochs',None)
        if original!=requested or resumed['classes']!=classes:
            raise ValueError('Checkpoint configuration/labels changed; only total epochs may change')
    if not resumed and not any(c.backend=='sklearn' and c.algorithm=='dummy' for c in configs):
        configs.insert(0,ModelConfig(algorithm='dummy'))
    run_id = 'train-'+uuid.uuid4().hex
    destination = resolve_inside(dataset.directory,f'tasks/{task_id}/training/{run_id}')
    reference = _reference_rows(preparation,fitted.input_columns)
    leaderboard = []
    with staged_directory(destination) as staging:
        (staging/'candidates').mkdir()
        for i,config in enumerate(configs):
            candidate_started=perf_counter()
            name = f'candidate-{i:03d}'
            if progress:
                progress({'stage':'training','candidate':name,'model':f'{config.backend}:{config.algorithm}'})
            # Configuration, training, or export failures abort atomically; never silently omit a requested model.
            adapter = create_model(config,task_type)
            schema = infer_schema(train[features],config.categorical_columns)
            scale = config.scale_numeric if config.scale_numeric is not None else config.algorithm in {'linear','mlp'}
            if resumed:
                schema = resumed['schema']['feature_schema']
                fitted = FittedRecipe.load(Path(resume)/'preprocessing.json')
                encoder = load_safe(Path(resume)/'encoder.skops')
            else:
                encoder = fit_encoder(train[features],schema,scale=scale,max_features=max_features)
            x_train = encode(encoder,train[features],schema,max_bytes)
            x_val = encode(encoder,validation[features],schema,max_bytes)
            input_schema = {'raw_columns':list(fitted.input_columns),'feature_schema':schema,
                            'encoded_feature_names':encoder.get_feature_names_out().tolist(),
                            'encoded_features':int(x_train.shape[1]),'extra_raw_columns':'ignored',
                            'categorical_unknowns':'all-zero encoding', 'numeric_missing':'training median fallback',
                            'excluded_rows':'returned with excluded_by_preprocessing status'}
            origin = {'dataset_id':dataset_id,'task_id':task_id,'preparation_run':preparation_run,
                      'reset_patience':bool(reset_patience),
                      'preparation_manifest_sha256':preparation_digest,'training_run':run_id,'candidate':name,
                      'target':target,'task_type':task_type,'selection_split':'validation'}
            latest_checkpoint = None
            save_checkpoint = None
            if config.backend in {'pytorch','tensorflow'}:
                checkpoint_root = resolve_inside(dataset.directory,f'tasks/{task_id}/checkpoints/{run_id}/{name}')
                writer = checkpoint_writer(checkpoint_root,{'origin':origin,'config':config.to_dict(),
                    'schema':input_schema,'classes':classes,'metric':metric},encoder,fitted)
                def save_checkpoint(model,state):
                    nonlocal latest_checkpoint
                    latest_checkpoint = writer(model,state)
                    return latest_checkpoint
            def on_progress(event):
                if progress:
                    progress({**event,'candidate':name})
            encoding_seconds=perf_counter()-candidate_started
            fit_started=perf_counter()
            adapter.fit_validation(x_train,y_train,validation_data=(x_val,y_val),progress=on_progress,
                checkpoint=save_checkpoint,resume=resume,reset_patience=reset_patience,max_bytes=max_bytes)
            fit_seconds=perf_counter()-fit_started
            probabilities = adapter.predict_proba(x_val) if task_type=='classification' else None
            scores = evaluate_predictions(y_val,adapter.predict(x_val),task_type,probabilities=probabilities,n_classes=len(classes))
            if scores.get(metric) is None:
                raise ValueError(f'{metric} is undefined on this validation split; choose another metric')
            record = {'candidate':name,'model':config.to_dict(),'validation':scores,
                      'fit_seconds':fit_seconds,'encoding_seconds':encoding_seconds,
                      'is_baseline':config.backend=='sklearn' and config.algorithm=='dummy',
                      'training':getattr(adapter,'training_summary',{}),
                      'checkpoint':str(latest_checkpoint) if latest_checkpoint else None}
            leaderboard.append(record)
            detail={}
            if task_type=='classification':
                from .evaluation import classification_details
                detail=classification_details(y_val,adapter.predict(x_val),classes)
            origin['resumed_from']=str(Path(resume).resolve()) if resume else None
            input_schema['raw_dtypes']={c:str(reference[c].dtype) for c in reference}
            _write_bundle(staging/'candidates'/name,adapter,encoder,fitted,config,task_type,classes,input_schema,
                          {'validation':scores,'validation_details':detail,'fit_seconds':fit_seconds,
                           'encoding_seconds':encoding_seconds,'training_rows':len(train),'validation_rows':len(validation)},
                          origin,reference,x_train[:64])
            record['total_seconds']=perf_counter()-candidate_started
            del x_train,x_val,adapter,encoder
        baseline=next((r for r in leaderboard if r['is_baseline']),None)
        for record in leaderboard:
            record['baseline_score']=baseline['validation'][metric] if baseline else None
            record['improvement_over_baseline']=(record['validation'][metric]-record['baseline_score'])*(1 if direction=='max' else -1) if baseline else None
        winner = sorted(leaderboard,key=lambda r:(-r['validation'][metric] if direction=='max' else r['validation'][metric],r['candidate']))[0]['candidate']
        write_json(staging/'selection.json',{'winner':winner,'metric':metric,'direction':direction,
                                            'selection_split':'validation','test_evaluated':False,'leaderboard':leaderboard,
                                            'label':label,'experiment':experiment,
                                            'resumed_from':str(Path(resume).resolve()) if resume else None})
        seal(staging,'training_run',{'dataset_id':dataset_id,'task_id':task_id,'preparation_run':preparation_run,
                                     'preparation_manifest_sha256':preparation_digest,'winner':winner,'label':label})
    return TrainingResult(run_id,destination,winner,destination/'candidates'/winner,metric,leaderboard)


def evaluate_test(workspace,dataset_id,task_id,training_run,*,candidate=None,max_rows=200_000,max_bytes=512*1024**2):
    if any(type(v) is not int or v < 1 for v in (max_rows,max_bytes)):
        raise ValueError('Evaluation limits must be positive integers')
    from prediction import Predictor
    from data.manifest import validate_name
    workspace = workspace if isinstance(workspace,DatasetWorkspace) else DatasetWorkspace(workspace)
    dataset = workspace.get(dataset_id)
    root = resolve_inside(dataset.directory,f'tasks/{validate_name(task_id)}/training/{validate_name(training_run)}')
    run = verify_artifacts(root,'training_run')
    metadata = run['metadata']
    if metadata['dataset_id']!=dataset_id or metadata['task_id']!=task_id:
        raise ValueError('Training run identity mismatch')
    preparation = metadata['preparation_run']
    manifest = workspace.get_task_run(dataset_id,task_id,preparation,verify=True)
    if not manifest.metadata.get('test_labeled',True):
        raise ValueError('The supplied test file has no target labels. Use models predict for predictions; test metrics require labels.')
    prep_root = resolve_inside(dataset.directory,f'tasks/{task_id}/runs/{preparation}')
    if sha256_file(prep_root/'manifest.json') != metadata['preparation_manifest_sha256']:
        raise ValueError('Preparation manifest changed since training')
    selected = validate_name(candidate or metadata['winner'])
    predictor = Predictor.load(resolve_inside(root,'candidates/'+selected))
    frame = _load_frame(prep_root/'prepared/test/part-00000.parquet',max_rows,max_bytes)
    if frame.empty:
        raise ValueError('Test split is empty')
    target = manifest.metadata['target']
    matrix = encode(predictor.encoder,frame,predictor.schema['feature_schema'],max_bytes)
    y = _encode_target(frame[target],predictor.classes) if predictor.task_type=='classification' else frame[target].to_numpy(dtype=float)
    proba = predictor.adapter.predict_proba(matrix) if predictor.task_type=='classification' else None
    scores = evaluate_predictions(y,predictor.adapter.predict(matrix),predictor.task_type,probabilities=proba,n_classes=len(predictor.classes))
    report = {'split':'test','candidate':selected,'rows':len(frame),'metrics':scores}
    output = root/'evaluations'/('test-'+uuid.uuid4().hex)
    with staged_directory(output) as staging:
        write_json(staging/'metrics.json',report)
        seal(staging,'evaluation',{'training_run':training_run,'candidate':selected})
    return {**report,'directory':str(output)}
