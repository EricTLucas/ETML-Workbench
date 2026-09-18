from pathlib import Path
import json
import numpy as np
from data.manifest import write_json
from training.artifacts import staged_directory, seal


def export_onnx(source,destination,*,sample_data,max_dense_bytes=128*1024**2):
    from prediction import Predictor
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Export destination must be outside the source bundle')
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError('Install the onnx extra: pip install -e ".[onnx]"') from exc
    predictor = Predictor.load(source)
    settings = json.loads((Path(source)/'model_config.json').read_text(encoding='utf-8'))
    config = settings['config']
    if config['backend'] != 'sklearn' or config['algorithm'] not in {'linear','random_forest'}:
        raise ValueError('ONNX currently supports sklearn linear and random_forest models only')
    reference = sample_data.head(256)
    _, matrix = predictor.encoded(reference)
    if matrix is None:
        raise ValueError('No usable reference rows for ONNX parity verification')
    if matrix.shape[0]*matrix.shape[1]*4 > max_dense_bytes:
        raise ValueError('ONNX reference matrix exceeds the dense memory limit')
    values = matrix.toarray() if hasattr(matrix,'toarray') else matrix
    values = np.asarray(values,dtype=np.float32)
    options = {id(predictor.adapter.estimator):{'zipmap':False}} if predictor.task_type=='classification' else None
    graph = convert_sklearn(predictor.adapter.estimator,initial_types=[('features',FloatTensorType([None,values.shape[1]]))],
                            options=options,target_opset=17)
    content = graph.SerializeToString()
    session = ort.InferenceSession(content,providers=['CPUExecutionProvider'])
    output = session.run(None,{'features':values})
    expected = predictor.adapter.predict(matrix)
    actual = np.asarray(output[0]).reshape(-1)
    if predictor.task_type=='classification':
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_allclose(output[1],predictor.adapter.predict_proba(matrix),rtol=1e-4,atol=1e-5)
    else:
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    with staged_directory(destination) as staging:
        (staging/'model.onnx').write_bytes(content)
        write_json(staging/'input_schema.json',{'input':'encoded feature matrix; preprocessing not included',
                                               'name':'features','dtype':'float32','shape':[None,int(values.shape[1])],
                                               'features':predictor.schema['encoded_feature_names']})
        write_json(staging/'labels.json',{'classes':predictor.classes,'classification_output':'integer class codes'})
        write_json(staging/'parity.json',{'verified':True,'rows':len(values),'rtol':1e-4,'atol':1e-5})
        seal(staging,'onnx_model',{'preprocessing_included':False,'backend':config['backend'],'algorithm':config['algorithm']})
    return Path(destination).resolve()
