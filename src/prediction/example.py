import json
import pandas as pd
from .predictor import Predictor


def input_template(bundle):
    predictor=Predictor.load(bundle)
    types=predictor.schema.get('raw_dtypes',{})
    return {'values':{c:None for c in predictor.schema['raw_columns']},'dtypes':types,
            'instructions':'Replace nulls with values; null means missing. Supply raw feature values before preprocessing.'}


def predict_example(bundle,values):
    if not isinstance(values,dict): raise ValueError('One example must be a JSON object of column values')
    predictor=bundle if isinstance(bundle,Predictor) else Predictor.load(bundle)
    frame=pd.DataFrame([values])
    output=predictor.predict(frame).iloc[0]
    raw=frame.loc[:,predictor.schema['raw_columns']]
    prepared=predictor.fitted.apply(raw)
    result={'input':json.loads(raw.to_json(orient='records'))[0],
            'prepared_features':json.loads(prepared.to_json(orient='records'))[0] if len(prepared) else None,
            'status':str(output['status']),'prediction':output['prediction'],
            'task_type':predictor.task_type,'target':predictor.metadata.get('target')}
    if result['status']=='predicted' and predictor.task_type=='classification':
        result['probabilities']=[{'class':value,'probability':float(output[f'probability_{i}'])}
                                 for i,value in enumerate(predictor.classes)]
        result['probabilities'].sort(key=lambda item:item['probability'],reverse=True)
    target=predictor.metadata.get('target')
    if target in values and values[target] is not None:
        result['actual']=values[target]
        if result['status']=='predicted':
            result['correct']=bool(result['prediction']==values[target]) if predictor.task_type=='classification' else None
            if predictor.task_type=='regression': result['residual']=float(values[target])-float(result['prediction'])
    # Normalize NumPy scalars and missing values for API/CLI callers.
    return json.loads(json.dumps(result,default=lambda value:value.item() if hasattr(value,'item') else str(value),allow_nan=False))
