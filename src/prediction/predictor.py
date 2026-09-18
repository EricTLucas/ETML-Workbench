from importlib.metadata import version
from pathlib import Path
import json
import numpy as np
import pandas as pd

from models import ModelConfig, create_model
from models.adapters.sklearn import load_safe
from preprocessing import FittedRecipe
from training.artifacts import verify_artifacts, require_files
from training.features import encode
from .validation import validate_input


class Predictor:
    def __init__(self, adapter, encoder, fitted, schema, classes, task_type, metadata=None):
        self.adapter, self.encoder, self.fitted = adapter, encoder, fitted
        self.schema, self.classes, self.task_type = schema, classes, task_type
        self.metadata = metadata or {}

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        manifest = verify_artifacts(root,'model_bundle')
        require_files(manifest,['model_config.json','encoder.skops','preprocessing.json','schema.json',
                                'labels.json','environment.json',manifest['metadata']['model_file']])
        read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
        saved_environment = read('environment.json')
        if saved_environment.get('scikit-learn') != version('scikit-learn'):
            raise ValueError('scikit-learn version differs from the bundle; use its recorded environment')
        settings = read('model_config.json')
        config = ModelConfig(**settings['config'])
        adapter = create_model(config,settings['task_type']).load(root)
        fitted = FittedRecipe.from_dict(read('preprocessing.json'))
        schema = read('schema.json')
        if list(fitted.input_columns) != schema['raw_columns']:
            raise ValueError('Preprocessing input schema mismatch')
        return cls(adapter,load_safe(root/'encoder.skops'),fitted,schema,read('labels.json')['classes'],
                   settings['task_type'],manifest['metadata'])

    def encoded(self, frame, *, strict=False):
        raw = validate_input(frame,self.schema['raw_columns'],strict=strict)
        prepared = self.fitted.apply(raw)
        if not prepared.index.is_unique or not prepared.index.isin(raw.index).all():
            raise ValueError('Preprocessing changed row identity')
        if not len(prepared):
            return prepared.index, None
        matrix = encode(self.encoder,prepared,self.schema['feature_schema'])
        return prepared.index, matrix

    def predict(self, frame, *, probabilities=True, strict=False):
        index, matrix = self.encoded(frame,strict=strict)
        output = pd.DataFrame({'input_row':np.arange(len(frame)),
                               'status':['excluded_by_preprocessing']*len(frame)})
        output['prediction'] = pd.Series([None]*len(frame),dtype=object)
        if self.task_type=='classification' and probabilities:
            for i in range(len(self.classes)):
                output[f'probability_{i}'] = np.nan
        if matrix is not None:
            values = self.adapter.predict(matrix)
            if self.task_type=='classification':
                codes = np.asarray(values,dtype=int)
                if not np.array_equal(codes,values) or (codes<0).any() or (codes>=len(self.classes)).any():
                    raise ValueError('Model emitted an unknown class')
                values = [self.classes[i] for i in codes]
                if probabilities:
                    proba = self.adapter.predict_proba(matrix)
                    if (proba.shape != (len(index),len(self.classes)) or not np.isfinite(proba).all()
                            or (proba < 0).any() or not np.allclose(proba.sum(axis=1),1.,atol=1e-5)):
                        raise ValueError('Invalid probability output')
                    for i in range(len(self.classes)):
                        output.loc[index,f'probability_{i}'] = proba[:,i]
            elif not np.isfinite(values).all():
                raise ValueError('Nonfinite model predictions')
            output.loc[index,'prediction'] = values
            output.loc[index,'status'] = 'predicted'
        return output
