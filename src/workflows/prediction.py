from pathlib import Path
from tempfile import TemporaryDirectory
import os
from eda_tool.loader import open_dataset
from preprocessing.transforms.base import batches
from prediction import Predictor


def predict_file(bundle,source,output,*,batch_size=50_000,loader_options=None,probabilities=True,strict=False):
    predictor = Predictor.load(bundle)
    output = Path(output).resolve()
    if output.suffix.lower() != '.csv':
        raise ValueError('Prediction output must be a .csv file')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    dataset = open_dataset(source,**(loader_options or {}))
    rows, predicted = 0, 0
    with TemporaryDirectory(prefix='.predictions-',dir=output.parent) as temp:
        staging = Path(temp)/'predictions.csv'
        with staging.open('w',encoding='utf-8',newline='') as stream:
            wrote = False
            with batches(lambda:dataset.iter_batches(batch_size=batch_size)) as iterator:
                for frame in iterator:
                    result = predictor.predict(frame,probabilities=probabilities,strict=strict)
                    result['input_row'] += rows
                    result.to_csv(stream,index=False,header=not wrote)
                    rows += len(frame)
                    predicted += int(result.status.eq('predicted').sum())
                    wrote = True
            if not wrote:
                import pandas as pd
                result = predictor.predict(pd.DataFrame(columns=list(dataset.schema())),probabilities=probabilities,strict=strict)
                result.to_csv(stream,index=False)
        # Hard-link publication is exclusive, preserving a destination created by another process.
        os.link(staging,output)
    return {'output':str(output),'input_rows':rows,'predicted_rows':predicted,'excluded_rows':rows-predicted,
            'probability_columns':{f'probability_{i}':v for i,v in enumerate(predictor.classes)} if probabilities else {}}
