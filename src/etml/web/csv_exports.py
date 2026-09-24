"""Filter prediction exports without rerunning inference or loading the full file."""
from pathlib import Path
import re
import uuid
import numpy as np
import pandas as pd
from data.manifest import resolve_inside, validate_name

def ranges(text):
    if not isinstance(text,str) or len(text)>10000: raise ValueError('Row selection must be a short list of row numbers or ranges.')
    result=[]
    for part in text.split(','):
        if not part.strip():continue
        match=re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?',part)
        if not match:raise ValueError('Use row numbers or ranges, such as 1, 3-10.')
        a=int(match[1]);b=int(match[2] or a)
        if a<1 or b<a:raise ValueError('Rows start at 1; range ends must not precede starts.')
        result.append((a,b))
    return result

def filter_csv(project, export_id, payload):
    source=resolve_inside(project.directory/'exports',validate_name(export_id))/'predictions.csv'
    if not source.is_file():raise ValueError('Prediction export not found.')
    columns=payload.get('columns')
    available=pd.read_csv(source,nrows=0).columns.tolist()
    if not isinstance(columns,list) or not columns or len(set(columns))!=len(columns) or any(c not in available for c in columns):
        raise ValueError('Select at least one valid output column, without duplicates.')
    include=ranges(payload.get('include_rows',''));exclude=ranges(payload.get('exclude_rows',''))
    new_id='predictions-'+uuid.uuid4().hex;root=project.directory/'exports'/new_id;root.mkdir(parents=True)
    target=root/'predictions.csv';offset=count=0
    try:
        # String reads preserve leading zeros and empty strings in the generated CSV.
        pd.DataFrame(columns=columns).to_csv(target,index=False)
        with pd.read_csv(source,chunksize=10000,dtype=str,keep_default_na=False) as reader:
            for frame in reader:
                positions=np.arange(offset+1,offset+len(frame)+1);offset+=len(frame)
                mask=np.zeros(len(frame),dtype=bool) if include else np.ones(len(frame),dtype=bool)
                for a,b in include:mask|=(positions>=a)&(positions<=b)
                for a,b in exclude:mask&=~((positions>=a)&(positions<=b))
                selected=frame.loc[mask,columns];count+=len(selected)
                selected.to_csv(target,index=False,header=False,mode='a')
        if any(b>offset for _,b in include+exclude):raise ValueError(f'Row selection exceeds the {offset} source rows.')
    except Exception:
        target.unlink(missing_ok=True);raise
    return {'id':new_id,'rows':count,'columns':columns,'name':'filtered-predictions.csv',
            'url':'/downloads/'+project.name+'/'+new_id+'/predictions.csv'}
