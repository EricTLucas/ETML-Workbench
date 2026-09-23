"""Versioned, project-owned input snapshots for non-tabular model families."""
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import pandas as pd
from .manifest import write_json, resolve_inside
from training.artifacts import staged_directory, seal, verify_artifacts


def _read(path,max_rows,max_bytes):
    from eda_tool.loader import open_dataset
    from preprocessing.transforms.base import batches
    parts=[]; rows=0; size=0
    with batches(lambda:open_dataset(path).iter_batches(batch_size=min(10000,max_rows+1))) as iterator:
        for part in iterator:
            rows+=len(part); size+=int(part.memory_usage(deep=True).sum())
            if rows>max_rows or size>max_bytes:
                raise ValueError('Input exceeds row/memory limits; narrow the dataset or explicitly raise limits')
            parts.append(part)
    if not parts:
        raise ValueError('Input is empty')
    return pd.concat(parts,ignore_index=True)


def _hub(spec,root,max_rows):
    from datasets import load_dataset
    source=spec['source']
    data=load_dataset(source['dataset'],name=source.get('config'),split=source.get('split','train'),
                      revision=source.get('revision'),streaming=True)
    rows=[]
    columns=spec['columns']; image_column=columns.get('image')
    for i,row in enumerate(data):
        if i>=max_rows:
            raise ValueError('Hub split exceeds max_rows; select a smaller split explicitly')
        result={name:row[name] for name in set(columns.values())}
        if image_column:
            from PIL import Image
            im=result[image_column]
            if not isinstance(im,Image.Image):
                raise ValueError('The configured Hub image column must decode to an image')
            path=root/f'hub-{i:08d}.png'
            im.save(path)
            result[image_column]=str(path)
        rows.append(result)
    return pd.DataFrame(rows)


def prepare_model_input(project,spec,*,max_rows=200000,max_bytes=512*1024**2):
    """Import CSV/parquet manifests or a Hub split; no test rows are used for fitting.

    Image manifests have a path and label column. Text manifests have text and,
    except for language modeling, target columns. Series have time/value columns.
    Recommendation manifests contain user/item and optional positive weight.
    """
    spec=dict(spec)
    modality=spec.get('modality'); columns=spec.get('columns',{})
    required={'image':{'image','target'},'text':{'text'},'time_series':{'time','value'},
              'recommendation':{'user','item'}}
    if modality not in required or not required[modality]<=set(columns):
        raise ValueError('Invalid modality or missing column mappings')
    if len(set(columns.values()))!=len(columns):
        raise ValueError('Column roles must refer to different columns')
    if max_rows<1 or max_bytes<1:
        raise ValueError('Input limits must be positive')
    seed=spec.get('seed',42)
    fractions=spec.get('fractions',[0.7,0.15,0.15])
    if len(fractions)!=3 or any(not np.isfinite(v) or v<0 for v in fractions) or not np.isclose(sum(fractions),1) or min(fractions[:2])<=0:
        raise ValueError('Fractions must be train/validation/test, sum to one, and include train and validation')
    supplied=bool(spec.get('validation'))
    if spec.get('test') and not supplied:
        raise ValueError('Explicit test input requires explicit validation input')
    if spec.get('source') and any(spec.get(k) for k in ('train','validation','test')):
        raise ValueError('Choose local files or a Hub source, not both')
    destination=project.directory/'inputs'/('input-'+uuid.uuid4().hex[:12])
    with staged_directory(destination) as root:
        raw=root/'raw'; raw.mkdir(); assets=root/'assets'; assets.mkdir()
        frames={}; total_bytes=0
        for split in ('train','validation','test'):
            path=spec.get(split)
            if split=='train' and spec.get('source'):
                if spec['source'].get('provider')!='huggingface':
                    raise ValueError('Only huggingface is supported as a remote model-input source')
                frame=_hub(spec,raw,max_rows)
                frame.to_parquet(raw/'hub.parquet',index=False)
                base=raw
            elif path:
                path=Path(path).expanduser().resolve()
                if path.stat().st_size>max_bytes:
                    raise ValueError('Raw file exceeds import byte budget')
                shutil.copy2(path,raw/(split+path.suffix))
                frame=_read(path,max_rows,max_bytes); base=Path(spec.get('image_root') or path.parent).resolve()
            elif split=='train':
                raise ValueError('A training manifest or Hub source is required')
            else:
                continue
            if frame.empty or not set(columns.values())<=set(frame):
                raise ValueError('Empty input or a mapped column is absent')
            frame=frame[list(columns.values())].copy()
            if frame.isna().any().any():
                raise ValueError('Mapped input fields cannot be missing; resolve missing values first')
            if modality=='image':
                from PIL import Image
                paths=[]
                for i,value in enumerate(frame[columns['image']]):
                    source=Path(str(value)).expanduser()
                    if not source.is_absolute(): source=base/source
                    source=source.resolve()
                    total_bytes+=source.stat().st_size
                    if total_bytes>max_bytes:
                        raise ValueError('Image import exceeds byte budget')
                    with Image.open(source) as im:
                        im.verify()
                    target=assets/f'{split}-{i:08d}{source.suffix.lower()}'
                    shutil.copy2(source,target); paths.append(target.relative_to(root).as_posix())
                frame[columns['image']]=paths
            elif modality=='text':
                frame[columns['text']]=frame[columns['text']].astype(str)
                if (frame[columns['text']].str.strip()=='').any(): raise ValueError('Text must not be empty')
            elif modality=='time_series':
                frame[columns['time']]=pd.to_datetime(frame[columns['time']],utc=True,errors='raise')
                frame=frame.sort_values(columns['time']).reset_index(drop=True)
                frame[columns['value']]=pd.to_numeric(frame[columns['value']],errors='raise')
                if not np.isfinite(frame[columns['value']]).all(): raise ValueError('Series values must be finite')
            else:
                for key in ('user','item'): frame[columns[key]]=frame[columns[key]].astype(str)
                weight=columns.get('weight')
                if weight:
                    frame[weight]=pd.to_numeric(frame[weight],errors='raise')
                    if not np.isfinite(frame[weight]).all() or (frame[weight]<=0).any():
                        raise ValueError('Implicit interaction weights must be finite and positive')
                    frame=frame.groupby([columns['user'],columns['item']],as_index=False)[weight].sum()
                else:
                    frame=frame.drop_duplicates([columns['user'],columns['item']])
            frames[split]=frame.reset_index(drop=True)
        if sum(len(f) for f in frames.values())>max_rows or sum(int(f.memory_usage(deep=True).sum()) for f in frames.values())>max_bytes:
            raise ValueError('Combined input exceeds row/memory budget')
        if not supplied:
            frame=frames['train']; n=len(frame)
            if modality=='time_series':
                a=int(n*fractions[0]); b=int(n*(fractions[0]+fractions[1]))
                frames={'train':frame.iloc[:a],'validation':frame.iloc[a:b],'test':frame.iloc[b:]}
            else:
                from sklearn.model_selection import train_test_split
                label=columns.get('target') if modality in {'image','text'} and spec.get('task','classification')=='classification' else None
                stratify=frame[label] if label else None
                train,rest=train_test_split(frame,train_size=fractions[0],random_state=seed,stratify=stratify)
                if fractions[2]:
                    val,test=train_test_split(rest,train_size=fractions[1]/sum(fractions[1:]),random_state=seed,
                                              stratify=rest[label] if label else None)
                else: val,test=rest,rest.iloc[:0]
                frames={'train':train,'validation':val,'test':test}
        if any(frames[k].empty for k in ('train','validation')):
            raise ValueError('Training and validation splits must be nonempty')
        if modality=='time_series':
            combined=pd.concat([frames[k] for k in ('train','validation','test') if k in frames])
            differences=combined[columns['time']].diff().dropna()
            if (differences<=pd.Timedelta(0)).any() or differences.nunique()!=1:
                raise ValueError('Time series must have unique, regular timestamps, with train < validation < test')
        if modality=='recommendation':
            seen=set()
            for split in ('train','validation','test'):
                if split not in frames: continue
                pairs=set(zip(frames[split][columns['user']],frames[split][columns['item']]))
                if pairs & seen: raise ValueError('User–item pairs must not overlap splits')
                seen|=pairs
        for split,frame in frames.items():
            (root/'splits').mkdir(exist_ok=True)
            frame.to_parquet(root/'splits'/f'{split}.parquet',index=False)
        metadata={'modality':modality,'columns':columns,'task':spec.get('task'),
                  'split_counts':{k:len(v) for k,v in frames.items()},'spec':spec,'seed':seed}
        write_json(root/'input.json',metadata)
        seal(root,'model_input',metadata)
    return destination


def load_model_input(path):
    path=Path(path).resolve()
    manifest=verify_artifacts(path,'model_input')
    return path,manifest['metadata']
