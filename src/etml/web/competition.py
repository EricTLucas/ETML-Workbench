"""Project-local prediction assets and explicitly requested Kaggle submissions."""
import csv
import io
import json
import math
import re
import shutil
import subprocess
import uuid
import time
from pathlib import Path
from data.manifest import write_json, resolve_inside, validate_name
from workflows.project_models import ProjectModels


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def state(project):
    root=project.directory/'prediction-library'
    return {'files':[read(p) for p in sorted(root.glob('*/metadata.json'))],
            'presets':read(root/'presets.json',{}), 'competition':read(root/'competition.json',{}),
            'submissions':[read(p) for p in sorted((project.directory/'kaggle').glob('*/submission.json'))]}


def remember(project, source, filename):
    import pandas as pd
    pd.read_csv(source,nrows=1)
    ident='csv-'+uuid.uuid4().hex
    root=project.directory/'prediction-library'/ident
    root.mkdir(parents=True)
    shutil.copyfile(source,root/'input.csv')
    info={'id':ident,'name':Path(filename.replace('\\','/')).name}
    write_json(root/'metadata.json',info)
    return info


def saved_file(project, ident):
    path=resolve_inside(project.directory/'prediction-library',validate_name(ident))/'input.csv'
    if not path.is_file():raise ValueError('Saved prediction file not found.')
    return path


def preset(project,payload):
    from .csv_exports import ranges
    name=payload.get('name','').strip()
    if not name or len(name)>100:raise ValueError('Enter a preset name of up to 100 characters.')
    columns=payload.get('columns')
    if not isinstance(columns,list) or not columns or any(not isinstance(c,str) for c in columns):raise ValueError('Select export columns.')
    ranges(payload.get('include_rows',''));ranges(payload.get('exclude_rows',''))
    path=project.directory/'prediction-library'/'presets.json'
    presets=read(path,{})
    presets[name]={k:payload.get(k,'') for k in ('columns','include_rows','exclude_rows')}
    write_json(path,presets,overwrite=path.exists())
    return presets


def cli(args):
    executable=shutil.which('kaggle')
    if not executable:raise ValueError('Install the Kaggle CLI (python -m pip install kaggle) and configure Kaggle authentication first.')
    try:
        result=subprocess.run([executable,*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,shell=False)
    except subprocess.TimeoutExpired:
        raise ValueError('Kaggle timed out. Submission may have arrived; refresh status before submitting again.')
    if result.returncode:raise ValueError('Kaggle command failed. Check authentication, competition rules and your Kaggle submissions page. No automatic resubmission was attempted.')
    return result.stdout


def refresh(project, ident):
    path=resolve_inside(project.directory/'kaggle',validate_name(ident))/'submission.json'
    item=read(path)
    if not item:raise ValueError('Submission not found.')
    output=cli(['competitions','submissions',item['competition'],'-v'])
    rows=list(csv.DictReader(io.StringIO(output.strip().lstrip('\ufeff'))))
    matches=[r for r in rows if (item.get('ref') and str(r.get('ref'))==item['ref']) or (not item.get('ref') and r.get('description')==item['description'])]
    if len(matches)==1:
        row=matches[0];item['ref']=str(row.get('ref',''));item['status']=row.get('status','pending')
        raw=row.get('publicScore','')
        try:score=float(raw)
        except (ValueError,TypeError):score=None
        item['public_score']=score if score is not None and math.isfinite(score) else None
        if item['public_score'] is not None:
            from workflows.model_lifecycle import saved_record
            store=ProjectModels(project);record=saved_record(project,item['model'])
            scores=record.get('competition_scores',{})
            if scores.get(item['competition'],{}).get('created_at',0)<=item.get('created_at',0):
                scores[item['competition']]={'score':item['public_score'],'submission_id':ident,'ref':item['ref'],'created_at':item.get('created_at',0)}
            store.update(item['model'],competition_scores=scores)
    else:item['note']='Matching submission not visible yet. Refresh later; unrelated submissions are ignored.'
    write_json(path,item,overwrite=True)
    return item


def submit(project, export_id, payload):
    competition=payload.get('competition','').strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,149}',competition):raise ValueError('Enter the competition URL name, such as titanic.')
    source=resolve_inside(project.directory/'exports',validate_name(export_id))
    origin=read(source/'origin.json')
    if not origin:raise ValueError('Generate a new prediction export to attach its model provenance.')
    if not (source/'predictions.csv').is_file():raise ValueError('Prediction CSV not found.')
    ident='submission-'+uuid.uuid4().hex
    description=str(payload.get('message',''))[:200]+' [ETML '+ident+']'
    root=project.directory/'kaggle'/ident;root.mkdir(parents=True)
    shutil.copyfile(source/'predictions.csv',root/'submission.csv')
    item={'created_at':time.time(),'id':ident,'model':origin['model'],'competition':competition,'export_id':export_id,'description':description,'status':'submitting','public_score':None}
    path=root/'submission.json';write_json(path,item)
    pref=project.directory/'prediction-library'/'competition.json'
    write_json(pref,{'name':competition,'message':str(payload.get('message',''))[:200]},overwrite=pref.exists())
    try:
        output=cli(['competitions','submit','-c',competition,'-f',str(root/'submission.csv'),'-m',description])
        match=re.search(r'Submission ref:\s*(\d+)',output,re.I)
        item.update(status='pending',ref=match[1] if match else None)
        write_json(path,item,overwrite=True)
    except Exception:
        item['status']='unknown';write_json(path,item,overwrite=True);raise
    try:return refresh(project,ident)
    except ValueError:
        item['note']='Submitted. Score lookup unavailable; refresh later.'
        write_json(path,item,overwrite=True);return item
