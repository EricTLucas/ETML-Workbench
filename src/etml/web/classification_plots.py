"""Bounded, reproducible classification views of a saved model's raw split."""
from pathlib import Path
import base64
import json
import uuid
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data.manifest import resolve_inside, validate_name, write_json
from workflows.model_inspection import ModelInspection


def saved_plots(project, model=None):
    result=[]
    for path in sorted((project.directory/'visualizations').glob('plot-*/plot.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if model is None or record['model']==model:
            image=path.with_name('plot.png').read_bytes()
            result.append(dict(record,image='data:image/png;base64,'+base64.b64encode(image).decode('ascii')))
    return result


def create_plot(project, name, payload):
    from prediction import Predictor
    from eda_tool.loader import open_dataset
    from models.dependencies import ensure_dependencies
    inspection=ModelInspection(project,name)
    if inspection.kind!='model_bundle' or inspection.manifest['metadata']['task_type']!='classification':
        raise ValueError('Column plots require a supervised tabular classification model.')
    ensure_dependencies(inspection.record['model'])
    predictor=Predictor.load(inspection.bundle)
    columns=payload.get('columns',[])
    if not isinstance(columns,list) or not 1<=len(columns)<=2 or len(set(columns))!=len(columns) or any(c not in predictor.schema['raw_columns'] for c in columns):
        raise ValueError('Choose one or two distinct feature columns.')
    split=payload.get('split','test');splits=inspection.splits()
    if split not in splits:raise ValueError('Choose an available saved split.')
    # Smallest random priorities form a uniform sample without retaining the split.
    rng=np.random.default_rng(42);sample=None;priorities=np.array([]);total=0
    for batch in open_dataset(splits[split]['path']).iter_batches(batch_size=10000):
        total+=len(batch);keys=rng.random(len(batch))
        combined=pd.concat([sample,batch],ignore_index=True) if sample is not None else batch.reset_index(drop=True)
        keys=np.concatenate([priorities,keys]);keep=np.argsort(keys,kind='stable')[:2000]
        sample=combined.iloc[keep].reset_index(drop=True);priorities=keys[keep]
    if sample is None or sample.empty:raise ValueError('No rows are available to visualize.')
    predictions=predictor.predict(sample,probabilities=False)
    target=inspection.manifest['metadata']['target']
    valid=predictions['status'].eq('predicted').to_numpy(copy=True);coordinates=[];ticks=[]
    for column in columns:
        values=sample[column]
        if pd.api.types.is_numeric_dtype(values):
            numbers=pd.to_numeric(values,errors='coerce').to_numpy(dtype=float)
            valid &= np.isfinite(numbers);coordinates.append(numbers);ticks.append(None)
        else:
            labels=sorted(values.dropna().astype(str).unique())
            if len(labels)>30:raise ValueError('Choose a categorical axis with at most 30 sampled categories, or a numeric column.')
            mapping={v:i for i,v in enumerate(labels)}
            numbers=values.astype(str).map(mapping).to_numpy(dtype=float)
            valid &= values.notna().to_numpy() & np.isfinite(numbers)
            coordinates.append(numbers);ticks.append(labels)
    if not valid.any():raise ValueError('No rows have usable axes and predictions.')
    figure=Figure(figsize=(12,5),facecolor='#100f17');FigureCanvasAgg(figure)
    axes=figure.subplots(1,2,sharex=True,sharey=True)
    colors=['#c084fc','#7ce1df','#ff82b2','#ebcb8b','#90acff','#ceB8ad']
    from matplotlib import colormaps
    classes=list(predictor.classes)
    # Class identities and point positions are identical in both panels.
    y=coordinates[1] if len(columns)==2 else np.random.default_rng(17).uniform(-.08,.08,len(sample))
    unknown=0
    for ax,label,values in zip(axes,['True classes','Predicted classes'],[sample[target] if target in sample else pd.Series([None]*len(sample)),predictions.prediction]):
        ax.set_facecolor('#100f17')
        assigned=np.zeros(len(sample),dtype=bool)
        for i,cls in enumerate(classes):
            mask=valid & values.eq(cls).fillna(False).to_numpy(dtype=bool);assigned|=mask
            color=colors[i] if i<len(colors) else colormaps['hsv']((i*.618)%1)
            ax.scatter(coordinates[0][mask],y[mask],s=22,alpha=.7,color=color,label=str(cls),edgecolors='none')
        missing=valid & ~assigned
        if missing.any():
            ax.scatter(coordinates[0][missing],y[missing],s=22,color='#a4adbc',label='Unknown / unlabeled')
            unknown=max(unknown,int(missing.sum()))
        ax.set_title(label,color='#eee8f7');ax.set_xlabel(columns[0],color='#eee8f7')
        if len(columns)==2:ax.set_ylabel(columns[1],color='#eee8f7')
        else:ax.set_yticks([]);ax.set_ylabel('Display jitter only',color='#a4adbc')
        for i,labels in enumerate(ticks):
            if labels is not None:
                if i==0:ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
                else:ax.set_yticks(range(len(labels)),labels)
        ax.tick_params(colors='#ddd6ee');ax.grid(alpha=.12,color='#c084fc')
        for spine in ax.spines.values():spine.set_color('#51465e')
        ax.legend(facecolor='#201929',labelcolor='#eee8f7',fontsize=8,loc='best',ncol=2 if len(classes)>6 else 1)
    title=name+' · '+split+' · '+' vs '.join(columns)
    figure.suptitle(title,color='#eee8f7')
    figure.tight_layout(rect=(0,0,1,.95))
    plot_id='plot-'+uuid.uuid4().hex;root=project.directory/'visualizations'/plot_id;root.mkdir(parents=True)
    figure.savefig(root/'plot.png',dpi=130);figure.clear()
    record={'id':plot_id,'model':name,'title':title,'columns':columns,'split':split,
            'reference':inspection.record['input'],'source_rows':total,'sampled_rows':len(sample),
            'plotted_rows':int(valid.sum()),'omitted_rows':int((~valid).sum()),'unknown_labels':unknown,
            'note':'Uniform sample (seed 42), up to 2,000 rows. Both panels use the same rows and class colors. '+('Vertical jitter is for visibility only.' if len(columns)==1 else ''),
            'include_summary':False}
    write_json(root/'plot.json',record)
    return dict(record,image='data:image/png;base64,'+base64.b64encode((root/'plot.png').read_bytes()).decode('ascii'))


def select_plot(project, plot_id, include):
    if type(include) is not bool:raise ValueError('Choose whether to include the graph.')
    path=resolve_inside(project.directory/'visualizations',validate_name(plot_id))/'plot.json'
    record=json.loads(path.read_text(encoding='utf-8'));record['include_summary']=include
    write_json(path,record,overwrite=True)
    return {'id':plot_id,'include_summary':include}

