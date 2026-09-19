from pathlib import Path
import json
import os
from tempfile import TemporaryDirectory
from data.manifest import resolve_inside,validate_name
from .artifacts import verify_artifacts


def read_history(run,candidate=None):
    candidate = validate_name(candidate or run['winner'])
    root = resolve_inside(Path(run['directory']),'candidates/'+candidate)
    manifest = verify_artifacts(root,'model_bundle')
    if not any(f['path']=='history.json' for f in manifest['files']):
        return {'candidate':candidate,'rows':[],'summary':{}}
    payload = json.loads((root/'history.json').read_text(encoding='utf-8'))
    rows = payload['history']
    if isinstance(rows,dict):
        flattened = {('train_' if split=='validation_0' else 'validation_')+metric:values
                     for split,metrics in rows.items() for metric,values in metrics.items()}
        count = len(next(iter(flattened.values()),[]))
        rows = [{'round':i+1,**{k:v[i] for k,v in flattened.items()}} for i in range(count)]
    return {'candidate':candidate,'rows':rows,'summary':payload['summary']}


def plot_history(payload,destination):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    root = Path(destination).resolve()
    if root.suffix.lower()!='.png' or not payload['rows']:
        raise ValueError('Learning curves require history and a new .png path')
    if root.exists():
        raise FileExistsError(root)
    rows = payload['rows']
    unit = 'epoch' if 'epoch' in rows[0] else 'round'
    keys = [k for k in rows[0] if k.startswith('train_')]
    metric = 'loss' if 'train_loss' in keys else keys[0].removeprefix('train_')
    figure = Figure(figsize=(9,5),facecolor='#100f17')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_facecolor('#100f17')
    for prefix,color in (('train_','#c084fc'),('validation_','#f0abfc')):
        key = prefix+metric
        if key in rows[0]:
            axes.plot([r[unit] for r in rows],[r[key] for r in rows],color=color,
                      linewidth=2.3,label=prefix.rstrip('_').capitalize())
    axes.set(title='Learning curve',xlabel=unit.capitalize(),ylabel=metric.replace('_',' ').capitalize())
    axes.tick_params(colors='#ddd6ee')
    for text in (axes.title,axes.xaxis.label,axes.yaxis.label): text.set_color('#eee8f7')
    for spine in axes.spines.values(): spine.set_color('#51465e')
    axes.grid(alpha=.12,color='#c084fc')
    axes.legend(facecolor='#201929',labelcolor='#eee8f7')
    figure.tight_layout()
    root.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=root.parent) as temp:
        staging = Path(temp)/'curve.png'
        figure.savefig(staging,dpi=150)
        os.link(staging,root)
    return root
