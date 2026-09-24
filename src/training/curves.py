"""Portable learning-curve images from recorded history only."""
import base64
import math
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def learning_curves(payload):
    history=payload.get('history',payload.get('rows',[])) if isinstance(payload,dict) else payload
    series={}; unit='Epoch'
    if isinstance(history,list):
        for i,row in enumerate(history):
            if not isinstance(row,dict): continue
            for key,value in row.items():
                if 'loss' in key and isinstance(value,(int,float)) and math.isfinite(value):
                    series.setdefault(key,[]).append((row.get('epoch',row.get('round',i+1)),value))
    elif isinstance(history,dict):
        unit='Boosting round'
        for split,metrics in history.items():
            if not isinstance(metrics,dict): continue
            for key,values in metrics.items():
                if 'loss' not in key or not isinstance(values,list): continue
                label=('train' if split=='validation_0' else 'validation' if split=='validation_1' else split)+'_'+key
                series[label]=[(i+1,v) for i,v in enumerate(values) if isinstance(v,(int,float)) and math.isfinite(v)]
    series={k:v for k,v in series.items() if v}
    if not series:return []
    figure=Figure(figsize=(9,4.8),facecolor='#100f17');FigureCanvasAgg(figure)
    ax=figure.subplots();ax.set_facecolor('#100f17')
    for (label,points),color in zip(series.items(),['#c084fc','#7ce1df','#ff82b2','#ebcb8b']*10):
        ax.plot(*zip(*points),label=label.replace('_',' ').title(),color=color,linewidth=2,marker='.' if len(points)<30 else None)
    ax.set(xlabel=unit,ylabel='Loss',title='Loss over '+unit.lower()+'s')
    ax.tick_params(colors='#eee8f7')
    for t in (ax.title,ax.xaxis.label,ax.yaxis.label):t.set_color('#eee8f7')
    for spine in ax.spines.values():spine.set_color('#51465e')
    ax.grid(alpha=.15,color='#c084fc');ax.legend(facecolor='#201929',labelcolor='#eee8f7')
    figure.tight_layout();stream=BytesIO();figure.savefig(stream,format='png',dpi=130)
    figure.clear()
    return [{'title':'Loss over '+unit.lower()+'s','image':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')}]
