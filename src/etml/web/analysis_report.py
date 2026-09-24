"""Self-contained project analysis report using the selected comparison cohort."""
from html import escape
import json
import uuid
from eda_tool.workflows import run_eda
from eda_tool.profiler import ProfileConfig


def text(value):return escape(str(value),quote=True)

def table(values):
    return '<table><tbody>'+''.join('<tr><th>'+text(k)+'</th><td>'+text(v)+'</td></tr>' for k,v in values.items() if not isinstance(v,(dict,list)))+'</tbody></table>'

REPORT_STYLE = """
pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}
.report-box{border:1px solid #393044;background:#15121d;border-radius:16px;margin:20px 0;overflow:hidden}
.report-box-heading{padding:22px 26px;border-bottom:1px solid #393044}
.report-box-heading h3{font-size:21px;margin:0 0 6px}
.report-box-heading p{color:#b9afc6;margin:0;font-size:14px}
.report-box-body{padding:24px 26px}
.report-test{border-color:#7ce1df;background:linear-gradient(135deg,#152c30,#20152e)}
.report-test h3,.report-test .report-metric strong{color:#7ce1df}
.report-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px}
.report-metric{background:#201929;border:1px solid #332b40;border-radius:12px;padding:20px;min-width:0}
.report-metric small{display:block;color:#b9afc6;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.report-metric strong{display:block;font-size:27px;font-weight:500;margin-top:8px;overflow-wrap:anywhere}
.report-metric strong.report-unavailable{font-size:18px}
@media(max-width:600px){.report-box-heading,.report-box-body{padding:18px}.report-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

def metric_box(label, values, note, *, highlight=False):
    cards=[]
    for key,value in values.items():
        if isinstance(value,(dict,list)):continue
        shown='Unavailable' if value is None else format(value,'.5g') if isinstance(value,float) else str(value)
        cards.append('<div class="report-metric"><small>'+text(key.replace('_',' ').title())+'</small><strong'+(' class="report-unavailable"' if value is None else '')+'>'+text(shown)+'</strong></div>')
    body='<div class="report-metrics">'+''.join(cards)+'</div>' if cards else '<p>No '+text(label.lower())+' are available for this run.</p>'
    return '<div class="report-box'+(' report-test' if highlight else '')+'"><div class="report-box-heading"><h3>'+text(label)+'</h3><p>'+text(note)+'</p></div><div class="report-box-body">'+body+'</div></div>'

def export_analysis(models, project_name, payload):
    project=models.service.store.get(project_name)
    groups=models.compare(project_name)['groups']
    reference=payload.get('reference')
    group=next((g for g in groups if g['reference']==reference),None)
    if reference is not None and group is None:raise ValueError('This dataset/split has no comparable saved results. Refresh Compare.')
    if reference is None and groups:raise ValueError('Select a dataset and split in Compare before exporting.')
    dataset_id=reference['dataset'] if reference else payload.get('dataset_id')
    prep=models.service.preparation(project_name,dataset_id)
    result=run_eda(prep.source(),export_html=True,generate_summary=True,
                   profile_config=ProfileConfig(batch_size=50000,sample_size=5000),
                   title=project_name+' — Analysis summary')
    try:html=result.html
    finally:result.close()
    content='<section><h2>Model leaderboard</h2><p>Held-out test results for one dataset and split. Models without test results are not ranked. Choosing models repeatedly using test results compromises an independent test estimate.</p>'
    if group:
        metric='accuracy' if any('accuracy' in m['metrics'] for m in group['models']) else 'rmse'
        candidates=[m for m in group['models'] if metric in m['metrics']]
        ranked=sorted(candidates,key=lambda m:((-m['metrics'][metric] if metric=='accuracy' else m['metrics'][metric]),m['name']))
        content+='<p>Ranked by test '+text(metric)+'. '+('Highest' if metric=='accuracy' else 'Lowest')+' is best. Ties use model name.</p>'
        content+='<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Model</th><th>Algorithm</th><th>'+text(metric)+'</th></tr></thead><tbody>'
        content+=''.join('<tr><td>'+str(i+1)+'</td><td>'+text(m['name'])+'</td><td>'+text(m['model'])+'</td><td>'+text(round(m['metrics'][metric],6))+'</td></tr>' for i,m in enumerate(ranked))+'</tbody></table></div></section>'
        content+='<div class="report-box"><div class="report-box-heading"><h3>Run details</h3><p>Dataset and split used for this comparison.</p></div><div class="report-box-body table-wrap">'+table(reference)+'</div></div>'
        if ranked:
            best=models.details(project_name,ranked[0]['name']);record=best['record'];details=best['details'];metrics=details['metrics']
            content+='<section><h2>Best model: '+text(record['name'])+'</h2><p>'+text(best['entry']['name'])+' — '+text(best['entry']['description'])+'</p>'+metric_box('Test results',record['test_evaluation']['metrics'],'Held-out evaluation of this saved model.',highlight=True)
            content+=metric_box('Validation results',metrics.get('validation',{}),'Used for tuning and model comparison.')
            content+=metric_box('Training results',metrics.get('training_metrics',{}),'In-sample metrics; these do not estimate held-out performance.')
            content+='<h3>Training overview</h3>'+table(metrics)+'<h3>Hyperparameters</h3><pre>'+text(json.dumps(record['params'],indent=2))+'</pre>'
            content+='<h3>Training input and split</h3>'+table(details.get('training_origin',{}))
            curves=best.get('learning_curves',[])
            content+='<h3>Learning curves</h3>'+(''.join('<figure><img alt="'+text(c['title'])+'" src="'+c['image']+'"></figure>' for c in curves) if curves else '<p>No epoch loss history was recorded for this model.</p>')
            content+='<details><summary>Full configuration, metrics and environment</summary><pre>'+text(json.dumps({'record':record,'details':details},indent=2,default=str))+'</pre></details></section>'
    else:content+='<p>No evaluated models are available for this dataset yet.</p></section>'
    # The EDA renderer supplies escaped content, embedded images and its offline styles.
    html=html.replace('<footer>',content+'<footer>',1).replace('</style>',REPORT_STYLE+'</style>',1)
    export_id='analysis-'+uuid.uuid4().hex;root=project.directory/'exports'/export_id;root.mkdir(parents=True)
    (root/'analysis.html').write_text(html,encoding='utf-8')
    return {'url':'/downloads/'+project_name+'/'+export_id+'/analysis.html','name':project_name+'-analysis.html',
            'best_model':ranked[0]['name'] if group and ranked else None}
