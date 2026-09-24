'use strict';
const ms={overview:null,selected:null,tab:'train',creating:false,category:'prediction',task:null,entry:null,started:null,lastJob:null};
const modelCategories={
  prediction:['Tabular prediction','classification,regression','tabular','Predict a class or a numeric value from your prepared dataset.'],
  exploration:['Clusters, projections & anomalies','clustering,projection,anomaly','tabular','Explore patterns using explicitly selected features.'],
  image:['Images','classification','image','CNNs, ResNet and Vision Transformers.'],
  text:['Text & language','classification,language_modeling,text_to_text','text','BERT, GPT-2 and T5 for different text tasks.'],
  series:['Time series','forecasting','time_series','ARIMA, LSTM and GRU for forecasting.'],
  recommendation:['Recommendations','recommendation','recommendation','SVD and ALS for user–item interactions.']
};
function resetModels(){Object.assign(ms,{overview:null,selected:null,tab:'train',creating:false,category:'prediction',task:null,entry:null,started:null,lastJob:null});}
async function showModels(){
  ms.overview=await api(endpoint()+'/models');
  state.section='models';ms.creating=false;ms.tab='train';ms.selected=null;ms.entry=null;
  renderProject();
  window.scrollTo(0,0);
}
function modelPath(){return endpoint()+'/models/'+encodeURIComponent(ms.selected.record.name);}
function objectHTML(value){return '<pre class="json-view">'+esc(JSON.stringify(value,null,2))+'</pre>';}
function keyValueHTML(values){
  return '<dl class="key-values">'+Object.entries(values||{}).filter(([,value])=>value!==undefined).map(([key,value])=>'<dt>'+esc(pretty(key))+'</dt><dd>'+esc(scalar(value))+'</dd>').join('')+'</dl>';
}
function pretty(key){return key.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());}
function scalar(value){if(value===null||value===undefined)return 'Unavailable';if(typeof value==='number')return Number.isInteger(value)?String(value):Number(value.toPrecision(5)).toString();return typeof value==='object'?JSON.stringify(value):String(value);}
function metricCards(values){
  return '<div class="metric-grid">'+Object.entries(values||{}).filter(([,v])=>v!==undefined&&(v===null||typeof v!=='object')).map(([k,v])=>'<div class="metric"><small>'+esc(pretty(k))+'</small><strong>'+esc(scalar(v))+'</strong></div>').join('')+'</div>';
}
function modelTabs(){
  return '<nav class="tabs model-tabs" aria-label="Model views">'+[['train','Train'],['results','Results'],['details','Model details'],['predictions','Predictions'],['compare','Compare'],['visualize','Visualize']].map(([key,label])=>'<button data-model-tab="'+key+'" class="'+(ms.tab===key?'active':'')+'">'+label+'</button>').join('')+'</nav>';
}
function renderModels(){
  const root=$('#data-content');
  if(!ms.overview){root.innerHTML='<div class="loading-block">Loading models…</div>';return;}
  const records=ms.overview.models;
  root.innerHTML='<div class="model-toolbar"><div><span class="eyebrow">Project models</span><p>'+records.length+' saved '+(records.length===1?'model':'models')+'</p></div><div class="row"><label>Review a saved model<select id="saved-model"><option value="">Choose a model</option>'+records.map(r=>'<option value="'+esc(r.name)+'" '+(r.name===ms.selected?.record.name?'selected':'')+'>'+esc(r.name)+' · '+esc(r.model)+' · '+esc(r.status)+'</option>').join('')+'</select></label><button id="new-model" class="primary">+ New model</button></div></div>'+modelTabs()+'<div id="model-content"></div>';
  bind('#saved-model',async e=>{if(e.target.value)await selectModel(e.target.value);},'change');
  bind('#new-model',()=>{ms.creating=true;ms.tab='train';ms.entry=null;renderModels();});
  $$('[data-model-tab]').forEach(b=>b.onclick=guarded(()=>{ms.tab=b.dataset.modelTab;renderModels();}));
  if(ms.tab==='train')renderTrain();
  else if(ms.tab==='compare')renderCompare().catch(error=>notify(error.message,true));
  else if(ms.tab==='visualize')renderVisualize().catch(error=>notify(error.message,true));
  else if(!ms.selected)$('#model-content').innerHTML='<section class="empty"><h2>Select a saved model</h2><p>Choose a model above or train a new one to see its results, details and predictions.</p></section>';
  else if(ms.tab==='results')renderResults();
  else if(ms.tab==='details')renderModelDetails();
  else renderPredictions();
}
async function selectModel(name,tab='results'){
  setBusy(true);
  try{ms.selected=await api(endpoint()+'/models/'+encodeURIComponent(name));ms.creating=false;ms.tab=tab;}
  finally{setBusy(false);}
  renderModels();
}
function renderTrain(){
  const root=$('#model-content');
  if(state.busy&&ms.started){root.innerHTML=progressHTML();updateModelProgress(ms.lastJob||{});return;}
  if(!ms.creating){
    root.innerHTML='<section class="empty"><div class="icon">◈</div><h2>Build your next model.</h2><p>Choose a model family, adjust its defaults and train. Or select a saved model above to review its results.</p><button id="begin-model" class="primary">Train a new model →</button></section>';
    bind('#begin-model',()=>{ms.creating=true;renderTrain();});return;
  }
  root.innerHTML='<section class="section">'+stepHead('1','Choose a model family','Recommendations use the task and input type, just like the CLI.')+'<div class="section-body"><div class="category-grid">'+Object.entries(modelCategories).map(([id,[label,,,description]])=>'<button class="category '+(ms.category===id?'selected':'')+'" data-category="'+id+'"><strong>'+label+'</strong><span>'+description+'</span></button>').join('')+'</div><div id="model-selection" style="margin-top:24px"></div></div></section><div id="model-settings"></div>';
  $$('[data-category]').forEach(button=>button.onclick=()=>{ms.category=button.dataset.category;ms.entry=null;ms.task=null;renderTrain();});
  renderModelSelection();
}
function renderModelSelection(){
  const [,tasks,modality]=modelCategories[ms.category];
  const choices=tasks.split(',');
  const ready=dataset()?.state?.split;
  if(ms.category==='prediction'){
    ms.task=dataset()?.state?.task?.task_type||'classification';
    if(!ready){
      $('#model-selection').innerHTML='<div class="callout">Prepare a dataset and create its training split in Data first.</div><button id="back-to-data" class="primary">Go to Data →</button>';
      $('#model-settings').innerHTML='';bind('#back-to-data',()=>{state.section='data';state.tab='preprocess';renderProject();});return;
    }
  }else if(!choices.includes(ms.task))ms.task=choices[0];
  const entries=ms.overview.catalog.filter(e=>e.modality===modality&&e.tasks.includes(ms.task)).sort((a,b)=>Number(b.recommended_tasks.includes(ms.task))-Number(a.recommended_tasks.includes(ms.task))||a.name.localeCompare(b.name));
  if(!entries.some(e=>e.key===ms.entry?.key))ms.entry=entries[0];
  $('#model-selection').innerHTML='<div class="grid"><label>Task<select id="model-task" '+(ms.category==='prediction'?'disabled':'')+'>'+choices.map(t=>'<option value="'+t+'" '+(t===ms.task?'selected':'')+'>'+pretty(t)+'</option>').join('')+'</select></label><label>Model<select id="model-key" aria-label="Model">'+entries.map(e=>'<option value="'+esc(e.key)+'" '+(e.key===ms.entry?.key?'selected':'')+'>'+esc(e.name)+(e.recommended_tasks.includes(ms.task)?' · Suggested':'')+'</option>').join('')+'</select></label></div><div class="callout">Suggestions are starting points, not measured winners. Compare validation results before selecting your final model.</div><div id="model-help"></div>';
  bind('#model-task',e=>{ms.task=e.target.value;ms.entry=null;renderModelSelection();},'change');
  bind('#model-key',e=>{ms.entry=entries.find(entry=>entry.key===e.target.value);renderModelHelp();renderSettings();},'change');
  renderModelHelp();renderSettings();
}
function renderModelHelp(){
  const e=ms.entry;
  $('#model-help').innerHTML='<h3>'+esc(e.name)+'</h3><p style="margin-top:8px">'+esc(e.description)+'</p>'+(e.limitations?'<p class="model-note">'+esc(e.limitations)+'</p>':'')+(e.extra?'<details class="model-note"><summary>Automatically installed if missing: '+esc(e.extra)+'</summary><code class="shortcut">python -m pip install -e &quot;.['+esc(e.extra)+']&quot;</code></details>':'<span class="badge" style="margin-top:12px">Included in the base installation</span>');
}
const parameterChoices={optimizer:['adam','adamw','sgd','rmsprop'],regularizer:['none','l1','l2'],kernel:['rbf','linear','poly','sigmoid'],weights:['uniform','distance'],linkage:['ward','complete','average','single'],covariance_type:['full','tied','diag','spherical']};
function parameterField(key,value){
  let choices=parameterChoices[key];
  if(key==='loss')choices=ms.task==='classification'?['auto','cross_entropy']:ms.entry.modality==='text'?['auto','cross_entropy']:['auto','mse','mae','huber'];
  if(typeof value==='boolean')choices=['true','false'];
  if(choices&&!choices.includes(String(value)))choices=[String(value),...choices];
  const id='param-'+key;
  return '<label>'+esc(pretty(key))+(choices?'<select id="'+id+'" aria-label="'+esc(pretty(key))+'" data-param="'+esc(key)+'">'+choices.map(c=>'<option '+(c===String(value)?'selected':'')+'>'+esc(c)+'</option>').join('')+'</select>':'<input id="'+id+'" aria-label="'+esc(pretty(key))+'" data-param="'+esc(key)+'" value="'+esc(typeof value==='string'?value:JSON.stringify(value))+'" '+(typeof value==='number'&&key!=='patience'?'type="number" step="any"':'type="text"')+'>')+(Array.isArray(value)?'<small>Comma-separated values inside brackets</small>':value===null||key==='patience'?'<small>Use null for no limit / no early stopping</small>':'')+'</label>';
}
function renderSettings(){
  const e=ms.entry;
  $('#model-settings').innerHTML='<form id="train-model-form"><section class="section">'+stepHead('2','Make it yours','Common defaults are filled in. Change only what you need.')+'<div class="section-body stack"><label>Model name<input id="model-name" required value="'+esc(ms.overview.default_name)+'" maxlength="80"></label><div class="grid three">'+Object.entries(e.defaults).map(([k,v])=>parameterField(k,v)).join('')+'</div>'+(!Object.keys(e.defaults).length?'<p>This model uses its standard library defaults.</p>':'')+'<details><summary>Additional hyperparameters</summary><p class="model-note">Optional JSON overrides for parameters supported by this model.</p><textarea id="extra-params" rows="3" spellcheck="false">{}</textarea></details></div></section><section class="section">'+stepHead('3','Training input','Saved models keep a reference to the exact input used for training.')+'<div class="section-body" id="training-input"></div></section><div class="actions"><button class="primary" type="submit">Train model →</button></div></form>';
  renderTrainingInput();
  bind('#train-model-form',async event=>{
    event.preventDefault();
    const params={};
    $$('[data-param]').forEach(input=>{
      const key=input.dataset.param, original=e.defaults[key], text=input.value.trim();
      if(typeof original==='string')params[key]=text;
      else {try{params[key]=JSON.parse(text);}catch{throw Error(pretty(key)+' needs a valid number, boolean, list or null.');}}
    });
    const extra=JSON.parse($('#extra-params').value);
    if(!extra||typeof extra!=='object'||Array.isArray(extra))throw Error('Additional hyperparameters must be a JSON object.');
    const body={name:$('#model-name').value.trim(),action:'new',key:e.key,task:ms.task,params:{...params,...extra}};
    if(ms.category==='prediction')body.dataset_id=dataset().id;
    else if(ms.category==='exploration'){
      body.source=$('#exploration-source').value;
      body.features=$('#exploration-features').value.split(',').map(v=>v.trim()).filter(Boolean);
    }else{
      const saved=$('#special-input').value;
      if(saved)body.input_id=saved;
      else{
        const columns={};$$('[data-input-role]').forEach(input=>{if(input.value.trim())columns[input.dataset.inputRole]=input.value.trim();});
        const spec={columns,train:$('#input-train').value.trim()};
        if($('#input-validation').value.trim())spec.validation=$('#input-validation').value.trim();
        if($('#input-test').value.trim())spec.test=$('#input-test').value.trim();
        spec.fractions=$('#input-fractions').value.split(',').map(Number);
        const advanced=JSON.parse($('#input-advanced').value);
        if(!advanced||typeof advanced!=='object'||Array.isArray(advanced))throw Error('Input configuration must be an object.');
        body.input_spec={...spec,...advanced};
      }
    }
    await runModelTraining(body);
  },'submit');
}
function renderTrainingInput(){
  if(ms.category==='prediction'){
    const d=dataset(),s=d.state.split;
    $('#training-input').innerHTML='<div class="grid three">'+Object.entries(s.prepared_counts).map(([k,v])=>'<div class="metric"><small>'+pretty(k)+'</small><strong>'+v+' rows</strong></div>').join('')+'</div><p class="model-note">Target: <strong>'+esc(d.state.task.target)+'</strong> · Recipe fitted on training rows only.</p><details><summary>Split reference</summary>'+objectHTML(s)+'</details>';return;
  }
  if(ms.category==='exploration'){
    $('#training-input').innerHTML='<div class="stack"><label>Source CSV / Parquet path<input id="exploration-source" required value="'+esc(dataset()?.sources?.[0]||'')+'"></label><label>Feature columns, comma separated<input id="exploration-features" required placeholder="age, income, visits"></label><div class="callout">All selected input rows will be fitted. Exclude target labels, identifiers and held-out test rows. This produces exploratory metrics, not validation accuracy.</div></div>';return;
  }
  const e=ms.entry,inputs=ms.overview.inputs.filter(i=>i.modality===e.modality&&i.task===ms.task);
  const roles={image:['image','target'],text:ms.task==='language_modeling'?['text']:['text','target'],time_series:['time','value'],recommendation:['user','item','weight']}[e.modality];
  $('#training-input').innerHTML='<div class="stack"><label>Model input<select id="special-input"><option value="">Import new input tables</option>'+inputs.map(i=>'<option value="'+esc(i.id)+'">'+esc(i.id)+'</option>').join('')+'</select></label><div id="new-special-input" class="stack"><label>Training table path<input id="input-train" placeholder="CSV / Parquet table on this computer"></label><div class="grid"><label>Validation table (optional)<input id="input-validation"></label><label>Test table (optional)<input id="input-test"></label></div><label>Fractions when splitting one table<input id="input-fractions" value="0.7, 0.15, 0.15"></label><div class="grid">'+roles.map(role=>'<label>'+pretty(role)+' column'+(role==='weight'?' (optional)':'')+'<input data-input-role="'+role+'" value="'+(Object.keys(dataset()?.columns||{}).includes(role)?role:'')+'"></label>').join('')+'</div><p class="model-note">'+(e.modality==='image'?'The image column contains file paths; image files will be copied into the project.':e.modality==='time_series'?'Rows are ordered by time. Choose a regular time index for ARIMA.':e.modality==='recommendation'?'Use positive user–item interactions, with an optional weight column.':'The text column contains inputs; target contains labels or output text.')+'</p><details><summary>Advanced input configuration / Hugging Face</summary><p class="model-note">Optional overrides using the CLI input configuration format, including a source object for Hugging Face.</p><textarea id="input-advanced" rows="4">{}</textarea></details></div></div>';
  bind('#special-input',event=>{$('#new-special-input').hidden=!!event.target.value;},'change');
}
function progressHTML(){
  return '<section class="section"><div class="section-head"><div class="step-title"><div class="spinner"></div><div><h2>Training in progress</h2><p id="training-phase">Preparing your model…</p></div></div><span id="training-time" class="badge">0 s</span></div><div class="section-body"><p class="model-note">Keep the server running. Epoch metrics appear when the backend reports them; other models report their training stage.</p><div id="training-events"></div></div></section>';
}
function updateModelProgress(jobState){
  ms.lastJob=jobState;
  if(!$('#training-events'))return;
  const events=jobState.events||[],last=events.at(-1);
  $('#training-phase').textContent=last?.message||[pretty(last?.stage||jobState.status||'Preparing'),last?.model||''].join(' · ');
  $('#training-time').textContent=Math.round((Date.now()-(ms.started||Date.now()))/1000)+' s';
  const epochs=events.filter(e=>e.stage==='epoch');
  if(epochs.length){
    const columns=['epoch',...new Set(epochs.flatMap(e=>Object.keys(e).filter(k=>!['stage','epoch','candidate','checkpoint'].includes(k))))];
    $('#training-events').innerHTML=tableHTML({columns,rows:epochs.slice(-20).map(e=>columns.map(c=>e[c]??null))});
  }else $('#training-events').innerHTML='<div class="callout">Waiting for the next backend update. No intermediate loss is available yet.</div>';
}
async function runModelTraining(body){
  ms.tab='train';ms.started=Date.now();ms.lastJob=null;
  $('#model-content').innerHTML=progressHTML();
  try{
    const result=await job(endpoint()+'/models/train',body);
    ms.overview=await api(endpoint()+'/models');await selectModel(result.name,'results');
  }catch(error){
    ms.overview=await api(endpoint()+'/models');
    if(ms.overview.models.some(r=>r.name===body.name))await selectModel(body.name,'results');
    else renderModels();
    throw error;
  }finally{ms.started=null;}
}
function savedActions(){
  const r=ms.selected.record;
  return (r.status!=='complete'?'<button id="retry-model" class="primary">Retry training with saved settings</button>':'')+(ms.selected.can_resume?'<form id="resume-model" class="row"><label>Additional epochs<input name="epochs" type="number" min="1" step="1" value="10" required style="max-width:120px"></label><button>Continue training</button></form>':'');
}
function bindSavedActions(){
  bind('#retry-model',()=>runModelTraining({action:'retry',name:ms.selected.record.name}));
  bind('#resume-model',async event=>{event.preventDefault();await runModelTraining({action:'resume',name:ms.selected.record.name,additional_epochs:Number(new FormData(event.target).get('epochs'))});},'submit');
}
function renderResults(){
  const selected=ms.selected,r=selected.record,details=selected.details;
  if(!details){
    $('#model-content').innerHTML='<section class="section"><div class="section-head"><h2>'+esc(r.name)+' · '+esc(r.status)+'</h2></div><div class="section-body"><p>This is a saved configuration. Training has not completed.</p>'+(r.error?'<div class="callout">'+esc(r.error)+'</div>':'')+(selected.entry.extra?'<code class="shortcut">python -m pip install -e &quot;.['+esc(selected.entry.extra)+']&quot;</code>':'')+'<div class="actions">'+savedActions()+'</div></div></section>';
    bindSavedActions();return;
  }
  const metrics=details.metrics;
  const test=r.test_evaluation?.metrics;
  if(selected.kind==='exploration_model'){
    $('#model-content').innerHTML='<section class="section"><div class="section-head"><div><div class="eyebrow">Exploration complete</div><h2>'+esc(r.name)+' · '+esc(selected.entry.name)+'</h2></div></div><div class="section-body">'+metricCards(metrics)+'<div class="callout">'+esc(metrics.protocol)+'</div>'+objectHTML(metrics)+'<div class="actions"><button id="exploration-predict" class="primary">Inspect assignments / predictions →</button></div></div></section>';
    bind('#exploration-predict',()=>{ms.tab='predictions';renderModels();});return;
  }
  const trainingSummary='<section class="section"><div class="section-head"><div><div class="eyebrow">Training complete</div><h2>'+esc(r.name)+' · '+esc(selected.entry.name)+'</h2></div><span class="badge">Saved</span></div><div class="section-body">'+metricCards({fit_seconds:metrics.fit_seconds,training_rows:metrics.training_rows,validation_rows:metrics.validation_rows,epochs_completed:metrics.epochs_completed??r.metrics?.training?.epoch})+'</div></section>';
  $('#model-content').innerHTML=[
    ['Test',test,'Held-out performance. Use validation for tuning to preserve an independent test estimate.'],
    ['Validation',metrics.validation,'Used for tuning and model comparison.'],
    ['Training',metrics.training_metrics,'In-sample metrics; these do not estimate held-out performance.']
  ].map(([label,values,note])=>'<section class="section '+(label==='Test'?'test-results':'')+'"><div class="section-head"><div><h2>'+label+' metrics</h2><p>'+note+'</p></div></div><div class="section-body">'+(values&&Object.keys(values).length?metricCards(values):'<p>'+esc(label==='Test'?(selected.kind==='model_bundle'?'No labeled test evaluation is available.':'This model adapter does not report aggregate held-out test metrics.'):label==='Validation'?'No validation metrics are available for this run.':'This adapter does not report final training metrics.')+'</p>')+'</div></section>').join('')+trainingSummary+'<section class="section"><div class="section-head"><h2>More results</h2></div><div class="section-body">'+(r.metrics?.baseline_score!==null&&r.metrics?.baseline_score!==undefined?metricCards({validation_baseline:r.metrics.baseline_score,improvement_over_baseline:r.metrics.improvement_over_baseline}):'')+(metrics.baseline?'<h3>Baseline</h3>'+metricCards(metrics.baseline):'')+(metrics.protocol?'<div class="callout">'+esc(metrics.protocol)+'</div>':'')+'<details><summary>All metrics and training history</summary>'+objectHTML({metrics,training:r.metrics,history:selected.history,test:r.test_evaluation})+'</details><div class="actions">'+(selected.kind==='model_bundle'&&selected.splits.test?'<button id="evaluate-model">Evaluate test split</button>':'')+savedActions()+'<button id="go-predict" class="primary">Try predictions →</button></div></div></section>';
  $('#model-content').insertAdjacentHTML('beforeend',learningCurvesHTML(selected));
  bindSavedActions();
  bind('#go-predict',()=>{ms.tab='predictions';renderModels();});
  bind('#evaluate-model',async()=>{await job(modelPath()+'/evaluate',{});await selectModel(r.name);});
}
function renderModelDetails(){
  const selected=ms.selected,r=selected.record,d=selected.details;
  $('#model-content').innerHTML='<section class="section"><div class="section-head"><div><h2>'+esc(r.name)+'</h2><p>'+esc(selected.entry.name)+' · '+esc(r.status)+'</p></div></div><div class="section-body stack"><p>'+esc(selected.entry.description)+'</p><h3>Hyperparameters</h3>'+keyValueHTML(r.params)+'<h3>Training input and split</h3>'+keyValueHTML(d?{target:d.training_origin?.target,task:d.configuration?.task_type||d.configuration?.task,fitted_on:d.fitting_split,dataset:d.training_origin?.dataset_id,training_rows:d.split_details?.prepared_counts?.train,validation_rows:d.split_details?.prepared_counts?.validation,test_rows:d.split_details?.prepared_counts?.test,created_at:r.created_at}:typeof r.input==='object'?r.input:{source:r.input})+'<details><summary>Complete model details, environment and history</summary>'+objectHTML({record:r,details:d})+'</details></div></section>'+(d?'<section class="section"><div class="section-head"><div><h2>Export this model</h2><p>Download a ZIP containing the prediction bundle, configuration and model details.</p></div></div><form id="export-model" class="section-body stack"><label class="check"><input type="checkbox" name="include_data">Include the associated input and split data</label><div class="actions"><button class="primary">Create export →</button></div><div id="export-result"></div></form></section>':'<div class="callout">Complete training before exporting a prediction bundle.</div>');
  bind('#export-model',async event=>{event.preventDefault();const result=await job(modelPath()+'/export',{include_data:new FormData(event.target).get('include_data')==='on'});$('#export-result').innerHTML='<div class="callout success">Export ready. <a href="'+esc(result.url)+'" download="'+esc(result.name)+'">Download '+esc(result.name)+'</a><p class="model-note">Also saved at '+esc(result.directory)+'</p></div>';},'submit');
}
function renderPredictions(){
  const s=ms.selected;
  if(!s.details){$('#model-content').innerHTML='<div class="callout">Complete or retry training before making predictions.</div>';return;}
  const fields=s.custom_fields, target=s.custom_target;
  const actualField=target?'<label>Known true value: '+esc(target.name)+' (optional)'+(target.task==='classification'?'<select id="custom-actual"><option value="">Not supplied</option>'+target.classes.map((v,i)=>'<option value="'+i+'">'+esc(v)+'</option>').join('')+'</select>':'<input id="custom-actual" type="number" step="any">')+'</label>':'';

  $('#model-content').innerHTML='<section class="section"><div class="section-head"><div><h2>Try your model</h2><p>Inspect an existing row or provide a new example.</p></div></div><div class="section-body"><div class="grid prediction-panels"><form id="predict-row" class="prediction-box stack"><h3>Dataset example</h3><label>Saved split<select name="split">'+Object.entries(s.splits).map(([k,v])=>'<option value="'+k+'">'+pretty(k)+' · '+v.rows+' rows</option>').join('')+'</select></label><label>Row number<input name="row" type="number" min="1" step="1" value="1" required></label><button class="primary" '+(!Object.keys(s.splits).length?'disabled':'')+'>Predict this row</button></form><section class="prediction-box"><h3>Custom example</h3><p class="model-note">Provide raw values, before preprocessing. Leave a field blank for a missing value.</p>'+(!s.custom_supported?'<div class="callout">This model only supports stored assignments for fitted rows. It cannot predict new examples.</div>':'<form id="predict-custom" class="stack">'+Object.entries(fields.values).map(([key,value])=>'<label>'+esc(key)+'<input data-custom="'+esc(key)+'" value="'+esc(value===null?'':typeof value==='object'?JSON.stringify(value):value)+'" placeholder="'+esc(fields.dtypes?.[key]||'value')+'"></label>').join('')+actualField+'<button class="primary">Predict custom example</button></form>')+'</section></div></div></section><div id="prediction-result"></div>';
  if(s.kind==='model_bundle'){
    $('#model-content').insertAdjacentHTML('beforeend','<section class="section"><div class="section-head"><div><h2>Predict a CSV</h2><p>Upload raw feature columns. Predictions populate '+esc(target.name)+'. Existing target values are preserved in a separate actual column. Rows excluded by preprocessing receive a blank prediction. After generation, choose columns and rows for your download.</p></div></div><form id="predict-csv" class="section-body stack"><input type="file" name="file" accept=".csv,text/csv" required aria-label="Prediction CSV"><button class="primary">Generate predictions CSV</button><div id="csv-result"></div></form></section>');
    bind('#predict-csv',async event=>{event.preventDefault();const result=await job(modelPath()+'/predict-csv',new FormData(event.target),true);renderCSVExport(result);},'submit');
  }
  const rowForm=$('#predict-row');
  function rowBounds(){const split=$('[name=split]',rowForm).value;$('[name=row]',rowForm).max=s.splits[split]?.rows||1;}
  rowBounds();bind('#predict-row [name=split]',rowBounds,'change');
  bind('#predict-row',async event=>{event.preventDefault();const f=new FormData(event.target);const result=await job(modelPath()+'/predict',{mode:'row',split:f.get('split'),row:Number(f.get('row')||1)});showPrediction(result);},'submit');
  bind('#predict-custom',async event=>{
    event.preventDefault();const values={};
    $$('[data-custom]').forEach(input=>{
      const key=input.dataset.custom,text=input.value,dtype=fields.dtypes?.[key]||'',initial=fields.values[key];
      if(!text.trim())values[key]=null;
      else if(/int|float|double|decimal/i.test(dtype)){values[key]=Number(text);if(!Number.isFinite(values[key]))throw Error(key+' must be a number.');}
      else if(/bool/i.test(dtype)){if(!['true','false','1','0'].includes(text.toLowerCase()))throw Error(key+' must be true or false.');values[key]=['true','1'].includes(text.toLowerCase());}
      else if(Array.isArray(initial)||typeof initial==='number')values[key]=JSON.parse(text);
      else values[key]=text;
    });
    if(target&&$('#custom-actual').value!==''){
      const value=$('#custom-actual').value;
      values[target.name]=target.task==='classification'?target.classes[Number(value)]:Number(value);
    }
    showPrediction(await job(modelPath()+'/predict',{mode:'custom',values}));
  },'submit');
}
function showPrediction(result){
  const prediction=result.prediction??result.embedding??result.forecast??result.recommendations??result.text;
  $('#prediction-result').innerHTML='<section class="section"><div class="section-head"><h2>Prediction result</h2><span class="badge">'+esc(result.status||'Predicted')+'</span></div><div class="section-body"><div class="grid"><div class="metric"><small>Model prediction</small><strong class="prediction-value">'+esc(scalar(prediction))+'</strong></div><div class="metric"><small>True value</small><strong>'+esc(result.actual===undefined?'Not supplied':scalar(result.actual))+'</strong></div></div>'+(result.correct!==undefined&&result.correct!==null?'<div class="callout '+(result.correct?'success':'')+'">'+(result.correct?'Matches the true value':'Does not match the true value')+'</div>':'')+(result.residual!==undefined?metricCards({residual:result.residual}):'')+(result.values||result.input?'<h3 style="margin-top:20px">Input values</h3>'+keyValueHTML(result.values||result.input):'')+'<details style="margin-top:18px"><summary>Probabilities and complete prediction output</summary>'+objectHTML(result)+'</details></div></section>';
  $('#prediction-result').scrollIntoView({behavior:'smooth'});
}

async function renderCompare(){
  const root=$('#model-content');root.innerHTML='<p>Loading model comparisons…</p>';
  const comparison=await api(endpoint()+'/models/compare');
  if(ms.tab!=='compare')return;
  root.innerHTML='<section class="section"><div class="section-head"><div><h2>Model leaderboard</h2><p>Results grouped by the same dataset and split. Training metrics are used only when test data is absent. Use validation for repeated tuning; choosing winners repeatedly on test scores makes the test estimate optimistic.</p></div></div><div class="section-body"><label>Dataset and split<select id="compare-group">'+comparison.groups.map((g,i)=>'<option value="'+i+'">'+esc(g.reference.dataset)+' · '+esc(g.reference.run_id)+' · '+g.models.length+' models · '+(g.evaluation_split==='train'?'Training fallback':'Test')+'</option>').join('')+'</select></label><label>Rank by<select id="compare-metric"></select></label><div id="leaderboard"></div><p class="model-note">'+comparison.unavailable.length+' models have no comparable metrics.</p></div></section>';
  function chooseGroup(){
    const group=comparison.groups[Number($('#compare-group').value)];
    if(!group){$('#leaderboard').innerHTML='<p>Train a model to compare its recorded results.</p>';return;}
    const metrics=[...new Set(group.models.flatMap(m=>Object.keys(m.metrics)))];
    $('#compare-metric').innerHTML=metrics.map(k=>'<option value="'+esc(k)+'">'+esc(pretty(k))+'</option>').join('');
    $('#compare-metric').value=metrics.includes('accuracy')?'accuracy':metrics.includes('rmse')?'rmse':metrics[0];
    draw();
  }
  function draw(){
    const group=comparison.groups[Number($('#compare-group').value)],metric=$('#compare-metric').value;
    if(!group)return;
    const lower=/^(rmse|mse|mae|mape|log_loss|loss|brier|median_absolute_error|fit_seconds)/.test(metric);
    const rows=[...group.models].sort((a,b)=>{const x=a.metrics[metric],y=b.metrics[metric];if(x===undefined)return 1;if(y===undefined)return -1;return (lower?1:-1)*(x-y)||a.name.localeCompare(b.name);});
    const fields=[metric,...[...new Set(rows.flatMap(m=>Object.keys(m.metrics)))].filter(k=>k!==metric)];
    $('#leaderboard').innerHTML=(group.evaluation_split==='train'?'<div class="callout">Training metrics fallback — no test data. These are in-sample scores, not held-out performance.</div>':'<p class="model-note">Evaluation source: test data</p>')+'<p class="model-note">'+(lower?'Lower':'Higher')+' is better. Unavailable scores are unranked.</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Model</th>'+fields.map(k=>'<th>'+esc(pretty(k))+'</th>').join('')+'</tr></thead><tbody>'+rows.map((m,i)=>'<tr><td>'+(m.metrics[metric]===undefined?'—':i+1)+'</td><td>'+esc(m.name)+'<br><small>'+esc(m.model)+'</small></td>'+fields.map(k=>'<td>'+esc(scalar(m.metrics[k]))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
  }
  bind('#compare-group',chooseGroup,'change');bind('#compare-metric',draw,'change');chooseGroup();
}

function learningCurvesHTML(selected){
  const curves=selected.learning_curves||[];
  if(!curves.length)return '';
  return '<section class="section"><div class="section-head"><h2>Learning curves</h2></div><div class="section-body">'+curves.map(c=>'<img class="learning-curve" src="'+esc(c.image)+'" alt="'+esc(c.title)+'">').join('')+'</div></section>';
}
function renderCSVExport(result){
  $('#csv-result').innerHTML='<div class="callout success">'+result.rows+' rows processed · '+result.excluded+' rows have no prediction. <a href="'+esc(result.url)+'" download="'+esc(result.name)+'">Download all predictions</a></div><h3>Choose export columns</h3><div class="column-options">'+result.columns.map(c=>'<label class="check"><input type="checkbox" data-export-column="'+esc(c)+'" checked>'+esc(c)+'</label>').join('')+'</div><div class="grid"><label>Include rows<input id="csv-include" placeholder="All rows, or 1, 3-10"></label><label>Exclude rows<input id="csv-exclude" placeholder="None, or 2, 11-15"></label></div><p class="model-note">Original CSV row numbers, starting at 1 (header excluded). Exclusions take priority. Filtering affects the download only.</p><button type="button" id="filter-csv">Create filtered CSV</button><div id="filtered-result"></div>';
  bind('#filter-csv',async()=>{const columns=$$('[data-export-column]').filter(c=>c.checked).map(c=>c.dataset.exportColumn);if(!columns.length)throw Error('Select at least one column.');const filtered=await job(endpoint()+'/prediction-exports/'+encodeURIComponent(result.id)+'/filter',{columns,include_rows:$('#csv-include').value,exclude_rows:$('#csv-exclude').value});$('#filtered-result').innerHTML='<div class="callout success">'+filtered.rows+' rows · '+filtered.columns.length+' columns. <a href="'+esc(filtered.url)+'" download="'+esc(filtered.name)+'">Download filtered CSV</a></div>';});
}

async function renderVisualize(){
  const root=$('#model-content'),selected=ms.selected;
  root.innerHTML='<section class="section"><div class="section-head"><h2>Visualize data</h2></div><div class="section-body" id="visualize-controls"></div></section><div id="classification-gallery"></div><section class="section"><div class="section-head"><h2>Export analysis summary</h2></div><div class="section-body stack"><p>EDA, the accuracy leaderboard and best-model results, plus selected graphs. Uses test metrics, or training metrics if no test data exists. Regression uses RMSE.</p><label>Summary dataset and split<select id="summary-group"></select></label><button id="export-analysis" class="primary">Generate HTML summary</button><div id="analysis-result"></div></div></section>';
  const supported=selected?.kind==='model_bundle'&&selected.custom_target?.task==='classification';
  if(supported){
    const columns=Object.keys(selected.custom_fields.values);
    $('#visualize-controls').innerHTML='<form id="classification-plot" class="stack"><p>Compare true and predicted classes on the same rows. Choose one feature for a 1D plot or two for a 2D plot.</p><div class="grid"><label>First column<select name="x">'+columns.map(c=>'<option>'+esc(c)+'</option>').join('')+'</select></label><label>Second column<select name="y"><option value="">None — 1D</option>'+columns.map(c=>'<option>'+esc(c)+'</option>').join('')+'</select></label><label>Data split<select name="split"><option value="all">All data · '+Object.values(selected.splits).reduce((n,s)=>n+s.rows,0)+' rows</option>'+Object.entries(selected.splits).map(([k,v])=>'<option value="'+k+'">'+pretty(k)+' · '+v.rows+' rows</option>').join('')+'</select></label></div><p class="model-note">Plots use a reproducible sample of up to 2,000 rows. Missing axes or excluded predictions are omitted. A 1D plot uses small vertical jitter for visibility, not a second feature.</p><button class="primary">Generate classification graph</button></form>';
    bind('#classification-plot',async event=>{event.preventDefault();const f=new FormData(event.target),columns=[f.get('x')];if(f.get('y'))columns.push(f.get('y'));if(new Set(columns).size!==columns.length)throw Error('Choose two different columns.');await job(modelPath()+'/visualizations',{columns,split:f.get('split')});await refreshClassificationGallery();},'submit');
    await refreshClassificationGallery();
  }else $('#visualize-controls').innerHTML='<p>Select a trained tabular classification model above to graph its true and predicted classes. Summary export is available below for other tasks too.</p>';
  const comparison=await api(endpoint()+'/models/compare');
  if(ms.tab!=='visualize')return;
  $('#summary-group').innerHTML=comparison.groups.length?comparison.groups.map((g,i)=>'<option value="'+i+'">'+esc(g.reference.dataset)+' · '+esc(g.reference.run_id)+'</option>').join(''):'<option value="">Current dataset — EDA only</option>';
  const current=comparison.groups.findIndex(g=>JSON.stringify(g.reference)===JSON.stringify(selected?.record.input));
  if(current>=0)$('#summary-group').value=String(current);
  bind('#export-analysis',async()=>{const group=comparison.groups[Number($('#summary-group').value)];const result=await job(endpoint()+'/analysis-summary',{reference:group?.reference||null,dataset_id:dataset()?.id});$('#analysis-result').innerHTML='<div class="callout success"><a href="'+esc(result.url)+'" download="'+esc(result.name)+'">Download analysis summary</a></div>';});
}
async function refreshClassificationGallery(){
  const plots=await api(modelPath()+'/visualizations');
  if(ms.tab!=='visualize')return;
  $('#classification-gallery').innerHTML=plots.map(p=>'<section class="section"><div class="section-head"><div><h2>'+esc(p.title)+'</h2><p>'+esc(p.note)+' Plotted '+p.plotted_rows+' of '+p.source_rows+' rows; '+p.omitted_rows+' sampled rows omitted.</p></div></div><div class="section-body"><img class="learning-curve" src="'+esc(p.image)+'" alt="'+esc(p.title)+'"><label class="check"><input type="checkbox" data-summary-plot="'+esc(p.id)+'" '+(p.include_summary?'checked':'')+'>Add this graph to the HTML summary</label></div></section>').join('');
  $$('[data-summary-plot]').forEach(input=>input.onchange=guarded(async()=>{try{await job(endpoint()+'/visualizations/'+encodeURIComponent(input.dataset.summaryPlot)+'/summary',{include:input.checked});}catch(error){input.checked=!input.checked;throw error;}}));
}
