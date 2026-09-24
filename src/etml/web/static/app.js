'use strict';
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const brand = '<div class="brand"><span class="e">E</span><span class="t">T</span>M<span class="l">L</span><span class="word">Workbench</span></div>';
const state = {boot:null, project:null, tab:'preview', section:'data', recipe:null, preview:null, dirty:false, busy:false};
const token = $('meta[name="etml-token"]').content;
let noticeTimer;
async function api(path, body, form=false) {
  const options={headers:{'x-etml-token':token}};
  if(body!==undefined){options.method='POST';options.body=form?body:JSON.stringify(body);if(!form)options.headers['Content-Type']='application/json';}
  const response=await fetch('/api/'+path,options);
  let result;try{result=await response.json();}catch{throw Error('The server did not return a response. Check that workbench ui is running.');}
  if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));
  return result;
}
function notify(message,error=false,working=false) {
  clearTimeout(noticeTimer);const n=$('#notice');n.hidden=false;n.className=error?'error':'';
  n.innerHTML=(working?'<div class="spinner"></div>':'')+'<span>'+esc(message)+'</span>'+(!working?'<button aria-label="Dismiss message">×</button>':'');
  if(!working){$('button',n).onclick=()=>n.hidden=true;if(!error)noticeTimer=setTimeout(()=>n.hidden=true,9000);}
}
function guarded(fn){return async e=>{try{await fn(e);}catch(error){notify(error.message,true);}};}
function bind(id, fn, event='click'){const el=$(id);if(el)el.addEventListener(event,guarded(fn));}
const endpoint=()=> 'projects/'+encodeURIComponent(state.project.name);
const dataset=()=>state.project.dataset;
function payload(extra={}){return {dataset_id:dataset().id,...extra};}
async function job(path, data, form=false) {
  state.busy=true;setBusy(true);
  try{
    const queued=await api(path,data,form);
    return await waitJob(queued.job_id);
  }finally{state.busy=false;setBusy(false);}
}
async function waitJob(id){
  for(;;){
    const current=await api('jobs/'+id);
    if(state.section==='models')updateModelProgress(current);
    if(current.status==='done'){$('#notice').hidden=true;if(current.result?.warning)notify(current.result.warning,true);return current.result;}
    if(current.status==='failed')throw Error(current.error);
    notify(current.label+'…',false,true);await new Promise(r=>setTimeout(r,650));
  }
}
function setBusy(value){$$('button,input,select,textarea').forEach(el=>{if(value){el.dataset.wasDisabled=String(el.disabled);el.disabled=true;}else if(el.dataset.wasDisabled!==undefined){el.disabled=el.dataset.wasDisabled==='true';delete el.dataset.wasDisabled;}});}
async function refresh(){state.project=await api(endpoint()+(dataset()?'?dataset_id='+encodeURIComponent(dataset().id):''));}
async function openProject(name, id=null){
  state.project=await api('projects/'+encodeURIComponent(name)+(id?'?dataset_id='+encodeURIComponent(id):''));
  state.recipe=dataset()?.recipe||null;state.preview=null;state.dirty=false;state.tab='preview';state.section='data';resetModels();
  localStorage.setItem('etml-project',name);renderProject();
  if(state.project.busy){
    state.busy=true;setBusy(true);
    try{
      const active=await api('jobs/'+state.project.busy);
      if(active.label==='Training model'){
        ms.overview=await api(endpoint()+'/models');ms.started=Date.now()-(active.elapsed_seconds||0)*1000;
        state.section='models';renderProject();setBusy(true);
      }
      const result=await waitJob(state.project.busy);
      await refresh();state.recipe=dataset()?.recipe||null;
      if(state.section==='models'&&result.name){
        ms.overview=await api(endpoint()+'/models');
        ms.selected=await api(endpoint()+'/models/'+encodeURIComponent(result.name));ms.tab='results';
      }
    }finally{state.busy=false;ms.started=null;renderProject();}
  }
}
function home(){
  state.project=null;state.recipe=null;state.preview=null;
  $('#app').innerHTML='<div class="landing"><header>'+brand+'<span class="badge"><span class="dot"></span> Local workspace</span></header><main class="welcome"><div class="eyebrow">Your machine learning workspace</div><h1 style="margin-top:15px">A place for every experiment.</h1><p>Bring your data. Understand it. Build from there.</p><div class="project-cards"><section class="card"><div class="icon">+</div><h2>Start something new</h2><p>Keep your data, preparation steps and models together in one project.</p><form id="new-project"><label>Project name<input name="name" placeholder="e.g. customer-churn" required maxlength="80" autocomplete="off"></label><button class="primary">Create project <span aria-hidden="true">↗</span></button></form></section><section class="card"><div class="icon">▦</div><h2>Pick up where you left off</h2><p>Open a saved project and continue with your existing data and recipes.</p><form id="existing-project"><label>Existing project<select name="name">'+(state.boot.projects.length?state.boot.projects.map(n=>'<option>'+esc(n)+'</option>').join(''):'<option value="">No projects yet</option>')+'</select></label><button '+(!state.boot.projects.length?'disabled':'')+'>Open project <span aria-hidden="true">→</span></button></form></section></div><div class="footnote">Your files stay in your local project folder.</div></main></div>';
  const last=localStorage.getItem('etml-project');if(state.boot.projects.includes(last))$('#existing-project select').value=last;
  bind('#new-project',async e=>{e.preventDefault();const name=new FormData(e.target).get('name').trim();const created=await api('projects',{name});state.boot.projects.push(created.name);state.boot.projects.sort();await openProject(created.name);},'submit');
  bind('#existing-project',async e=>{e.preventDefault();await openProject(new FormData(e.target).get('name'));},'submit');
}
function tableHTML(data, affected=[]){
  if(!data?.rows?.length)return '<p class="table-caption">No rows remain in this preview.</p>';
  return '<div class="table-scroll"><table><thead><tr><th>Row</th>'+data.columns.map(c=>'<th class="'+(affected.includes(c)?'changed':'')+'">'+esc(c)+'</th>').join('')+'</tr></thead><tbody>'+data.rows.map((row,i)=>'<tr><td>'+esc(data.indexes?.[i]??i+1)+'</td>'+row.map((v,j)=>'<td class="'+(affected.includes(data.columns[j])?'changed':'')+'" title="'+esc(v??'Missing')+'">'+(v===null?'<span class="muted">—</span>':esc(typeof v==='object'?JSON.stringify(v):v))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
}
function renderProject(){
  const d=dataset(), name=state.project.name;
  $('#app').innerHTML='<div class="shell"><aside class="sidebar">'+brand+'<div class="project-name"><small>Current project</small><strong>'+esc(name)+'</strong><button id="switch-project" class="small quiet">← All projects</button></div><nav aria-label="Project navigation"><button id="nav-data" class="nav-item" aria-current="'+(state.section==='data'?'page':'false')+'"><span aria-hidden="true">▦</span> Data</button><button id="nav-models" class="nav-item" aria-current="'+(state.section==='models'?'page':'false')+'"><span aria-hidden="true">◈</span> Models</button></nav><div class="sidebar-bottom"><div class="side-note"><strong>Your next model starts here.</strong><p>Explore, prepare and split your data before training.</p></div><span class="badge"><span class="dot"></span> Saved on this computer</span></div></aside><main class="main"><header class="topline"><div><div class="eyebrow">Project / '+(state.section==='models'?'Models':'Data')+'</div><h1>'+(state.section==='models'?'From data to discovery.':'Your data, in focus.')+'</h1></div><div class="row"><button id="mobile-home" class="small quiet">Projects</button>'+(d&&state.section==='data'?'<button id="add-data" class="primary">+ Add data</button>':'')+'</div></header><div id="data-content"></div></main></div>';
  bind('#switch-project',async()=>{state.boot=await api('bootstrap');home();});bind('#mobile-home',async()=>{state.boot=await api('bootstrap');home();});
  bind('#add-data',showImport);
  bind('#nav-data',()=>{state.section='data';renderProject();});
  bind('#nav-models',async()=>{if(state.tab==='preprocess'&&state.section==='data')readRecipes();await showModels();});
  const content=$('#data-content');
  if(state.section==='models'){renderModels();return;}

  if(!d){content.innerHTML='<section class="empty"><div class="icon">↥</div><div class="eyebrow">A fresh start</div><h2 style="margin-top:12px">Give your project some data.</h2><p>Drop in a dataset, use a local file path, or explore a dataset from the library.</p><button id="first-data" class="primary">+ Add data</button><small style="margin-top:18px">CSV · TSV · Parquet · JSON · JSONL</small></section>';bind('#first-data',showImport);return;}
  content.innerHTML='<div class="row between" style="margin-bottom:16px"><span class="badge">'+Object.keys(d.columns).length+' columns · Raw data preserved</span><label>Dataset<select class="dataset-picker" id="dataset-picker">'+state.project.datasets.map(v=>'<option value="'+esc(v.id)+'" '+(v.id===d.id?'selected':'')+'>'+esc(v.name)+'</option>').join('')+'</select></label></div><nav class="tabs" aria-label="Data views"><button data-tab="preview" class="'+(state.tab==='preview'?'active':'')+'">Overview & preview</button><button data-tab="preprocess" class="'+(state.tab==='preprocess'?'active':'')+'">Preprocess & split</button></nav><div id="tab-content"></div>';
  bind('#dataset-picker',e=>openProject(name,e.target.value),'change');
  $$('[data-tab]').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;renderProject();});
  if(state.tab==='preview')renderPreview();else renderPreparation();
}
function reportURL(){return '/reports/'+encodeURIComponent(state.project.name)+'/'+encodeURIComponent(dataset().id)+'/'+dataset().report.split('/').map(encodeURIComponent).join('/');}
function renderPreview(){
  const d=dataset();
  $('#tab-content').innerHTML='<section class="section"><div class="section-head"><div><h2>Data overview</h2><p>Your automatically generated exploratory report</p></div>'+(d.report?'<a href="'+reportURL()+'" target="_blank" rel="noopener">Open report ↗</a>':'<button id="generate-report" class="small">Generate EDA</button>')+'</div>'+(d.report?'<div class="report-wrap"><iframe title="Embedded exploratory data report" src="'+reportURL()+'" sandbox="allow-scripts allow-same-origin" loading="lazy"></iframe><a class="report-link" href="'+reportURL()+'" target="_blank" rel="noopener" aria-label="Open full EDA report in a new tab"><span>Explore the full report ↗</span></a></div>':'<div class="section-body"><p>No report yet. Generate an overview to see distributions, relationships and missing values.</p></div>')+'</section><section class="section"><div class="section-head"><div><h2>Raw data</h2><p>First 10 rows'+(Object.keys(d.presplit).length?' of the uploaded training partition':'')+' · Original values, before preprocessing</p></div><span class="badge">Read only</span></div>'+tableHTML(d.preview)+'</section><div class="actions"><button id="start-prep" class="primary">Continue to preprocessing →</button></div>'+artifactsHTML();
  bind('#start-prep',()=>{state.tab='preprocess';renderProject();});
  bind('#generate-report',async()=>{await job(endpoint()+'/analyze',payload());await refresh();renderProject();});
}
function artifactsHTML(){
  return '<details class="section" style="margin-top:26px"><summary class="section-head" style="cursor:pointer">Saved project files <span class="muted">Raw · Processed · Splits</span></summary><div class="section-body">'+dataset().artifacts.map(a=>'<div class="artifact"><span>✓</span><div>'+esc(a.label)+'<code>'+esc(a.path)+'</code></div></div>').join('')+'</div></details>';
}
function stepHead(n,title,description){return '<div class="section-head"><div class="step-title"><span class="step-number">'+n+'</span><div><h2>'+title+'</h2><p>'+description+'</p></div></div></div>';}
function renderPreparation(){
  const d=dataset(),saved=d.state, task=saved.task;
  const selectedTarget=task?.target||d.provenance?.target||'';
  $('#tab-content').innerHTML='<div class="progress-steps"><span class="'+(task?'done':'current')+'">01 Choose a target</span><i>→</i><span class="'+(saved.processed?'done':task?'current':'')+'">02 Review & prepare</span><i>→</i><span class="'+(saved.split?'done':saved.processed?'current':'')+'">03 Split for training</span></div><section class="section">'+stepHead('1','What do you want to predict?','Choose the target and the kind of problem you are solving.')+'<form class="section-body stack" id="target-form"><div class="grid"><label>Target column<select name="target" required><option value="">Choose a column</option>'+Object.keys(d.columns).map(c=>'<option '+(c===selectedTarget?'selected':'')+'>'+esc(c)+'</option>').join('')+'</select></label><label>Task<select name="task_type"><option value="classification">Classification — predict a category</option><option value="regression">Regression — predict a number</option></select></label></div><details><summary class="muted">More options</summary><div class="grid" style="margin-top:14px"><label>Exclude feature columns<select name="excluded" multiple size="4">'+Object.keys(d.columns).map(c=>'<option '+(task?.excluded_columns?.includes(c)?'selected':'')+'>'+esc(c)+'</option>').join('')+'</select></label><label>Missing target values<select name="missing_target"><option value="error">Stop and ask me to fix them</option><option value="drop">Drop rows with missing targets</option></select></label></div></details><div class="row between"><small>Recommendations will leave your target unchanged.</small><button class="primary">'+(task?'Update target & recommendations':'Get recommendations →')+'</button></div>'+(saved.processed?'<small>Updating the target starts a new preparation. Earlier files stay in the project.</small>':'')+'</form></section><div id="recipe-section"></div><div id="change-preview"></div><div id="split-section"></div>'+artifactsHTML();
  $('#target-form [name=task_type]').value=task?.task_type||d.provenance?.task_type||'classification';
  $('#target-form [name=missing_target]').value=task?.missing_target||'error';
  bind('#target-form',async e=>{e.preventDefault();const f=new FormData(e.target);const result=await job(endpoint()+'/suggest',payload({target:f.get('target'),task_type:f.get('task_type'),excluded:f.getAll('excluded'),missing_target:f.get('missing_target')}));state.recipe=result.recipe;state.preview=null;state.dirty=false;await refresh();renderPreparation();$('#recipe-section').scrollIntoView({behavior:'smooth',block:'start'});},'submit');
  if(task)renderRecipes();
  if(state.preview)renderChanges();
  if(saved.processed&&!state.dirty)renderSplit();
}
function readRecipes(){
  if(!state.recipe)return;
  $$('.recipe').forEach((card,i)=>{
    const step=state.recipe.steps[i];
    step.enabled=$('[name=enabled]',card).checked;
    step.columns=[...$('[name=columns]',card).selectedOptions].map(o=>o.value);
    step.params=JSON.parse($('[name=params]',card).value);
    if(!step.params||Array.isArray(step.params)||typeof step.params!=='object')throw Error('Recipe parameters must be a JSON object.');
    step.status='proposed';step.approval_fingerprint=null;
  });
}
function invalidatePreview(){
  state.preview=null;state.dirty=true;$('#change-preview').innerHTML='';
  $('#split-section').innerHTML='<p class="callout">Preview and save your edited recipe before creating new splits.</p>';
}
function renderRecipes(){
  if(!state.recipe)state.recipe={format_version:1,name:'Project preparation',steps:[]};
  const columns=Object.keys(dataset().columns).filter(c=>c!==dataset().state.task.target&&!dataset().state.task.excluded_columns.includes(c));
  $('#recipe-section').innerHTML='<section class="section">'+stepHead('2','Review your recipe','Keep, adjust or disable recommendations. Add your own steps when needed.')+'<div class="section-body"><div id="recipe-list">'+(state.recipe.steps.length?state.recipe.steps.map((s,i)=>'<article class="recipe"><div class="row between"><label class="check"><input name="enabled" type="checkbox" '+(s.enabled?'checked':'')+'><strong>'+esc(s.operation.replaceAll('_',' '))+'</strong></label><button class="small quiet remove-step" data-index="'+i+'" aria-label="Remove step '+(i+1)+'">Remove</button></div><p>'+esc(s.reason||'Custom preprocessing step')+'</p><div class="recipe-fields"><label>Columns<select name="columns" multiple size="'+Math.min(3,Math.max(1,columns.length))+'">'+columns.map(c=>'<option '+(s.columns.includes(c)?'selected':'')+'>'+esc(c)+'</option>').join('')+'</select></label><label>Parameters (JSON)<textarea name="params" rows="3" spellcheck="false">'+esc(JSON.stringify(s.params,null,2))+'</textarea></label></div>'+(s.operation==='fill_missing'?'<div class="row" style="margin-top:12px"><small>Quick strategy:</small>'+['mean','median','mode'].map(v=>'<button class="small strategy" data-index="'+i+'" data-value="'+v+'">'+v+'</button>').join('')+'</div>':'')+'<details><summary>Why this step?</summary><small>'+esc(JSON.stringify(s.evidence||{}))+'</small></details></article>').join(''):'<div class="callout">No automatic changes are needed. You can add a custom step or preview the unchanged features.</div>')+'</div><div class="row"><select id="custom-operation" aria-label="Custom recipe operation" style="max-width:240px">'+Object.keys(state.boot.transforms).map(v=>'<option value="'+v+'">'+v.replaceAll('_',' ')+'</option>').join('')+'</select><button id="add-step" class="quiet">+ Add custom step</button></div><p class="table-caption" style="padding-left:0">Previewing approves the enabled steps shown above. Only feature columns are transformed.</p><div class="actions"><button id="preview-recipe" class="primary">Approve & preview changes →</button></div></div></section>';
  $$('.recipe input,.recipe select,.recipe textarea').forEach(el=>el.addEventListener('input',invalidatePreview));
  $$('.remove-step').forEach(el=>el.onclick=guarded(()=>{readRecipes();state.recipe.steps.splice(Number(el.dataset.index),1);invalidatePreview();renderRecipes();}));
  $$('.strategy').forEach(el=>el.onclick=guarded(()=>{readRecipes();state.recipe.steps[Number(el.dataset.index)].params={strategy:el.dataset.value};invalidatePreview();renderRecipes();}));
  bind('#add-step',()=>{readRecipes();const op=$('#custom-operation').value;const defaults={fill_missing:{strategy:'median'},drop_missing:{},convert_type:{dtype:'Float64'},normalize_categories:{},map_categories:{mapping:[]},select_columns:{},drop_columns:{},rename_columns:{names:[]}};state.recipe.steps.push({operation:op,columns:[],params:defaults[op],reason:'Custom step',evidence:{},enabled:true,status:'proposed'});invalidatePreview();renderRecipes();});
  bind('#preview-recipe',async()=>{readRecipes();const result=await job(endpoint()+'/preview',payload({recipe:state.recipe}));state.preview=result;state.recipe=result.recipe;state.dirty=false;await refresh();renderPreparation();$('#change-preview').scrollIntoView({behavior:'smooth'});});
}
function renderChanges(){
  const p=state.preview, info=p.preview_info||{};
  const description=info.changed
    ? 'Showing '+p.before.rows.length+' changed source '+(p.before.rows.length===1?'row':'rows')+' · Original row numbers are preserved.'
    : 'No changed rows found. Showing the first source rows for reference.';
  const notes=[
    info.removed_rows?.length?'Rows removed: '+info.removed_rows.join(', '):'',
    info.removed_columns?.length?'Columns removed or renamed: '+info.removed_columns.join(', '):'',
    info.added_columns?.length?'Columns added or renamed: '+info.added_columns.join(', '):'',
    ...Object.entries(info.type_changes||{}).map(([column,types])=>column+': '+types.before+' → '+types.after)
  ].filter(Boolean).map(message=>'<p class="model-note">'+esc(message)+'</p>').join('');

  $('#change-preview').innerHTML='<section class="section">'+stepHead('↔','See what changes',description)+'<div class="section-body"><div class="before-after"><div><h3>Before</h3>'+tableHTML(p.before,p.affected_columns)+'</div><div><h3>After</h3>'+tableHTML(p.after,[...p.affected_columns,...(info.added_columns||[])])+'</div></div>'+notes+'<div class="callout">This preview is for inspection. When you split, the recipe is fitted again using only training rows, so validation and test data stay unseen.</div><div class="actions"><button id="save-processed" class="primary">Save processed data & continue →</button></div></div></section>';
  bind('#save-processed',async()=>{await job(endpoint()+'/save',payload({token:state.preview.token}));state.preview=null;await refresh();renderPreparation();$('#split-section').scrollIntoView({behavior:'smooth'});notify('Processed data saved in your project.');});
}
function renderSplit(){
  const d=dataset(), saved=d.state.split, presplit=Object.keys(d.presplit).length>0;
  $('#split-section').innerHTML='<section class="section">'+stepHead('3','Set aside data for training','Your raw data is preserved. Each split is saved as a new project artifact.')+'<form id="split-form" class="section-body stack"><div class="callout success">✓ Processed version saved · '+esc(d.state.processed.rows)+' rows</div>'+(presplit?'<div class="callout">Using your existing uploaded train/test partitions. The uploaded test set will be preserved.</div>'+(!d.presplit.validation?'<label>Validation fraction to hold out of training (0–0.99)<input name="validation_fraction" type="number" min="0" max=".99" step=".01" value=".2"></label>':'<p>Using the supplied validation partition.</p>'):'<label>Train / validation / test<select id="split-preset"><option value="70,15,15">70 / 15 / 15 — balanced default</option><option value="80,10,10">80 / 10 / 10</option><option value="60,20,20">60 / 20 / 20</option><option value="80,0,20">80 / 0 / 20</option><option value="custom">Custom split</option></select></label><div class="grid three" id="ratios" hidden><label>Train %<input name="train" type="number" min="1" max="100" value="70" required></label><label>Validation %<input name="validation" type="number" min="0" max="99" value="15" required></label><label>Test %<input name="test" type="number" min="0" max="99" value="15" required></label></div><div><div class="split-bar"><span style="width:70%"></span><span style="width:15%"></span><span style="width:15%"></span></div><div class="split-legend"><span>Training</span><span>Validation</span><span>Test</span></div></div>')+'<div class="grid"><label>Split method<select name="strategy"><option value="random">Random</option>'+(d.state.task.task_type==='classification'?'<option value="stratified">Stratified — keep class proportions</option>':'')+'<option value="group">By group — keep related rows together</option><option value="chronological">Chronological — earlier rows first</option></select></label><label>Random seed<input name="seed" type="number" value="42" min="0" required></label></div><label id="split-column-label" hidden>Group / time column<select name="split_column">'+Object.keys(d.columns).filter(c=>c!==d.state.task.target).map(c=>'<option>'+esc(c)+'</option>').join('')+'</select><small>Chronological splitting expects ISO 8601 dates.</small></label><div class="actions"><button class="primary">'+(saved?'Create another split':'Create training splits →')+'</button></div></form>'+(saved?'<div class="section-body" style="border-top:1px solid var(--line)"><h3>✓ Ready for training</h3><div class="grid three" style="margin-top:15px">'+Object.entries(saved.prepared_counts).map(([k,v])=>'<div class="card" style="padding:18px"><small>'+esc(k)+'</small><h2>'+esc(v)+' rows</h2></div>').join('')+'</div><p style="margin-top:20px">Splits and the training-fitted recipe are saved. Continue to Models to choose and train a model.</p><button id="continue-models" class="primary" style="margin-top:18px">Continue to models →</button><code class="shortcut">workbench '+esc(state.project.name)+' --models --projects-dir &quot;'+esc(state.project.projects_dir)+'&quot; --dataset-id '+esc(d.id)+'</code></div>':'')+'</section>';
  bind('#continue-models',showModels);
  if(saved){
    const config=saved.config;
    if(!presplit){
      const percentages=['train','validation','test'].map(k=>Math.round(config[k]*10000)/100);
      const preset=percentages.join(',');
      $('#split-preset').value=['70,15,15','80,10,10','60,20,20','80,0,20'].includes(preset)?preset:'custom';
      $('#ratios').hidden=$('#split-preset').value!=='custom';
      $$('#ratios input').forEach((el,i)=>el.value=percentages[i]);
      updateBar();
    }
    $('#split-form [name=strategy]').value=config.strategy;
    $('#split-form [name=seed]').value=config.seed;
    $('#split-column-label').hidden=!['group','chronological'].includes(config.strategy);
    if(config.group_column||config.time_column)$('#split-form [name=split_column]').value=config.group_column||config.time_column;
    const validation=$('#split-form [name=validation_fraction]');
    if(validation)validation.value=saved.validation_fraction;
  }
  bind('#split-preset',e=>{const custom=e.target.value==='custom';$('#ratios').hidden=!custom;if(!custom){e.target.value.split(',').forEach((v,i)=>$('#ratios input[name='+['train','validation','test'][i]+']').value=v);}updateBar();},'change');
  $$('#ratios input').forEach(el=>el.addEventListener('input',updateBar));
  bind('#split-form [name=strategy]',e=>{$('#split-column-label').hidden=!['group','chronological'].includes(e.target.value);},'change');
  bind('#split-form',async e=>{e.preventDefault();const f=new FormData(e.target);const config={strategy:f.get('strategy'),seed:Number(f.get('seed')),train:presplit ? .7 : Number(f.get('train'))/100,validation:presplit ? .15 : Number(f.get('validation'))/100,test:presplit ? .15 : Number(f.get('test'))/100};if(config.strategy==='group')config.group_column=f.get('split_column');if(config.strategy==='chronological'){config.time_column=f.get('split_column');config.time_format='ISO8601';}await job(endpoint()+'/split',payload({config,validation_fraction:Number(f.get('validation_fraction')??.2)}));await refresh();renderPreparation();$('#split-section').scrollIntoView({behavior:'smooth'});notify('Training splits saved. Your data is ready.');},'submit');
}
function updateBar(){const values=$$('#ratios input').map(e=>Number(e.value));$$('.split-bar span').forEach((el,i)=>el.style.width=Math.max(0,Math.min(100,values[i]))+'%');}
function showImport(){
  const dialog=$('#import-dialog');
  dialog.innerHTML='<div class="section-head"><div><div class="eyebrow">Bring your data</div><h2 id="import-title">Add a dataset</h2></div><button id="close-import" class="quiet small" aria-label="Close import">×</button></div><nav class="tabs" aria-label="Import source"><button class="active" data-source-tab="upload">Upload</button><button data-source-tab="path">File path</button><button data-source-tab="library">Dataset library</button></nav><form id="import-form" class="section-body"><div id="import-fields"></div><label class="check" style="margin-top:22px"><input type="checkbox" name="analyze" checked>Automatically generate the EDA</label><div class="actions"><button class="primary">Add to project →</button></div></form>';
  let sourceTab='upload', dropped=null;
  const draw=()=>{
    const field=$('#import-fields');dropped=null;
    if(sourceTab==='upload'){
      field.innerHTML='<div class="dropzone" id="dropzone"><div style="font-size:32px;color:var(--purple)">↥</div><h3>Drop your dataset here</h3><p>or choose a file from your computer</p><input name="file" id="upload-file" type="file" accept=".csv,.tsv,.parquet,.json,.jsonl,.ndjson" aria-label="Choose dataset file"><p id="file-name"></p></div><p class="upload-limit" style="margin-top:13px">CSV, TSV, Parquet, JSON or JSONL · Up to '+Math.round(state.boot.max_upload_bytes/1024**2)+' MB</p>';
      const zone=$('#dropzone');zone.ondragover=e=>{e.preventDefault();zone.classList.add('over');};zone.ondragleave=()=>zone.classList.remove('over');zone.ondrop=e=>{e.preventDefault();zone.classList.remove('over');dropped=e.dataTransfer.files[0];$('#file-name').textContent=dropped?.name||'';};
      $('#upload-file').onchange=()=>{dropped=null;$('#file-name').textContent=$('#upload-file').files[0]?.name||'';};
    }else if(sourceTab==='path'){
      field.innerHTML='<label>Local file path<input name="path" placeholder="C:\\data\\customers.csv" required></label><p class="upload-limit" style="margin-top:12px">Choose a file on the computer running the workbench. It will be copied into the project.</p>';
    }else{
      field.innerHTML='<div class="import-fields"><label>Provider<select id="provider" name="source"><option value="sklearn">Scikit-learn — built-in datasets</option><option value="huggingface">Hugging Face</option><option value="openml">OpenML</option><option value="url">Direct download URL</option></select></label><div id="provider-fields"></div></div>';
      const providerFields=()=>{
        const provider=$('#provider').value;
        $('#provider-fields').innerHTML=provider==='sklearn'?'<label>Dataset<select name="name">'+Object.entries(state.boot.library).map(([n,t])=>'<option value="'+n+'">'+n.replaceAll('_',' ')+' — '+t+'</option>').join('')+'</select></label>':provider==='huggingface'?'<div class="stack"><label>Dataset repository<input name="repository" placeholder="organization/dataset" required></label><div class="grid"><label>Configuration (optional)<input name="config"></label><label>Split<input name="split" value="train" required></label></div><label>Revision (optional)<input name="revision" placeholder="Commit hash, branch or tag"></label><small>Requires the huggingface extra. Downloads up to 200,000 rows. Choose a tabular dataset for this preparation flow.</small></div>':provider==='openml'?'<label>OpenML data ID<input name="data_id" type="number" min="1" value="61" required><small>For example: 61 is Iris. Downloads up to 200,000 rows.</small></label>':'<label>Direct data-file URL<input name="url" type="url" placeholder="https://example.com/data.csv" required></label>';
      };$('#provider').onchange=providerFields;providerFields();
    }
  };
  $$('[data-source-tab]').forEach(b=>b.onclick=()=>{sourceTab=b.dataset.sourceTab;$$('[data-source-tab]').forEach(v=>v.classList.toggle('active',v===b));draw();});
  $('#close-import').onclick=()=>dialog.close();draw();dialog.showModal();
  bind('#import-form',async e=>{
    e.preventDefault();const form=new FormData(e.target), analyze=form.get('analyze')==='on';
    let result;
    if(sourceTab==='upload'){
      const file=dropped||$('#upload-file').files[0];if(!file)throw Error('Choose a dataset file first.');if(file.size>state.boot.max_upload_bytes)throw Error('This file exceeds the upload limit.');
      const upload=new FormData();upload.set('file',file);upload.set('analyze',String(analyze));dialog.close();result=await job(endpoint()+'/upload',upload,true);
    }else{
      const body=Object.fromEntries(form);body.source=sourceTab==='path'?'path':body.source;body.analyze=analyze;dialog.close();result=await job(endpoint()+'/import',body);
    }
    await openProject(state.project.name,result.dataset_id);
  },'submit');
}
(async()=>{
  try{[state.boot]=await Promise.all([api('bootstrap'),new Promise(r=>setTimeout(r,650))]);home();$('#app').hidden=false;$('#splash').hidden=true;}
  catch(error){$('#splash p').textContent='Unable to connect. Restart workbench ui and reload this page.';$('.spinner', $('#splash'))?.remove();notify(error.message,true);}
})();
