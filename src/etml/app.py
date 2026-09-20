"""Optional local Streamlit interface; all operations delegate to the public backend."""
from pathlib import Path
import json
import sys
import streamlit as st
import pandas as pd
from data import DatasetWorkspace
from data.manifest import resolve_inside
from data.sources import SKLEARN_DATASETS,import_sklearn,import_huggingface,import_url,import_openml
from models import ModelConfig,available_models
from training import train_models
from training.experiments import latest_preparation,training_runs
from prediction import Predictor
from prediction.example import predict_example


def show_result(result,dataset_id,task_id):
    st.success('Training complete. Candidates were compared on validation data.')
    rows=[]
    for row in result.leaderboard:
        rows.append({'Model':row['model']['backend']+':'+row['model']['algorithm'],
                     'Candidate':row['candidate'],'Baseline':row['is_baseline'],
                     'Fit seconds':round(row['fit_seconds'],3),**row['validation'],
                     'Improvement over baseline':row['improvement_over_baseline']})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.caption('F1 macro weights each class equally. Fit time includes epoch checkpoint writing for neural models.')
    from etml.display import command
    st.code(command('workbench','models','compare',dataset_id,task_id))
    st.write('Winning bundle:',str(result.bundle))


def select_dataset(ws):
    datasets=ws.list_datasets()
    if not datasets:
        st.info('Import a dataset first.'); return None
    choice=st.selectbox('Dataset',[d.dataset_id for d in datasets],key='chosen_dataset')
    return ws.get(choice)


def select_task(dataset):
    paths=sorted((dataset.directory/'tasks').glob('*/task.json'))
    if not paths:
        st.info('Create a task on the Prepare page first.'); return None
    return st.selectbox('Task',[p.parent.name for p in paths],key='chosen_task')


def import_page(ws):
    st.header('Bring in your data')
    kind=st.radio('Source',['Upload files','scikit-learn','Hugging Face','Direct URL','OpenML'],horizontal=True)
    identifier=st.text_input('Dataset ID',placeholder='customer-churn')
    if kind=='Upload files':
        presplit=st.checkbox('My data already has separate training and test files',value=True)
        train=st.file_uploader('Training file' if presplit else 'Dataset file',type=['csv','tsv','parquet','json','jsonl'],key='train_upload')
        test=validation=None
        if presplit:
            left,right=st.columns(2)
            with left: test=st.file_uploader('Test file (optional)',type=['csv','tsv','parquet','json','jsonl'],key='test_upload')
            with right: validation=st.file_uploader('Validation file (optional)',type=['csv','tsv','parquet','json','jsonl'],key='validation_upload')
            st.caption('Test rows stay separate. If validation is absent, it will be taken only from training. Test files may omit the target.')
        if st.button('Import files',disabled=train is None):
            if presplit:
                uploads={role:(file.name,file) for role,file in {'train':train,'test':test,'validation':validation}.items() if file is not None}
                for _,file in uploads.values(): file.seek(0)
                data=ws.import_split_uploads(uploads,dataset_id=identifier or None,max_bytes=512*1024**2)
            else:
                train.seek(0)
                data=ws.import_upload(train,filename=train.name,dataset_id=identifier or None,max_bytes=512*1024**2)
            st.success('Imported '+data.dataset_id+'. Continue to Prepare.')
    elif kind=='scikit-learn':
        name=st.selectbox('Built-in dataset',list(SKLEARN_DATASETS))
        if st.button('Import dataset'):
            data=import_sklearn(ws,name,dataset_id=identifier or None)
            st.success('Imported '+data.dataset_id+'. Target: target. Continue to Prepare.')
    elif kind=='Hugging Face':
        repo=st.text_input('Dataset repository',placeholder='owner/dataset')
        config=st.text_input('Configuration (optional)')
        revision=st.text_input('Revision (optional; resolved commit will be recorded)')
        train=st.text_input('Training split',value='train')
        validation=st.text_input('Validation split (optional)')
        test=st.text_input('Test split (optional)')
        if st.button('Import dataset',disabled=not repo):
            roles={k:v for k,v in {'train':train,'validation':validation,'test':test}.items() if v}
            data=import_huggingface(ws,repo,config=config or None,revision=revision or None,
                splits=roles,dataset_id=identifier or None)
            st.success('Imported '+data.dataset_id+'. Continue to Prepare.')
    elif kind=='Direct URL':
        url=st.text_input('Direct data-file URL')
        if st.button('Import dataset',disabled=not url):
            data=import_url(ws,url,dataset_id=identifier or None)
            st.success('Imported '+data.dataset_id+'. Continue to Prepare.')
    else:
        identifier_openml=st.number_input('OpenML data ID',min_value=1,value=61)
        if st.button('Import dataset'):
            data=import_openml(ws,int(identifier_openml),dataset_id=identifier or None)
            st.success('Imported '+data.dataset_id+'. Continue to Prepare.')


def prepare_page(ws):
    from tasks import TaskConfig
    from workflows import prepare_task
    from preprocessing import Recipe
    from eda_tool.loader import open_dataset
    dataset=select_dataset(ws)
    if dataset is None: return
    source=resolve_inside(dataset.directory,dataset.manifest.split_files['train']) if dataset.manifest.split_files else dataset.raw_files
    columns=list(open_dataset(source).schema())
    target=st.selectbox('Target column',columns,index=columns.index('target') if 'target' in columns else 0)
    task_type=st.selectbox('Task type',['classification','regression'])
    name=st.text_input('Task ID',value='prediction')
    excluded=st.multiselect('Exclude columns (IDs, leakage, etc.)',[c for c in columns if c!=target])
    fraction=st.slider('Validation fraction of supplied training file',.05,.5,.2,.05)
    recipe_path=st.text_input('Approved recipe JSON path (optional)')
    st.caption('Preparation fits the approved recipe only on training rows. With no recipe, training still fits median imputation and categorical encoding on training data.')
    if st.button('Prepare task'):
        result=prepare_task(ws,dataset.dataset_id,TaskConfig(name,target,task_type,tuple(excluded)),
            validation_fraction=fraction,recipe=Recipe.load(recipe_path) if recipe_path else Recipe())
        st.success('Prepared '+result.run_id+'. Continue to Train.')
        st.json(result.prepared_counts)


def train_page(ws):
    dataset=select_dataset(ws)
    if dataset is None: return
    task=select_task(dataset)
    if task is None: return
    choices=st.multiselect('Models',list(available_models()),default=['sklearn:linear','sklearn:random_forest'])
    epochs=st.number_input('Neural epochs',min_value=1,value=30)
    st.caption('The latest preparation is used. A dummy baseline is included automatically.')
    if st.button('Train and compare',disabled=not choices):
        configs=[]
        for value in choices:
            backend,algorithm=value.split(':')
            params={'epochs':int(epochs)} if backend in {'pytorch','tensorflow'} else {'early_stopping_rounds':10} if backend=='xgboost' else {}
            configs.append(ModelConfig(backend,algorithm,params))
        with st.spinner('Training models...'):
            result=train_models(ws,dataset.dataset_id,task,latest_preparation(ws,dataset.dataset_id,task),configs=configs)
        st.session_state['training_result']=result
        st.session_state['training_result_scope']=(dataset.dataset_id,task)
    if st.session_state.get('training_result_scope')==(dataset.dataset_id,task):
        show_result(st.session_state['training_result'],dataset.dataset_id,task)


def choose_bundle(ws):
    direct=st.text_input('Model bundle path (optional; otherwise select a saved run)')
    if direct: return Path(direct)
    dataset=select_dataset(ws)
    if dataset is None: return None
    task=select_task(dataset)
    if task is None: return None
    runs=training_runs(ws,dataset.dataset_id,task)
    if not runs:
        st.info('Train a model first.'); return None
    run_id=st.selectbox('Training run',[r['run_id'] for r in runs])
    run=next(r for r in runs if r['run_id']==run_id)
    candidates=[r['candidate'] for r in run['leaderboard']]
    candidate=st.selectbox('Candidate',candidates,index=candidates.index(run['winner']))
    return Path(run['directory'])/'candidates'/candidate


def predict_page(ws):
    bundle=choose_bundle(ws)
    if bundle is None: return
    predictor=Predictor.load(bundle)
    st.caption('Enter one raw example. Leave a field blank for a missing value.')
    with st.form('one_example'):
        values={}
        rawtypes=predictor.schema.get('raw_dtypes',{})
        for column in predictor.schema['raw_columns']:
            value=st.text_input(column,key='feature_'+str(bundle)+'_'+column)
            if not value: values[column]=None
            elif any(kind in rawtypes.get(column,'').lower() for kind in ('int','float','double')):
                values[column]=value  # Convert after submit so intermediate typing never causes errors.
            else: values[column]=value
        submitted=st.form_submit_button('Predict example')
    if submitted:
        for column,value in values.items():
            dtype=rawtypes.get(column,'').lower()
            if value is not None:
                if 'int' in dtype: values[column]=int(value)
                elif 'float' in dtype or 'double' in dtype: values[column]=float(value)
                elif 'bool' in dtype:
                    if value.lower() not in {'true','false'}: raise ValueError(column+': enter true or false')
                    values[column]=value.lower()=='true'
        result=predict_example(predictor,values)
        if result['status']=='predicted':
            st.metric('Prediction',str(result['prediction']))
            if result.get('probabilities'):
                st.dataframe(pd.DataFrame(result['probabilities']),hide_index=True)
        else: st.info('This example was excluded by the saved preprocessing recipe.')
        with st.expander('Inspect input and prepared features',expanded=True): st.json(result)


def reproduce_page(ws):
    bundle=choose_bundle(ws)
    if bundle is None: return
    setup=bundle/'SETUP.md'
    if setup.exists():
        st.markdown(setup.read_text(encoding='utf-8'))
        st.download_button('Download package pins',(bundle/'requirements.txt').read_text(),file_name='requirements.txt')
    destination=st.text_input('Replay package output folder',value='exports/replay-package')
    st.caption('This explicit export includes training/validation rows, model artifacts and source code. Test rows are not included.')
    if st.button('Export reproducibility package'):
        from training.reproducibility import export_reproduction
        path=export_reproduction(bundle,destination,workspace=ws)
        st.success('Saved '+str(path)+'. Follow SETUP.md inside that folder.')


def main():
    st.set_page_config(page_title='ETML Workbench',page_icon='◈',layout='wide')
    st.markdown('''<style>
      [data-testid="stMainBlockContainer"] {padding: 2rem 1.25rem;}
      h1 {font-size: clamp(1.6rem, 3vw, 2.5rem); overflow-wrap: anywhere;}
      h2 {font-size: clamp(1.25rem, 2.5vw, 2rem);}
      [role="radiogroup"] {flex-wrap: wrap;}
    </style>''',unsafe_allow_html=True)
    st.title('ETML Workbench')
    st.caption('Data → preparation → models → individual predictions')
    default='datasets'
    if '--workspace' in sys.argv: default=sys.argv[sys.argv.index('--workspace')+1]
    workspace=st.sidebar.text_input('Workspace folder',value=default,key='workspace')
    page=st.sidebar.radio('Workbench',['Import','Prepare','Train','Predict one','Reproduce'],key='page')
    try:
        {'Import':import_page,'Prepare':prepare_page,'Train':train_page,'Predict one':predict_page,
         'Reproduce':reproduce_page}[page](DatasetWorkspace(workspace))
    except (ValueError,TypeError,OSError,ImportError,RuntimeError,KeyError) as exc:
        st.error(str(exc))


if __name__=='__main__': main()
