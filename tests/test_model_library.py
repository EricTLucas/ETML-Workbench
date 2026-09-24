import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from models import ModelConfig,create_model,available_models
from models.catalog import CATALOG,describe
from data.projects import ProjectStore
from data.model_inputs import prepare_model_input,load_model_input
from training.specialized import train_specialized,predict_specialized,export_specialized
from training.exploration import train_exploration,predict_exploration
from workflows.project_models import ProjectModels


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.project=ProjectStore(self.root/'projects').create('demo')

    def test_supervised_sklearn_roundtrips(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(42); x=csr_matrix(rng.uniform(.1,3,(72,4)))
        for task in ('classification','regression'):
            y=np.arange(72)%3 if task=='classification' else np.arange(72)/72
            for key,entry in available_models().items():
                if not key.startswith('sklearn:') or task not in entry['tasks']: continue
                with self.subTest(key=key,task=task):
                    config=ModelConfig(*key.split(':'))
                    model=create_model(config,task).fit_validation(x,y,validation_data=(x,y))
                    path=self.root/(task+'-'+key.split(':')[1]); path.mkdir()
                    model.save(path); restored=create_model(config,task).load(path)
                    np.testing.assert_allclose(model.predict(x),restored.predict(x))
                    if task=='classification':
                        p=restored.predict_proba(x); self.assertEqual(p.shape,(72,3)); np.testing.assert_allclose(p.sum(axis=1),1)

    def test_boosting_roundtrips(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(60,3))
        for backend in ('lightgbm','catboost'):
            if not importlib.util.find_spec(backend): continue
            for task in ('classification','regression'):
                with self.subTest(backend=backend,task=task):
                    y=np.arange(60)%3 if task=='classification' else x[:,0]*2
                    config=ModelConfig(backend,'boosted_trees',params={'iterations' if backend=='catboost' else 'n_estimators':5})
                    model=create_model(config,task).fit_validation(x,y,validation_data=(x,y))
                    path=self.root/(backend+task); path.mkdir(); model.save(path)
                    restored=create_model(config,task).load(path)
                    np.testing.assert_allclose(model.predict(x),restored.predict(x))
                    if task=='classification': np.testing.assert_allclose(model.predict_proba(x),restored.predict_proba(x))

    def test_catalog_and_names(self):
        self.assertTrue(set(available_models())<=set(CATALOG))
        self.assertGreaterEqual(len(CATALOG),45)
        self.assertIn('GPT-2',describe('transformers:gpt'))
        models=ProjectModels(self.project)
        self.assertEqual(models.create('sklearn:ridge')['name'],'model1')
        self.assertEqual(models.create('sklearn:ridge')['name'],'model2')
        with self.assertRaises(FileExistsError): models.create('sklearn:ridge',name='model1')
        with self.assertRaises(ValueError): models.create('sklearn:ridge',name='../bad')
        with self.assertRaises(ValueError): models.bundle('model1')

    def test_neural_settings_and_optimizers(self):
        x=np.random.default_rng(9).normal(size=(24,3)).astype(np.float32); y=x[:,0]
        for backend,module in [('pytorch','torch'),('tensorflow','tensorflow')]:
            if not importlib.util.find_spec(module): continue
            for optimizer in ('adam','adamw','sgd','rmsprop'):
                with self.subTest(backend=backend,optimizer=optimizer):
                    config=ModelConfig(backend,'mlp',params={'hidden_sizes':[4],'epochs':2,'batch_size':8,
                        'optimizer':optimizer,'regularizer':'l2','regularization_strength':.001,'loss':'mae'})
                    adapter=create_model(config,'regression').fit_validation(x,y,validation_data=(x,y))
                    self.assertTrue(np.isfinite(adapter.predict(x)).all())
                    self.assertAlmostEqual(adapter._metrics(x,y)['loss'],np.abs(adapter.predict(x)-y).mean(),places=5)
                    folder=self.root/(backend+optimizer); folder.mkdir(); adapter.save(folder)
                    np.testing.assert_allclose(adapter.predict(x),create_model(config,'regression').load(folder).predict(x),rtol=1e-5)
        with self.assertRaises(ValueError): create_model(ModelConfig('pytorch','mlp',params={'loss':'mse'}),'classification')

    def test_exploration_roundtrips_and_transductive_contract(self):
        frame=pd.DataFrame(np.random.default_rng(8).normal(size=(40,3)),columns=['a','b','c'])
        source=self.root/'explore.csv'; frame.to_csv(source,index=False)
        for algorithm in ('kmeans','hierarchical','gaussian_mixture','dbscan','pca','tsne','isolation_forest'):
            with self.subTest(algorithm=algorithm):
                path=self.root/algorithm
                train_exploration(source,'sklearn:'+algorithm,path,params={'perplexity':5,'max_iter':250} if algorithm=='tsne' else {})
                rows=pd.read_csv(path/'assignments.csv'); self.assertEqual(len(rows),len(frame))
                if algorithm in {'hierarchical','dbscan','tsne'}:
                    with self.assertRaisesRegex(ValueError,'out-of-sample'): predict_exploration(path,frame.iloc[0].to_dict())
                else:
                    result=predict_exploration(path,frame.iloc[0].to_dict()); self.assertTrue(result)

    def series_input(self):
        frame=pd.DataFrame({'time':pd.date_range('2020-01-01',periods=60),'value':np.sin(np.arange(60)/4)+np.arange(60)*.02})
        source=self.root/'series.csv'; frame.to_csv(source,index=False)
        return prepare_model_input(self.project,{'modality':'time_series','train':str(source),'columns':{'time':'time','value':'value'},'task':'forecasting'})

    def test_chronological_splits_and_irregular_rejection(self):
        path=self.series_input(); root,info=load_model_input(path)
        train=pd.read_parquet(root/'splits/train.parquet'); val=pd.read_parquet(root/'splits/validation.parquet')
        self.assertLess(train.time.max(),val.time.min()); self.assertEqual(sum(info['split_counts'].values()),60)
        frame=pd.DataFrame({'t':['2020-01-01','2020-01-03','2020-01-04','2020-01-05'],'v':[1,2,3,4]})
        source=self.root/'bad.csv'; frame.to_csv(source,index=False)
        with self.assertRaisesRegex(ValueError,'regular'):
            prepare_model_input(self.project,{'modality':'time_series','train':str(source),'columns':{'time':'t','value':'v'}})

    def test_specialized_only_project_reopens_without_tabular_import(self):
        from etml.cli import main
        self.series_input()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=main(['demo','--status','--projects-dir',str(self.root/'projects')])
        self.assertEqual(code,0,output.getvalue())
        self.assertIn('time_series',output.getvalue()); self.assertIn('--models',output.getvalue())

    def test_new_project_can_start_at_models(self):
        from etml.cli import main
        from unittest.mock import patch
        with patch('builtins.input',side_effect=['1','image-project','2']),contextlib.redirect_stdout(io.StringIO()) as output:
            code=main(['--models','--projects-dir',str(self.root/'projects')])
        self.assertEqual(code,0,output.getvalue())
        self.assertTrue((self.root/'projects/image-project/project.json').exists())

    def test_forecast_training_and_export(self):
        path=self.series_input()
        entries=[]
        if importlib.util.find_spec('statsmodels'): entries.append(('statsmodels:arima',{'order':[1,0,0]}))
        if importlib.util.find_spec('torch'): entries.extend((('pytorch:lstm',{'epochs':2,'window':5,'hidden_size':4}),('pytorch:gru',{'epochs':2,'window':5,'hidden_size':4})))
        for key,params in entries:
            with self.subTest(key=key):
                out=self.root/key.split(':')[1]
                result=train_specialized(path,key,out,params=params)
                self.assertIn('rmse',result['validation']); self.assertIsNotNone(result['baseline'])
                prediction=predict_specialized(out,{'horizon':3})
                exported=export_specialized(out,self.root/(out.name+'-export'))
                np.testing.assert_allclose(prediction['forecast'],predict_specialized(exported,{'horizon':3})['forecast'])
                self.assertEqual(len(prediction['forecast']),3)

    def test_recommendation_pairs_and_cold_start(self):
        frame=pd.DataFrame([(str(u),str(i),1.) for u in range(8) for i in range(10) if (u+i)%3!=0],columns=['user','item','weight'])
        source=self.root/'interactions.csv'; pd.concat([frame,frame.iloc[:2]]).to_csv(source,index=False)
        path=prepare_model_input(self.project,{'modality':'recommendation','train':str(source),
            'columns':{'user':'user','item':'item','weight':'weight'},'task':'recommendation'})
        _,info=load_model_input(path); self.assertEqual(sum(info['split_counts'].values()),len(frame))
        for key in ['scipy:svd']+(['implicit:als'] if importlib.util.find_spec('implicit') else []):
            with self.subTest(key=key):
                out=self.root/key.split(':')[1]; result=train_specialized(path,key,out,params={'factors':2})
                self.assertIn('recall_at_10',result['validation'])
                pred=predict_specialized(out,{'user':'new','k':3}); self.assertTrue(pred['popularity_fallback'])
                self.assertEqual(len(pred['recommendations']),3)
                seen=pd.read_parquet(path/'splits/train.parquet'); seen=set(seen[seen.user=='0'].item)
                self.assertFalse(seen & {x['item'] for x in predict_specialized(out,{'user':'0','k':10})['recommendations']})

    @unittest.skipUnless(importlib.util.find_spec('torchvision'),'vision optional')
    def test_lazy_image_import_train_predict(self):
        from PIL import Image
        images=[]
        for i in range(20):
            path=self.root/f'{i}.png'; Image.new('RGB',(36,36),(i*10,30,90)).save(path)
            images.append({'image':path.name,'label':i%2})
        source=self.root/'images.csv'; pd.DataFrame(images).to_csv(source,index=False)
        inputs=prepare_model_input(self.project,{'modality':'image','train':str(source),
            'columns':{'image':'image','target':'label'},'task':'classification','fractions':[.6,.2,.2]})
        path=self.root/'lenet'; result=train_specialized(inputs,'torchvision:lenet',path,params={'epochs':1,'batch_size':4})
        self.assertIn('f1_macro',result['validation'])
        output=predict_specialized(path,{'image':str(self.root/'0.png')})
        self.assertEqual(len(output['probabilities']),2)

    def test_text_input_and_cli_help(self):
        from etml.cli import main
        source=self.root/'text.csv'; pd.DataFrame({'text':['hello '+str(i) for i in range(20)]}).to_csv(source,index=False)
        path=prepare_model_input(self.project,{'modality':'text','task':'language_modeling','train':str(source),'columns':{'text':'text'}})
        self.assertTrue((path/'raw/train.csv').exists())
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(main(['models','help','transformers:gpt']),0)
        self.assertIn('GPT-2',capture.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(main(['models','list','--task-type','forecasting']),0)
        self.assertIn('arima',capture.getvalue())

    @unittest.skipUnless(importlib.util.find_spec('transformers'),'text optional')
    def test_offline_transformer_training_and_generation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (PreTrainedTokenizerFast,BertConfig,BertForSequenceClassification,
            GPT2Config,GPT2LMHeadModel,T5Config,T5ForConditionalGeneration)
        vocab={'[PAD]':0,'[UNK]':1,'[BOS]':2,'[EOS]':3,'hello':4,'world':5,'good':6,'bad':7}
        tok=Tokenizer(WordLevel(vocab,unk_token='[UNK]')); tok.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',unk_token='[UNK]',
            bos_token='[BOS]',eos_token='[EOS]',model_input_names=['input_ids','attention_mask'])
        cases=[('bert','classification',BertForSequenceClassification(BertConfig(vocab_size=8,hidden_size=8,
            num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,num_labels=2,max_position_embeddings=32))),
            ('gpt','language_modeling',GPT2LMHeadModel(GPT2Config(vocab_size=8,n_embd=8,n_layer=1,n_head=2,n_positions=32,
                bos_token_id=2,eos_token_id=3,pad_token_id=0))),
            ('t5','text_to_text',T5ForConditionalGeneration(T5Config(vocab_size=8,d_model=8,d_ff=16,d_kv=4,
                num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,eos_token_id=3,pad_token_id=0)))]
        for algorithm,task,model in cases:
            with self.subTest(algorithm=algorithm):
                checkpoint=self.root/(algorithm+'-checkpoint'); model.save_pretrained(checkpoint,safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                frame=pd.DataFrame({'text':['hello world','good world','bad world','hello good']*5,
                                    'target':[0,1]*10 if task=='classification' else ['hello good']*20})
                source=self.root/(algorithm+'.csv'); frame.to_csv(source,index=False)
                cols={'text':'text'}
                if task!='language_modeling': cols['target']='target'
                inputs=prepare_model_input(self.project,{'modality':'text','task':task,'train':str(source),'columns':cols,
                                                         'fractions':[.6,.2,.2]})
                out=self.root/(algorithm+'-bundle')
                result=train_specialized(inputs,'transformers:'+algorithm,out,params={'checkpoint':str(checkpoint),
                    'epochs':1,'batch_size':4,'max_length':8},max_bytes=3*1024**3)
                self.assertEqual(result['epochs_completed'],1)
                output=predict_specialized(out,{'text':'hello world','max_new_tokens':3})
                self.assertIn('prediction' if task=='classification' else 'text',output)

    def test_named_tabular_and_project_prediction(self):
        from workflows.project_preparation import ProjectPreparation
        from workflows.project_models import train_named_tabular
        from preprocessing import Recipe
        from splitting import SplitConfig
        from etml.cli import main
        frame=pd.DataFrame({'x':np.arange(60),'target':np.arange(60)*2.})
        source=self.root/'tabular.csv'; frame.to_csv(source,index=False)
        self.project.workspace.import_files(source,dataset_id='data')
        service=ProjectPreparation(self.project); service.configure('target','regression')
        service.save_recipe(Recipe(())); fitted,_,_=service.fit_preview(); service.save_processed(fitted)
        service.split(SplitConfig())
        _,candidate=train_named_tabular(self.project,service,'sklearn:ridge',{},name='my-model')
        self.assertFalse(candidate['is_baseline'])
        store=ProjectModels(self.project); self.assertTrue(store.bundle('my-model').is_dir())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=main(['demo','--predict-model','my-model','--values','{"x": 8}',
                       '--projects-dir',str(self.root/'projects')])
        self.assertEqual(code,0,output.getvalue())
        self.assertIn('prediction',output.getvalue())

    def test_guided_regression_loss_choices(self):
        from etml.model_wizard import _settings
        questions=[]
        def choose(title,choices):
            questions.append((title,choices))
            return 1 if title=='Training settings' else 0
        options=_settings(CATALOG['pytorch:mlp'],choose,lambda prompt:'','regression')
        losses=next(choices for title,choices in questions if title=='loss')
        self.assertIn('mae',losses); self.assertNotIn('cross_entropy',losses)
        self.assertEqual(options['loss'],'auto')

    @unittest.skipUnless(importlib.util.find_spec('torchvision'),'vision optional')
    def test_vision_architecture_heads_without_allocating_weights(self):
        import torch
        from training.specialized import _torch_model
        for algorithm in ('alexnet','vgg16','resnet18','vit_b_16'):
            with self.subTest(algorithm=algorithm),torch.device('meta'):
                model,_=_torch_model({'model':'torchvision:'+algorithm,'options':{'pretrained':False},'seed':42},{'classes':['a','b','c']})
                result=model(torch.zeros(2,3,224,224,device='meta'))
                self.assertEqual(tuple(result.shape),(2,3))

    def test_new_models_full_supervised_bundle_workflow(self):
        from tasks import TaskConfig
        from workflows import prepare_task
        from training import train_models
        from prediction import Predictor
        from exporting import export_native
        frame=pd.DataFrame({'x':np.arange(90)/90,'category':['a','b','c']*30,'target':[0,1,2]*30})
        source=self.root/'full.csv'; frame.to_csv(source,index=False)
        self.project.workspace.import_files(source,dataset_id='data')
        prep=prepare_task(self.project.workspace,'data',TaskConfig('t','target','classification'))
        configs=[ModelConfig('sklearn','svm'),ModelConfig('sklearn','multinomial_logistic')]
        for backend in ('lightgbm','catboost'):
            if importlib.util.find_spec(backend):
                configs.append(ModelConfig(backend,'boosted_trees',params={'iterations' if backend=='catboost' else 'n_estimators':5}))
        result=train_models(self.project.workspace,'data','t',prep.run_id,configs=configs)
        self.assertEqual(len(result.leaderboard),len(configs))
        for candidate in result.leaderboard:
            bundle=result.directory/'candidates'/candidate['candidate']
            self.assertEqual(len(Predictor.load(bundle).predict(frame.iloc[:2])),2)
            export_native(bundle,self.root/('native-'+candidate['candidate']))
