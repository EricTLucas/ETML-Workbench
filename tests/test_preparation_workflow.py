from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from data import DatasetWorkspace
from preprocessing import Recipe, Step, FittedRecipe
from splitting import SplitConfig
from splitting.strategies import priority
from tasks import TaskConfig, ROW_ID
from workflows import prepare_task


class PreparationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ws=DatasetWorkspace(self.root/'datasets')

    def import_frame(self,frame):
        path=self.root/'input.csv'; frame.to_csv(path,index=False)
        self.ws.import_files(path,dataset_id='demo')

    def test_training_only_fit_and_target_alignment(self):
        train_ids=set(sorted(range(100),key=lambda i:priority(42,i))[:70])
        frame=pd.DataFrame({'x':[1. if i in train_ids else 99999. for i in range(100)],
                            'y':[float(i*3) for i in range(100)], 'id':range(100)})
        frame.loc[min(train_ids),'x']=np.nan
        self.import_frame(frame)
        recipe=Recipe((Step('fill_missing',('x',),{'strategy':'mean'}).approve(),))
        result=prepare_task(self.ws,'demo',TaskConfig('predict','y','regression',('id',)),recipe=recipe,batch_size=9)
        self.assertEqual(result.fitted.states[0]['fills']['x'],1.)
        self.assertEqual(result.fitted.input_columns,('x',))
        loaded=FittedRecipe.load(result.directory/'preprocessing/fitted.json')
        self.assertEqual(loaded.states[0]['fills']['x'],1.)
        for name in ('train','validation','test'):
            prepared=pd.read_parquet(result.directory/'prepared'/name/'part-00000.parquet')
            self.assertEqual(prepared.y.tolist(),frame.loc[prepared[ROW_ID],'y'].tolist())
            self.assertNotIn('id',prepared.columns)
        manifest=self.ws.get_task_run('demo','predict',result.run_id,verify=True)
        self.assertEqual(manifest.metadata['fitting_scope'],'train features only; target excluded')

    def test_row_filter_preserves_target_alignment(self):
        frame=pd.DataFrame({'x':[float(i) if i%7 else np.nan for i in range(90)],'y':np.arange(90.)})
        self.import_frame(frame)
        recipe=Recipe((Step('drop_missing',('x',)).approve(),))
        result=prepare_task(self.ws,'demo',TaskConfig('t','y','regression'),recipe=recipe,batch_size=8)
        self.assertLess(sum(result.prepared_counts.values()),90)
        for name in ('train','validation','test'):
            part=pd.read_parquet(result.directory/'prepared'/name/'part-00000.parquet')
            self.assertEqual(part.y.tolist(),part[ROW_ID].astype(float).tolist())

    def test_disabled_test_split_and_empty_first_feature_batch(self):
        frame=pd.DataFrame({'x':[None]*10+list(range(50)),'y':np.arange(60.)})
        self.import_frame(frame)
        result=prepare_task(self.ws,'demo',TaskConfig('t','y','regression'),
                            split_config=SplitConfig(train=.8,validation=.2,test=0),batch_size=3)
        self.assertEqual(result.prepared_counts['test'],0)
        self.assertTrue((result.directory/'prepared/test/part-00000.parquet').exists())

    def test_failure_does_not_publish_and_raw_is_unchanged(self):
        frame=pd.DataFrame({'x':[None]*60,'y':np.arange(60.)})
        self.import_frame(frame)
        before=self.ws.get('demo').raw_files[0].read_bytes()
        recipe=Recipe((Step('fill_missing',('x',),{'strategy':'mean'}).approve(),))
        with self.assertRaises(ValueError):
            prepare_task(self.ws,'demo',TaskConfig('t','y','regression'),recipe=recipe)
        runs=self.ws.get('demo').directory/'tasks/t/runs'
        self.assertEqual(list(runs.iterdir()),[])
        self.assertEqual(self.ws.get('demo').raw_files[0].read_bytes(),before)

    def test_task_immutability_and_processed_source_guard(self):
        self.import_frame(pd.DataFrame({'x':range(30),'y':range(30)}))
        self.ws.create_task('demo',TaskConfig('t','y','regression'))
        with self.assertRaises(ValueError):
            self.ws.create_task('demo',TaskConfig('t','x','regression'))
        with self.assertRaises(ValueError):
            prepare_task(self.ws,'demo',TaskConfig('t','y','regression'),source_version='v1')

    def test_run_integrity_detects_modified_artifact(self):
        self.import_frame(pd.DataFrame({'x':range(30),'y':range(30)}))
        result=prepare_task(self.ws,'demo',TaskConfig('t','y','regression'))
        (result.directory/'split_config.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.ws.get_task_run('demo','t',result.run_id,verify=True)

    def test_filter_cannot_remove_whole_training_class(self):
        self.import_frame(pd.DataFrame({'x':[1. if i%2 else np.nan for i in range(60)],'y':[i%2 for i in range(60)]}))
        recipe=Recipe((Step('drop_missing',('x',)).approve(),))
        with self.assertRaisesRegex(ValueError,'entire target class'):
            prepare_task(self.ws,'demo',TaskConfig('t','y','classification'),recipe=recipe)

    def test_group_control_is_not_a_model_feature(self):
        self.import_frame(pd.DataFrame({'x':range(120),'y':[i%2 for i in range(120)],'household':[i//3 for i in range(120)]}))
        result=prepare_task(self.ws,'demo',TaskConfig('t','y','classification'),
                            split_config=SplitConfig(strategy='group',group_column='household'))
        self.assertEqual(result.feature_columns,('x',))
        self.assertNotIn('household',pd.read_parquet(result.directory/'prepared/train/part-00000.parquet').columns)


if __name__=='__main__': unittest.main()
