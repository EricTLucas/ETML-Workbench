from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from splitting import SplitConfig, split_dataset
from tasks import TaskConfig, ROW_ID


class SplittingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.frame=pd.DataFrame({'x':range(120),'y':[i%3 for i in range(120)]})
        self.task=TaskConfig('t','y','classification')

    def run_split(self,frame=None,config=None,batch_size=7,task=None,name='split'):
        root=self.root/name
        stats=split_dataset(self.frame if frame is None else frame,root,task or self.task,
                            config or SplitConfig(strategy='stratified'),batch_size=batch_size)
        assignments=pd.read_parquet(root/'assignments/part-00000.parquet')
        return root,stats,assignments

    def test_reproducibility_across_batches_and_seed(self):
        _,_,a=self.run_split(name='a',batch_size=1)
        _,_,b=self.run_split(name='b',batch_size=23)
        assert_frame_equal(a,b)
        _,_,c=self.run_split(name='c',config=SplitConfig(strategy='stratified',seed=99))
        self.assertFalse(a.equals(c))

    def test_disjoint_complete_stratified_splits(self):
        root,stats,assignments=self.run_split()
        self.assertEqual(len(assignments),120)
        seen=set()
        for name in ('train','validation','test'):
            frame=pd.read_parquet(root/'splits'/name/'part-00000.parquet')
            ids=set(frame[ROW_ID])
            self.assertFalse(ids & seen); seen.update(ids)
            self.assertEqual(set(frame.y),{0,1,2})
            self.assertEqual(len(frame),stats['counts'][name])
        self.assertEqual(seen,set(range(120)))

    def test_random_exact_counts_and_no_pandas_index_dependency(self):
        self.frame.index=[4]*120
        _,stats,_=self.run_split(config=SplitConfig(strategy='random'))
        self.assertEqual(stats['counts'],{'train':84,'validation':18,'test':18,'excluded':0})

    def test_group_isolation(self):
        frame=self.frame.assign(group=[i//3 for i in range(120)])
        root,_,_=self.run_split(frame,SplitConfig(strategy='group',group_column='group'))
        groups=[]
        for name in ('train','validation','test'):
            groups.append(set(pd.read_parquet(root/'splits'/name/'part-00000.parquet').group))
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_chronological_order_and_ties(self):
        frame=self.frame.assign(time=[str(pd.Timestamp('2020-01-01')+pd.Timedelta(days=i//3)) for i in range(120)])
        root,_,_=self.run_split(frame,SplitConfig(strategy='chronological',time_column='time',time_format='ISO8601'))
        parts=[pd.read_parquet(root/'splits'/n/'part-00000.parquet') for n in ('train','validation','test')]
        self.assertLess(parts[0].time.max(),parts[1].time.min())
        self.assertLess(parts[1].time.max(),parts[2].time.min())

    def test_missing_targets_excluded_and_zero_fraction_supported(self):
        frame=self.frame.astype({'y':'Float64'}); frame.loc[0,'y']=pd.NA
        root,stats,assignment=self.run_split(frame,SplitConfig(strategy='random',train=.8,validation=.2,test=0),
                                            task=TaskConfig('t','y','classification',missing_target='drop'))
        self.assertEqual(stats['counts']['excluded'],1)
        self.assertEqual(assignment.iloc[0]['split'],'excluded')
        self.assertTrue(pd.read_parquet(root/'splits/test/part-00000.parquet').empty)

    def test_rare_class_and_empty_group_split_fail(self):
        with self.assertRaises(ValueError):
            self.run_split(pd.DataFrame({'x':range(6),'y':[0,0,0,0,0,1]}))
        with self.assertRaises(ValueError):
            self.run_split(self.frame.assign(g='only'),SplitConfig(strategy='group',group_column='g'),name='group')
        self.assertFalse(list(self.root.rglob('*.sqlite')))


if __name__=='__main__': unittest.main()
