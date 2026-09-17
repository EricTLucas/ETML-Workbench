import unittest
import pandas as pd
from tasks import TaskConfig, validate_task
from tasks.validation import eligible_rows
from splitting import SplitConfig
from preprocessing import Recipe, Step


class TaskConfigTests(unittest.TestCase):
    def test_roundtrip_and_exclusions(self):
        task = TaskConfig('survival','y','classification',('id',))
        self.assertEqual(TaskConfig.from_dict(task.to_dict()),task)
        self.assertEqual(validate_task(task,['x','y','id','group'],SplitConfig(strategy='group',group_column='group'),Recipe()),('x',))

    def test_invalid_settings(self):
        for kwargs in ({'task_id':'../escape'}, {'task_type':'unknown'}, {'missing_target':'mean'}, {'excluded_columns':('y',)}):
            values={'task_id':'task','target':'y','task_type':'regression',**kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TaskConfig(**values)
        for kwargs in ({'train':.9}, {'seed':True}, {'strategy':'stratified','group_column':'g'}, {'strategy':'chronological','time_column':'t'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SplitConfig(**kwargs)

    def test_target_cannot_be_transformed_or_created(self):
        task=TaskConfig('t','y','regression')
        for step in (Step('fill_missing',('y',),{'value':0}),Step('rename_columns',('x',),{'names':['y']}),
                     Step('rename_columns',('x',),{'names':['__etml_fake']})):
            with self.assertRaises(ValueError):
                validate_task(task,['x','y'],SplitConfig(),Recipe((step.approve(),)))

    def test_missing_and_regression_targets(self):
        frame=pd.DataFrame({'x':[1,2],'y':[1.,None]})
        with self.assertRaises(ValueError):
            eligible_rows(frame,TaskConfig('t','y','regression'))
        self.assertEqual(eligible_rows(frame,TaskConfig('t','y','regression',missing_target='drop')).tolist(),[True,False])
        for values in (['1','2'],[1.,float('inf')]):
            with self.assertRaises(ValueError):
                eligible_rows(pd.DataFrame({'y':values}),TaskConfig('t','y','regression'))

    def test_unapproved_recipe_and_missing_features(self):
        with self.assertRaises(ValueError):
            validate_task(TaskConfig('t','y','regression'),['y'],SplitConfig(),Recipe())
        with self.assertRaises(ValueError):
            validate_task(TaskConfig('t','y','regression'),['x','y'],SplitConfig(),Recipe((Step('fill_missing',('x',),{'value':0}),)))


if __name__=='__main__': unittest.main()
