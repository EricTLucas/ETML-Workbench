import unittest
import numpy as np
import pandas as pd
from eda_tool.visualizer import Visualizer
from eda_tool.profiler.base import SectionResult
from eda_tool.html_report import HtmlReport

class EDAFinishingTests(unittest.TestCase):
    def test_clean_data_omits_missing_plots(self):
        charts=Visualizer(pd.DataFrame({'x':range(30)})).summary()
        self.assertFalse(any(c.kind.startswith('missing_') for c in charts))
        for chart in charts: chart.close()

    def test_pair_selection_uses_absolute_strength_and_roles(self):
        frame=pd.DataFrame({'x':range(40),'y':range(40),'a':['a','b']*20,'b':['c','d']*20})
        pairs=[{'columns':p,'value':v,'status':'ok','method':m} for p,v,m in [
            (('x','y'),-.99,'pearson'),(('a','b'),.9,'cramers_v'),(('x','a'),.8,'correlation_ratio_eta'),(('y','b'),.1,'correlation_ratio_eta')]]
        profile={'summary':SectionResult('summary',{'rows':40,'percent_missing_cells':0}),
                 'columns':SectionResult('columns',{k:{'type':'numeric' if k in ['x','y'] else 'category'} for k in frame}),
                 'correlations':SectionResult('correlations',pd.DataFrame(),{'pairs':pairs})}
        charts=Visualizer(profile,data=frame).summary()
        self.assertEqual([c.kind for c in charts[:3]],['scatter','category_heatmap','box'])
        self.assertEqual(charts[0].request['y'],'y')
        for chart in charts:chart.close()

    def test_report_highlights_missing_and_unique(self):
        profile={'summary':SectionResult('summary',{'rows':20}),
                 'columns':SectionResult('columns',{'id':{'num_unique':20,'pct_missing':0},'x':{'num_unique':10,'pct_missing':.1}})}
        html=HtmlReport(profile,[]).render()
        self.assertIn('<span class="quality-alert">20</span>',html)
        self.assertIn('<span class="quality-alert">10.0%</span>',html)

