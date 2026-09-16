import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib as mpl
from pandas.testing import assert_frame_equal
from eda_tool.visualizer import Visualizer, VisualizerConfig, visualize_dataset, plotNumericColumn
from eda_tool.visualization.core import safe_stem


class VisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(42)
        cls.frame=pd.DataFrame({'x':rng.normal(20,4,120), 'y':rng.normal(10,2,120),
                               'size':rng.uniform(0,10,120), 'group':['A','B','C']*40,
                               'category':['Low','High']*60,
                               'text':['orchid data models','explore data shapes','models and charts']*40,
                               'date':pd.date_range('2025-01-01',periods=120)})
        cls.frame.loc[[1,5,10],'x']=np.nan

    def setUp(self):
        self.v=Visualizer(self.frame)

    def test_every_chart_renders(self):
        requests={
            'histogram':{'x':'x'},'density':{'x':'x','group':'group'},'ecdf':{'x':'x'},
            'box':{'x':'x','group':'group'},'violin':{'x':'x'},'raincloud':{'x':'x','group':'group'},
            'ridgeline':{'x':'x','group':'group'},'strip':{'x':'x'},'qq':{'x':'x'},'outliers':{'x':'x'},
            'scatter':{'x':'x','y':'y','group':'group'},'bubble':{'x':'x','y':'y','size':'size'},
            'density2d':{'x':'x','y':'y'},'hexbin':{'x':'x','y':'y'},'contour':{'x':'x','y':'y'},
            'bar':{'x':'group'},'lollipop':{'x':'group'},'pie':{'x':'group'},'donut':{'x':'group'},
            'category_heatmap':{'x':'group','y':'category','normalize':True},
            'grouped_bar':{'x':'category','group':'group'},'missing_bar':{},'missing_matrix':{},
            'missing_patterns':{},'correlation':{'columns':['x','y','size'],'bubbles':True},
            'association':{},'word_frequency':{'x':'text'},'wordcloud':{'x':'text'},
            'text_length':{'x':'text'},'time_series':{'x':'date','y':'y','frequency':'W','area':True},
            'scatter_matrix':{'columns':['x','y']},'parallel':{'columns':['x','y','size']},
            'joint':{'x':'x','y':'y'},'stacked_bar':{'x':'group','group':'category'},
            'line':{'x':'x','y':'y','area':True},'date_counts':{'x':'date','frequency':'W'}}
        self.assertEqual(set(requests),set(self.v.available()))
        for name,kwargs in requests.items():
            with self.subTest(chart=name):
                chart=self.v.plot(name,**kwargs)
                chart.figure.canvas.draw()
                self.assertEqual(chart.metadata['status'],'no_data' if name=='association' else 'ok')
                self.assertEqual(chart.metadata['palette'],'Orchid')
                self.assertEqual(mpl.colors.to_hex(chart.figure.get_facecolor()),'#080b10')
                chart.close()

    def test_profiler_acceptance_and_run_once(self):
        from eda_tool.profiler import Profiler,ProfileConfig
        class CountingProfiler(Profiler):
            calls=0
            def run(self):
                self.calls+=1
                return super().run()
        profiler=CountingProfiler(self.frame,ProfileConfig(sample_size=50,roles={'group':'category'}))
        v=Visualizer(profiler)
        for kind,kw in [('histogram',{'x':'x'}),('association',{}),('missing_bar',{})]:
            r=v.plot(kind,**kw)
            self.assertEqual(r.metadata['status'],'ok');r.close()
        self.assertEqual(profiler.calls,1)
        self.assertEqual(v.plot('histogram',x='x').metadata['method'],'sampled')
        self.assertEqual(v.plot('missing_bar').metadata['method'],'exact')

    def test_histogram_and_hexbin_counts(self):
        h=self.v.plot('histogram',x='x',kde=False)
        self.assertEqual(sum(p.get_height() for p in h.axes[0].patches),117)
        b=self.v.plot('hexbin',x='x',y='y')
        self.assertEqual(b.metadata['binned_count'],117)
        d=self.v.plot('density2d',x='x',y='y')
        self.assertEqual(np.array(d.metadata['bin_counts']).sum(),117)

    def test_ecdf_ends_at_one_and_fit(self):
        ecdf=self.v.plot('ecdf',x='x')
        self.assertEqual(ecdf.axes[0].lines[0].get_ydata()[-1],1)
        v=Visualizer(pd.DataFrame({'x':range(20),'y':np.arange(20)*3+4}))
        fit=v.plot('scatter',x='x',y='y').metadata['linear_fit']
        self.assertAlmostEqual(fit['slope'],3)
        self.assertAlmostEqual(fit['intercept'],4)

    def test_pairwise_spearman(self):
        f=pd.DataFrame({'a':[1.,2.,3.,4.,5.], 'b':[5.,np.nan,1.,2.,3.]})
        chart=Visualizer(f).plot('correlation',method='spearman',columns=['a','b'])
        pair=f.dropna()
        self.assertAlmostEqual(chart.metadata['matrix'].loc['a','b'],pair.a.rank().corr(pair.b.rank()))

    def test_no_mutation_or_global_style(self):
        before=self.frame.copy(deep=True)
        style=dict(mpl.rcParams)
        self.v.plot('raincloud',x='x',group='group')
        assert_frame_equal(self.frame,before)
        for key in ['figure.facecolor','font.size','axes.facecolor','text.usetex']:
            self.assertEqual(style[key],mpl.rcParams[key])

    def test_bounded_and_repeatable(self):
        a=Visualizer(self.frame,config=VisualizerConfig(max_points=20,max_auto_charts=3))
        b=Visualizer(self.frame,config=VisualizerConfig(max_points=20))
        assert_frame_equal(a.data,b.data)
        self.assertEqual(len(a.data),20)
        self.assertEqual(a.meta['method'],'sampled')
        self.assertEqual(len(a.summary()),3)

    def test_degenerate_inputs(self):
        for vals in [[1.],[2.,2.,2.],[np.nan,np.inf,-np.inf]]:
            for kind in ['histogram','density','box','violin','raincloud','ecdf','qq']:
                with self.subTest(values=vals,kind=kind):
                    r=Visualizer(pd.DataFrame({'x':vals})).plot(kind,x='x')
                    self.assertEqual(r.metadata['status'],'no_data' if not np.isfinite(vals).any() else 'ok')
        empty=Visualizer(pd.DataFrame({'x':pd.Series(dtype=float)})).plot('histogram',x='x')
        self.assertEqual(empty.metadata['status'],'no_data')

    def test_validation_and_save(self):
        for kind,opts in [('absent',{}),('scatter',{'x':'x'}),('histogram',{'x':'bad'}),
                          ('histogram',{'x':'x','made_up':True}),('histogram',{'x':'x','bins':0})]:
            with self.assertRaises((ValueError,TypeError)):
                self.v.plot(kind,**opts)
        with tempfile.TemporaryDirectory() as folder:
            chart=self.v.plot('raincloud',x='x')
            for ext in ['png','svg','pdf']:
                path=chart.save(Path(folder)/('chart.'+ext))
                self.assertGreater(path.stat().st_size,500)
            with self.assertRaises(ValueError):chart.save(Path(folder)/'chart.exe')
        self.assertNotIn('/',safe_stem('../../escape'))
        self.assertNotEqual(safe_stem('a/b'),safe_stem('a?b'))

    def test_old_entrypoints(self):
        from eda_tool.profiler import Profiler
        profile=Profiler(self.frame).run()
        with tempfile.TemporaryDirectory() as folder:
            figures=visualize_dataset(self.frame,profile,folder)
            self.assertGreater(len(figures),0)
            self.assertTrue((Path(folder)/'hist_x.png').exists())
            fig=plotNumericColumn(self.frame,profile,'x',folder)
            self.assertGreater(len(fig.axes),0)

    def test_no_sample_uses_aggregate_histogram(self):
        from eda_tool.profiler import Profiler,ProfileConfig
        profile=Profiler(self.frame,ProfileConfig(interactions=False)).run()
        v=Visualizer(profile)
        self.assertEqual(v.plot('histogram',x='x').metadata['status'],'ok')
        self.assertEqual(v.plot('raincloud',x='x').metadata['status'],'no_data')

    def test_group_cap_metadata_and_legacy_config(self):
        v=Visualizer(self.frame,config=VisualizerConfig(max_groups=1))
        chart=v.plot('raincloud',x='x',group='group')
        self.assertEqual(chart.metadata['omitted_groups'],2)
        self.assertEqual(chart.metadata['rows_used']+chart.metadata['omitted_observations'],117)
        from eda_tool.profiler import Profiler,ProfileConfig
        from eda_tool.visualizer import plotMissingBarChart
        profile=Profiler(self.frame,ProfileConfig(interactions=False)).run()
        charts=visualize_dataset(self.frame,profile,config=VisualizerConfig(max_auto_charts=2))
        self.assertGreater(len(charts),0)
        fig=plotMissingBarChart({'x':{},'y':{}},100,{'x':10})
        self.assertEqual([p.get_width() for p in fig.axes[0].patches],[90,100])

    def test_drilldown_and_stable_colors(self):
        self.assertEqual(set(self.v.column('x')),{'histogram','box','ecdf','raincloud'})
        self.assertEqual(set(self.v.column('date')),{'date_counts'})
        color=self.v.group_color('group','A')
        self.v.plot('scatter',x='x',y='y',group='group')
        self.assertEqual(color,self.v.group_color('group','A'))


if __name__=='__main__':
    unittest.main()
