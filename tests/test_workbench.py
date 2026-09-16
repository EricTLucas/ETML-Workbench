from contextlib import redirect_stdout,redirect_stderr
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import math
import os
import subprocess
import sys
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from eda_tool.pipeline import run_eda
from eda_tool.profiler import ProfileConfig
from eda_tool.visualizer import Visualizer,VisualizerConfig
from eda_tool.artifacts import load_run
from eda_tool.artifacts import codec
from eda_tool.html_report import HtmlReport,build_html_report
from eda_tool.cli import main


class HtmlInspector(HTMLParser):
    def __init__(self): super().__init__();self.tags=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        if tag=='img':self.images.append(dict(attrs))


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.df=pd.DataFrame({'x':np.arange(32,dtype=float),'y':np.arange(32,dtype=float)*2,
                              'group':['A','B']*16,'when':pd.date_range('2025-01-01',periods=32,tz='UTC')})
        self.df.loc[2,'x']=np.nan
        self.source=self.root/'input.csv';self.df.to_csv(self.source,index=False)
        self.pc=ProfileConfig(batch_size=7,sample_size=12,roles={'group':'category'})
        self.vc=VisualizerConfig(max_auto_charts=3)

    def run_small(self,**kwargs):
        result=run_eda(self.df,profile_config=self.pc,visualizer_config=self.vc,**kwargs)
        self.addCleanup(result.close)
        return result

    def test_memory_run_silent_and_progress(self):
        events=[];out=StringIO();err=StringIO();before=set(self.root.iterdir())
        with redirect_stdout(out),redirect_stderr(err):
            result=self.run_small(progress=events.append)
        self.assertEqual(out.getvalue(),'');self.assertEqual(err.getvalue(),'')
        self.assertEqual(set(self.root.iterdir()),before)
        self.assertEqual(result.profile['summary'].data['rows'],32)
        self.assertEqual([e.stage for e in events],['opening','profiling','complete'])
        self.assertIsNone(result.output_dir);self.assertEqual(result.charts,[])

    def test_summary_and_optional_html(self):
        a=self.run_small(generate_summary=True)
        self.assertEqual(len(a.charts),3);self.assertIsNone(a.html)
        b=self.run_small(export_html=True,title='Orchid example')
        self.assertIn('<!doctype html>',b.html)
        self.assertEqual(len(b.charts),3)
        self.assertIsNone(b.html_path)

    def test_explicit_requests_without_summary(self):
        r=self.run_small(chart_requests=[{'kind':'raincloud','x':'x','group':'group'}],export_html=True)
        self.assertEqual([c.kind for c in r.charts],['raincloud'])

    def test_saved_profile_excludes_rows_by_default(self):
        directory=self.root/'run'
        r=self.run_small(output_dir=directory)
        self.assertFalse((directory/'sample.parquet').exists())
        loaded=load_run(directory)
        self.assertIsNone(loaded.profile['interactions'].data['sample'])
        self.assertIsNone(loaded.profile['summary'].data['sample_values_first'])
        self.assertIsNotNone(r.profile['interactions'].data['sample'])
        self.assertEqual(loaded.profile['columns'].data['x']['mean'],r.profile['columns'].data['x']['mean'])
        assert_frame_equal(loaded.profile['correlations'].data,r.profile['correlations'].data)
        self.assertEqual(loaded.profile['columns'].data['when']['min'],self.df.when.min())

    def test_sample_roundtrip_and_no_source_read(self):
        directory=self.root/'run'
        r=self.run_small(output_dir=directory,save_sample=True)
        with patch('eda_tool.loader.open_dataset',side_effect=AssertionError('must not read source')):
            loaded=load_run(directory)
            assert_frame_equal(loaded.profile['interactions'].data['sample'],r.profile['interactions'].data['sample'])
            chart=Visualizer(loaded.profile).plot('scatter',x='x',y='y')
            self.addCleanup(chart.close)
            self.assertEqual(chart.metadata['method'],'sampled')
            self.assertEqual(chart.metadata['status'],'ok')

    def test_all_artifacts_and_saved_gallery(self):
        directory=self.root/'run'
        r=self.run_small(output_dir=directory,generate_summary=True,export_html=True,save_sample=True)
        self.assertTrue(r.html_path.exists())
        loaded=load_run(directory)
        self.assertEqual(len(loaded.charts),3)
        report=HtmlReport(loaded.profile,loaded.charts).save(self.root/'second.html')
        inspector=HtmlInspector();inspector.feed(report.read_text(encoding='utf-8'))
        self.assertEqual(len(inspector.images),3)
        self.assertTrue(all(i['src'].startswith('data:image/png;base64,') for i in inspector.images))
        self.assertNotIn('script',inspector.tags)

    def test_html_escapes_values(self):
        evil='<script>alert("x")</script>'
        r=run_eda(pd.DataFrame({evil:[1.,np.nan]}))
        self.addCleanup(r.close)
        html=HtmlReport(r.profile,title=evil).render()
        self.assertNotIn(evil,html);self.assertIn('&lt;script&gt;',html)
        parser=HtmlInspector();parser.feed(html)
        self.assertNotIn('script',parser.tags)
        self.assertIn('Unavailable',html)

    def test_existing_destination_preserved(self):
        folder=self.root/'existing';folder.mkdir();(folder/'keep.txt').write_text('keep')
        with patch('eda_tool.workflows.eda.Profiler.run',side_effect=AssertionError('no profiling')):
            with self.assertRaises(FileExistsError):self.run_small(output_dir=folder)
        self.assertEqual((folder/'keep.txt').read_text(),'keep')

    def test_failed_save_is_not_a_partial_run(self):
        folder=self.root/'failed'
        with patch('eda_tool.workflows.eda.write_run',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.run_small(output_dir=folder)
        self.assertFalse(folder.exists())
        self.assertEqual(list(self.root.glob('.eda-staging-*')),[])

    def test_invalid_options_and_sample_requirements(self):
        with self.assertRaises(ValueError):self.run_small(save_sample=True)
        with self.assertRaises(ValueError):self.run_small(chart_requests=[{}])
        with self.assertRaises(ValueError):
            run_eda(self.df,profile_config=ProfileConfig(interactions=False),output_dir=self.root/'no',save_sample=True)

    def test_integrity_and_path_traversal(self):
        folder=self.root/'run';self.run_small(output_dir=folder)
        profile=folder/'profile.json';original=profile.read_text(encoding='utf-8')
        profile.write_text(original+' ',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Integrity'):load_run(folder)
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest['profile']='../input.csv'
        (folder/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'inside'):load_run(folder)

    def test_codec_nonfinite_and_user_tag(self):
        data={'type':'timestamp', 'value':'ordinary text', 'values':(np.int64(2),np.nan,np.inf,pd.NA),
              'date':pd.Timestamp('2025-01-01',tz='UTC'),'mapping':{1:'one',None:'null'}}
        encoded=codec.encode(data)
        text=json.dumps(encoded,allow_nan=False)
        loaded=codec.decode(json.loads(text))
        self.assertEqual(loaded['value'],'ordinary text')
        self.assertTrue(math.isnan(loaded['values'][1]));self.assertTrue(math.isinf(loaded['values'][2]))
        self.assertIs(loaded['values'][3],pd.NA)
        self.assertEqual(loaded['date'],data['date'])
        self.assertEqual(loaded['mapping'],data['mapping'])

    def cli(self,args):
        out,err=StringIO(),StringIO()
        with redirect_stdout(out),redirect_stderr(err):code=main(args)
        return code,out.getvalue(),err.getvalue()

    def test_cli_analyze_and_saved_plot(self):
        folder=self.root/'cli'
        code,out,err=self.cli(['eda','analyze',str(self.source),'--output',str(folder),'--save-sample','--sample-size','10','--json','--quiet'])
        self.assertEqual(code,0,err);self.assertEqual(json.loads(out)['rows'],32);self.assertEqual(err,'')
        image=self.root/'scatter.svg'
        self.source.unlink()
        code,out,err=self.cli(['eda','plot','--run',str(folder),'--kind','scatter','--x','x','--y','y','--option','trend=false','--output',str(image),'--json'])
        self.assertEqual(code,0,err);self.assertTrue(image.exists())
        self.assertEqual(json.loads(out)['status'],'ok')

    def test_cli_source_plot_and_no_data_error(self):
        image=self.root/'hist.png'
        code,out,err=self.cli(['eda','plot',str(self.source),'--kind','histogram','--x','x','--option','kde=false','--output',str(image)])
        self.assertEqual(code,0,err)
        folder=self.root/'run';self.run_small(output_dir=folder)
        code,out,err=self.cli(['eda','plot','--run',str(folder),'--kind','raincloud','--x','x','--output',str(self.root/'no.png')])
        self.assertEqual(code,2);self.assertIn('--save-sample',err);self.assertFalse((self.root/'no.png').exists())

    def test_cli_catalog_and_report(self):
        code,out,err=self.cli(['eda','charts','--json'])
        self.assertEqual(code,0);self.assertEqual(len(json.loads(out)),36)
        folder=self.root/'run';self.run_small(output_dir=folder,chart_requests=[{'kind':'histogram','x':'x'}])
        code,out,err=self.cli(['eda','report','--run',str(folder),'--output',str(self.root/'report.html'),'--json'])
        self.assertEqual(code,0,err);self.assertEqual(json.loads(out)['charts'],1)

    def test_cli_errors_and_progress_stream(self):
        code,out,err=self.cli(['eda','analyze',str(self.source),'--output',str(self.root/'bad'),'--role','x=not_a_role'])
        self.assertEqual(code,2);self.assertIn('error:',err);self.assertEqual(out,'')
        code,out,err=self.cli(['eda','analyze',str(self.source),'--output',str(self.root/'good'),'--json'])
        self.assertEqual(code,0,err);self.assertEqual(json.loads(out)['status'],'ok');self.assertIn('Computing',err)

    def test_real_module_cli(self):
        completed=subprocess.run([sys.executable,'-m','eda_tool','eda','charts','--json'],capture_output=True,text=True,timeout=60)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn('raincloud',json.loads(completed.stdout))

    def test_legacy_html_adapter(self):
        r=self.run_small()
        html=build_html_report(r.profile,charts=[],path=Path('example.csv'))
        self.assertIn('<title>example</title>',html)


if __name__=='__main__':unittest.main()
