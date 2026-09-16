"""Whole-workbench command root, with EDA as the first implemented command group."""
import argparse
import json
from pathlib import Path
import sys
from .profiler import ProfileConfig
from .visualizer import Visualizer, VisualizerConfig
from .workflows import run_eda
from .artifacts import load_run
from .artifacts.codec import encode
from .html_report import HtmlReport


def _pairs(values):
    result={}
    for text in values or []:
        if '=' not in text: raise ValueError(f'Expected COLUMN=VALUE, got {text!r}')
        key,value=text.split('=',1)
        if not key: raise ValueError('Column name cannot be empty')
        result[key]=value
    return result


def _chart_options(args):
    options={}
    for key in ('x','y','group','size','title'):
        value=getattr(args,key,None)
        if value is not None: options[key]=value
    if args.columns is not None: options['columns']=args.columns
    for raw in args.option:
        if '=' not in raw: raise ValueError('--option requires NAME=JSON_VALUE')
        key,value=raw.split('=',1)
        if key in options: raise ValueError(f'Duplicate chart option: {key}')
        options[key]=json.loads(value)
    return options


def _add_data_options(parser):
    parser.add_argument('--batch-size',type=int,default=50_000)
    parser.add_argument('--sample-size',type=int,default=10_000)
    parser.add_argument('--categorical-threshold',type=int,default=10,help='Auto-category maximum distinct values; 0 disables')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--role',action='append',default=[],metavar='COLUMN=ROLE')
    parser.add_argument('--dtype',action='append',default=[],metavar='COLUMN=DTYPE')
    parser.add_argument('--max-pairs',type=int,default=100)
    parser.add_argument('--no-correlations',action='store_true')


def _configs(args):
    pc=ProfileConfig(batch_size=args.batch_size,sample_size=args.sample_size,seed=args.seed,
                     roles=_pairs(args.role),categorical_threshold=args.categorical_threshold,max_pairs=args.max_pairs,correlations=not args.no_correlations)
    vc=VisualizerConfig(seed=args.seed,max_points=args.sample_size,
                         max_auto_charts=getattr(args,'max_charts',10))
    dtype=_pairs(args.dtype)
    return pc,vc,{'dtype':dtype} if dtype else {}


def parser():
    root=argparse.ArgumentParser(prog='workbench',description='ML workbench. EDA is the first available command group.')
    root.add_argument('--version',action='version',version='workbench 0.2.0')
    groups=root.add_subparsers(dest='group_command',required=True)
    eda=groups.add_parser('eda',help='Profile data, render charts and export galleries')
    commands=eda.add_subparsers(dest='command',required=True)
    analyze=commands.add_parser('analyze',help='Run batch EDA once and save a new run')
    analyze.add_argument('input',help='Dataset file, directory or quoted glob')
    analyze.add_argument('--output',required=True,help='New run directory (must not already exist)')
    analyze.add_argument('--summary',action='store_true',help='Generate a bounded summary chart set')
    analyze.add_argument('--html',action='store_true',help='Include a self-contained HTML gallery; implies summary')
    analyze.add_argument('--save-sample',action='store_true',help='Explicitly persist dataset rows in sample.parquet')
    analyze.add_argument('--max-charts',type=int,default=10)
    analyze.add_argument('--title',default='Exploratory data analysis')
    analyze.add_argument('--json',action='store_true',help='Machine-readable outcome on stdout')
    analyze.add_argument('--quiet',action='store_true',help='Suppress progress messages on stderr')
    _add_data_options(analyze)
    plot=commands.add_parser('plot',help='Render a chart from a source or existing run')
    plot.add_argument('input',nargs='?',help='Source dataset; mutually exclusive with --run')
    plot.add_argument('--run',help='Existing saved run; source is never reopened')
    plot.add_argument('--kind',required=True)
    plot.add_argument('--output',required=True,help='New .png, .svg or .pdf file')
    for flag in ('x','y','group','size','title'): plot.add_argument('--'+flag)
    plot.add_argument('--columns',nargs='+')
    plot.add_argument('--option',action='append',default=[],metavar='NAME=JSON',help='Any additional chart option, e.g. bins=20 or kde=false')
    plot.add_argument('--json',action='store_true')
    _add_data_options(plot)
    charts=commands.add_parser('charts',help='List chart kinds and accepted options')
    charts.add_argument('--json',action='store_true')
    report=commands.add_parser('report',help='Export existing saved charts as an HTML gallery')
    report.add_argument('--run',required=True)
    report.add_argument('--output',required=True)
    report.add_argument('--title')
    report.add_argument('--json',action='store_true')
    return root


def _print(payload, machine, text):
    print(json.dumps(payload,ensure_ascii=False,allow_nan=False) if machine else text)


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='charts':
            catalog=Visualizer.available()
            _print(catalog,args.json,'\n'.join(f"{name:18} {spec['description']}\n{'':18} Options: {', '.join(spec['options'])}" for name,spec in catalog.items()))
            return 0
        if args.command=='analyze':
            pc,vc,loader=_configs(args)
            def progress(event):
                if not args.quiet: print(event.message,file=sys.stderr)
            result=run_eda(args.input,profile_config=pc,visualizer_config=vc,loader_options=loader,
                           generate_summary=args.summary,export_html=args.html,save_sample=args.save_sample,
                           output_dir=args.output,title=args.title,progress=progress)
            try:
                rows=result.profile['summary'].data['rows']
                payload={'status':'ok','output':str(result.output_dir),'rows':rows,
                         'charts':len(result.charts),'html':str(result.html_path) if result.html_path else None,
                         'sample_saved':bool(result.manifest.get('sample'))}
                _print(payload,args.json,f"Saved EDA run: {result.output_dir}\nRows: {rows:,}; charts: {len(result.charts)}")
            finally: result.close()
            return 0
        if args.command=='plot':
            if bool(args.input)==bool(args.run): raise ValueError('Provide exactly one source dataset or --run')
            path=Path(args.output)
            if path.exists(): raise FileExistsError(f'Output already exists: {path}')
            if path.suffix.lower() not in {'.png','.svg','.pdf'}: raise ValueError('Chart output must be PNG, SVG or PDF')
            options=_chart_options(args)
            if args.run:
                run=load_run(args.run)
                vc=VisualizerConfig(**run.manifest.get('visualizer_config',{}))
                viz=Visualizer(run.profile,config=vc)
            else:
                pc,vc,loader=_configs(args)
                result=run_eda(args.input,profile_config=pc,visualizer_config=vc,loader_options=loader)
                viz=Visualizer(result.profile,config=vc)
            chart=viz.plot(args.kind,**options)
            try:
                if chart.metadata.get('status')=='no_data':
                    raise ValueError(chart.metadata.get('reason','No chart data')+' Use a run saved with --save-sample for row-based charts.')
                chart.save(path)
                _print({'status':'ok','output':str(path.resolve()),'metadata':encode(chart.metadata)},args.json,f'Saved chart: {path.resolve()}')
            finally: chart.close()
            return 0
        if args.command=='report':
            run=load_run(args.run)
            path=HtmlReport(run.profile,run.charts,title=args.title or run.manifest.get('title','EDA report')).save(args.output)
            _print({'status':'ok','output':str(path.resolve()),'charts':len(run.charts)},args.json,f'Saved HTML gallery: {path.resolve()}')
            return 0
    except (ValueError,TypeError,OSError,ImportError,ArithmeticError,KeyError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
    return 0


def legacy_main(argv=None):
    old=argparse.ArgumentParser(prog='eda',description='Compatibility alias; prefer workbench eda analyze')
    old.add_argument('input');old.add_argument('--out',default='reports');old.add_argument('--no-open',action='store_true')
    args=old.parse_args(argv)
    print('eda is a compatibility alias. Reports no longer open a browser automatically.',file=sys.stderr)
    return main(['eda','analyze',args.input,'--output',args.out,'--html'])


if __name__=='__main__':
    raise SystemExit(main())
