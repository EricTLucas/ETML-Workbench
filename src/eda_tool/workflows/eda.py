"""Shared EDA workflow used by Python, CLI and future UI integrations."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
import platform
import uuid
import pandas as pd
from ..loader import open_dataset
from ..profiler import Profiler, ProfileConfig
from ..visualizer import Visualizer, VisualizerConfig
from ..html_report import HtmlReport
from ..artifacts.runs import write_run


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str


@dataclass
class EDARun:
    profile: dict
    charts: list = field(default_factory=list)
    output_dir: object = None
    manifest: dict = field(default_factory=dict)
    html: object = None
    html_path: object = None

    @property
    def results(self): return self.profile

    def close(self):
        for chart in self.charts: chart.close()


def _versions():
    result = {'python':platform.python_version(),'eda_tool':'0.2.0'}
    for name in ('pandas','numpy','matplotlib','pyarrow','wordcloud'):
        try: result[name] = version(name)
        except PackageNotFoundError: pass
    return result


def _source_description(source, dataset):
    files = []
    for item in getattr(dataset, '_items', ()):
        if isinstance(item,(str,Path)):
            p = Path(item).resolve(); stat = p.stat()
            files.append({'path':str(p),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if files: return {'kind':'files','files':files,'identity':'path/size/mtime; not a content hash'}
    return {'kind':'DataFrame' if isinstance(source,pd.DataFrame) else type(source).__name__,
            'identity':'in-memory or uploaded source; contents not recorded'}


def run_eda(source, *, profile_config=None, visualizer_config=None, loader_options=None,
            generate_summary=False, chart_requests=None, output_dir=None, export_html=False,
            save_sample=False, title='Exploratory data analysis', progress=None):
    """Profile once, optionally render/save charts and a self-contained gallery.

    No output directory means no filesystem writes. HTML then lives in result.html.
    save_sample requires output_dir and explicit consent from the caller. Existing
    output directories are never overwritten. HTML implies summary charts only
    when neither summary nor explicit chart requests were supplied.
    """
    pc = profile_config or ProfileConfig()
    vc = visualizer_config or VisualizerConfig()
    requests = list(chart_requests or [])
    if len(requests)>100: raise ValueError('At most 100 explicit chart requests per run')
    for request in requests:
        if not isinstance(request,dict) or not isinstance(request.get('kind'),str):
            raise ValueError('Each chart request needs a string kind')
    target = Path(output_dir).resolve() if output_dir is not None else None
    if target is not None and target.exists():
        raise FileExistsError(f'Output already exists: {target}. Choose a new run directory.')
    if save_sample and (target is None or not pc.interactions):
        raise ValueError('save_sample requires an output directory and profiler interactions enabled')
    def emit(stage, message):
        if progress is not None: progress(ProgressEvent(stage,message))
    emit('opening','Opening dataset')
    dataset = open_dataset(source, **(loader_options or {}))
    source_info = _source_description(source,dataset)
    emit('profiling','Computing batch profile')
    profile = Profiler(dataset,pc).run()
    result = EDARun(profile)
    try:
        if generate_summary or requests or export_html:
            emit('plotting','Rendering charts')
            viz = Visualizer(profile,config=vc)
            if generate_summary or (export_html and not requests):
                result.charts.extend(viz.summary())
            for request in requests:
                kwargs = dict(request); kind = kwargs.pop('kind')
                result.charts.append(viz.plot(kind,**kwargs))
        if len(result.charts)>100: raise ValueError('Combined summary and requests exceed 100 charts')
        if export_html:
            emit('reporting','Building optional HTML gallery')
            result.html = HtmlReport(profile,result.charts,title=title).render()
        manifest = {'run_id':uuid.uuid4().hex,'created_utc':datetime.now(timezone.utc).isoformat(),
                    'source':source_info,'profile_config':asdict(pc),'visualizer_config':asdict(vc),
                    'versions':_versions(),'title':title,
                    'source_loader_options':{str(k):str(v) for k,v in (loader_options or {}).items()},
                    'privacy':'No row sample unless opted in. Aggregates/charts may contain original values.'}
        if target is not None:
            emit('saving','Saving run artifacts')
            target.parent.mkdir(parents=True,exist_ok=True)
            # TemporaryDirectory owns only its generated sibling directory.
            with TemporaryDirectory(prefix='.eda-staging-',dir=target.parent) as temp:
                staging = Path(temp).resolve()
                if staging.parent != target.parent: raise ValueError('Invalid staging location')
                manifest = write_run(staging,profile,result.charts,manifest,save_sample=save_sample,html=result.html)
                if target.exists(): raise FileExistsError(f'Output appeared during execution: {target}')
                staging.rename(target)
            result.output_dir = target
            result.html_path = target/'report.html' if export_html else None
        result.manifest = manifest
        emit('complete','EDA complete')
        return result
    except BaseException:
        result.close()
        raise
