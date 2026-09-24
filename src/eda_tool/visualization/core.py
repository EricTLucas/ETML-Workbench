"""On-demand, bounded EDA rendering. No browser, file, or pyplot side effects."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import inspect
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class VisualizerConfig:
    colors: tuple = ('#C0A1FF', '#FF82B2', '#7CE1DF', '#EBCB8B', '#90ACFF', '#CEB8AD')
    background: str = '#080B10'
    foreground: str = '#E6EAF1'
    muted: str = '#A4ADBC'
    grid: str = '#29313D'
    figsize: tuple = (9, 5.6)
    dpi: int = 150
    max_points: int = 10_000
    max_groups: int = 8
    max_columns: int = 12
    max_auto_charts: int = 10
    max_matrix_columns: int = 6
    max_words: int = 100
    max_text_chars: int = 500_000
    bins: int = 30
    seed: int = 42

    def __post_init__(self):
        for key in ('dpi', 'max_points', 'max_groups', 'max_columns', 'max_auto_charts',
                    'max_matrix_columns', 'max_words', 'max_text_chars', 'bins'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if len(self.figsize) != 2 or any(x <= 0 for x in self.figsize):
            raise ValueError('figsize must have two positive dimensions')
        if not self.colors:
            raise ValueError('Supply at least one color')
        for color in (*self.colors, self.background, self.foreground, self.muted, self.grid):
            mpl.colors.to_rgba(color)


@dataclass
class ChartResult:
    kind: str
    figure: Figure
    metadata: dict
    request: dict = field(default_factory=dict)

    @property
    def axes(self):
        return self.figure.axes

    def save(self, path, **kwargs):
        path = Path(path)
        if path.suffix.lower() not in {'.png', '.svg', '.pdf'}:
            raise ValueError('Supported exports: PNG, SVG, PDF')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, facecolor=self.figure.get_facecolor(), bbox_inches='tight', **kwargs)
        return path

    def close(self):
        self.figure.clear()


class NoData(ValueError):
    """A valid request has insufficient observations; rendered as a no-data chart."""


CHARTS = {}


def register(name, description):
    def decorator(function):
        if name in CHARTS:
            raise ValueError(f'Duplicate chart: {name}')
        CHARTS[name] = (function, description)
        return function
    return decorator


def section(results, name):
    value = results.get(name)
    if value is None:
        return {}, {}
    return (value.data, getattr(value, 'metadata', {})) if hasattr(value, 'data') else (value, {})


def safe_stem(text):
    raw = str(text)
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', raw).strip('._')[:90] or 'chart'
    if clean != raw or clean.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        clean += '_' + hashlib.sha256(raw.encode()).hexdigest()[:8]
    return clean


class Visualizer:
    """Accept a Profiler (run once), results mapping, or a DataFrame.

    Explicit data= is treated as supplied data of unknown population coverage.
    Profiler samples retain their provenance. Never loads a dataset source.
    """
    def __init__(self, profiler_or_results, *, data=None, config=None):
        from . import charts  # noqa: F401: register built-in renderers
        self.config = config or VisualizerConfig()
        self.results = {}
        if isinstance(profiler_or_results, pd.DataFrame):
            if data is not None:
                raise ValueError('Do not provide data twice')
            frame = profiler_or_results
            meta = {'method': 'exact', 'population_rows': len(frame), 'source': 'supplied DataFrame'}
        else:
            self.results = profiler_or_results.run() if hasattr(profiler_or_results, 'run') else profiler_or_results
            if not isinstance(self.results, dict):
                raise TypeError('Expected Profiler, profile results dict, or DataFrame')
            interactions, imeta = section(self.results, 'interactions')
            summary, _ = section(self.results, 'summary')
            if data is not None:
                if not isinstance(data, pd.DataFrame):
                    raise TypeError('data must be a DataFrame')
                frame = data
                meta = {'method': 'supplied_data', 'population_rows': summary.get('rows'),
                        'source': 'explicit data; population coverage unverified'}
            else:
                frame = interactions.get('sample') if isinstance(interactions, dict) else None
                meta = {**imeta, 'source': 'profiler sample', 'method': imeta.get('method', 'unknown'),
                        'population_rows': summary.get('rows')}
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise TypeError('Profiler sample must be a DataFrame')
        self.meta = meta
        self.data = None
        if frame is not None:
            if not frame.columns.is_unique:
                raise ValueError('Duplicate column names are unsupported')
            if len(frame) > self.config.max_points:
                self.data = frame.sample(self.config.max_points, random_state=self.config.seed).copy()
                self.meta = {**meta, 'method': 'sampled', 'visualizer_downsampled': True}
            else:
                self.data = frame.copy(deep=True)
            self.data.reset_index(drop=True, inplace=True)
        self.meta['rows_available'] = 0 if self.data is None else len(self.data)
        self.columns, _ = section(self.results, 'columns')
        self._frame_roles = {}
        if isinstance(profiler_or_results, pd.DataFrame):
            from ..profiler.base import infer_role
            for name in frame:
                role = infer_role(frame[name])
                if role in {'numeric', 'text'} and 1 <= frame[name].nunique() <= 10:
                    role = 'category'
                self._frame_roles[name] = role


    @staticmethod
    def available():
        from . import charts  # noqa
        return {name: {'description': desc, 'options': [k for k in inspect.signature(fn).parameters if k not in {'v', 'ax'}]}
                for name, (fn, desc) in CHARTS.items()}

    def numeric_columns(self):
        if self.columns:
            return [c for c, p in self.columns.items() if p.get('type') == 'numeric']
        if self.data is None:
            return []
        if self._frame_roles:
            return [c for c, role in self._frame_roles.items() if role == 'numeric']
        return [c for c in self.data if pd.api.types.is_numeric_dtype(self.data[c])
                and not pd.api.types.is_bool_dtype(self.data[c])]

    def frame(self, columns, numeric=(), dropna=True):
        if self.data is None:
            raise NoData('Row sample unavailable. Keep profiler interactions enabled or supply data=.')
        columns = list(dict.fromkeys(columns))
        missing = set(columns) - set(self.data.columns)
        if missing:
            raise ValueError(f'Unknown columns: {sorted(missing, key=str)}')
        frame = self.data.loc[:, columns].copy()
        for c in numeric:
            if pd.api.types.is_complex_dtype(frame[c]):
                raise TypeError(f'{c} contains complex values')
            frame[c] = pd.to_numeric(frame[c], errors='raise').astype(float)
            frame.loc[~np.isfinite(frame[c]), c] = np.nan
        if dropna:
            frame.dropna(inplace=True)
        if frame.empty:
            raise NoData('No usable observations for this request')
        return frame

    def groups(self, x, group=None):
        frame = self.frame([x] + ([group] if group else []), numeric=[x])
        if group is None:
            return [(str(x), frame[x].to_numpy())], frame, {}
        counts = frame[group].value_counts()
        labels = list(counts.head(self.config.max_groups).index)
        groups = [(str(label), frame.loc[frame[group] == label, x].to_numpy()) for label in labels]
        shown = frame[frame[group].isin(labels)]
        return groups, shown, {'omitted_groups': max(0, len(counts)-len(labels)),
                               'omitted_observations': len(frame)-len(shown)}

    def color(self, i=0):
        return self.config.colors[i % len(self.config.colors)]

    def group_color(self, column, label):
        """Stable group identity across requests, independent of filtered counts."""
        labels = sorted(self.data[column].dropna().unique(), key=lambda x: (type(x).__name__,str(x)))
        index = next((i for i, value in enumerate(labels) if str(value) == str(label)), 0)
        return self.color(index)

    def cmap(self, diverging=False):
        colors = [self.color(2), self.config.background, self.color(1)] if diverging else [self.config.background, self.color()]
        return LinearSegmentedColormap.from_list('orchid', colors)

    def labels(self, ax, x=None, y=None):
        if x is not None:
            ax.set_xlabel(textwrap.fill(str(x), 65))
        if y is not None:
            ax.set_ylabel(textwrap.fill(str(y), 55))

    def info(self, frame=None, **extra):
        return {**self.meta, 'rows_used': len(frame) if frame is not None else self.meta['rows_available'], **extra}

    def plot(self, kind, *, title=None, xlabel=None, ylabel=None, **options):
        if kind not in CHARTS:
            raise ValueError(f'Unknown chart {kind!r}. Use Visualizer.available().')
        function, description = CHARTS[kind]
        inspect.signature(function).bind(self, None, **options)  # fail unknown/missing arguments early
        cfg = self.config
        style = {'figure.facecolor': cfg.background, 'axes.facecolor': cfg.background,
                 'text.color': cfg.foreground, 'axes.labelcolor': cfg.muted,
                 'xtick.color': cfg.muted, 'ytick.color': cfg.muted,
                 'axes.edgecolor': cfg.grid, 'grid.color': cfg.grid,
                 'font.family': 'DejaVu Sans', 'font.size': 10,
                 'axes.spines.top': False, 'axes.spines.right': False,
                 'text.usetex': False, 'text.parse_math': False, 'svg.fonttype': 'none'}
        with mpl.rc_context(style):
            fig = Figure(figsize=cfg.figsize, dpi=cfg.dpi, facecolor=cfg.background)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.grid(axis='y', alpha=.45, linewidth=.6)
            ax.set_axisbelow(True)
            try:
                metadata = function(self, ax, **options) or self.info()
                metadata.setdefault('status', 'ok')
            except NoData as exc:
                ax.set_axis_off()
                ax.text(.5, .5, textwrap.fill(str(exc), 65), ha='center', va='center', transform=ax.transAxes, color=cfg.muted)
                metadata = self.info(status='no_data', reason=str(exc))
            except Exception:
                fig.clear()
                raise
            for a in fig.axes:
                a.tick_params(labelsize=9)
                for label in a.get_xticklabels()+a.get_yticklabels():
                    label.set_parse_math(False)
            if xlabel is not None or ylabel is not None:
                self.labels(ax, xlabel, ylabel)
            name = title or kind.replace('_', ' ').title() + (' · ' + str(options['x']) if options.get('x') else '')
            wrapped_title = textwrap.fill(name, 75)
            subtitle_y = .96-.06*len(wrapped_title.splitlines())
            fig.text(.085, .97, wrapped_title, va='top', color=cfg.foreground, fontsize=16, fontweight='normal')
            method = metadata.get('method', 'unknown')
            count = metadata.get('rows_used')
            source = {'exact': 'FULL DATA', 'sampled': 'SAMPLED', 'supplied_data': 'SUPPLIED DATA',
                      'profile_aggregate': 'PROFILE AGGREGATES', 'mixed': 'MIXED METHODS'}.get(method, method.upper())
            subtitle = source + (f'  ·  n = {count:,}' if isinstance(count, (int, np.integer)) else '')
            if metadata.get('omitted_groups'):
                subtitle += f"  ·  {metadata['omitted_groups']} groups omitted"
            elif metadata.get('omitted_observations'):
                subtitle += f"  ·  {metadata['omitted_observations']:,} observations omitted"
            if metadata.get('limited_to_profiler_top_words'):
                subtitle += '  ·  profiler top words only'
            fig.text(.085, subtitle_y, subtitle, va='top', color=cfg.muted, fontsize=9)
            fig.subplots_adjust(left=.12, right=.95, top=max(.4,subtitle_y-.09), bottom=.17)
            fig.canvas.draw()
        metadata.update(kind=kind, palette='Orchid', description=description)
        return ChartResult(kind, fig, metadata, options)

    def summary(self, *, columns=None, output_dir=None):
        """Small automatic report; never generates all pairs or the full catalog."""
        requests = []
        summary, _ = section(self.results, 'summary')
        # Population aggregates take precedence over a potentially clean sample.
        missing = summary.get('percent_missing_cells')
        if missing is None:
            missing = any(p.get('pct_missing', 0) > 0 for p in self.columns.values()) if self.columns else (self.data is not None and self.data.isna().any().any())
        if missing:
            requests.append(('missing_bar', {}))
            if self.data is not None: requests.append(('missing_matrix', {}))
        corr, meta = section(self.results, 'correlations')
        if isinstance(corr, pd.DataFrame) and not corr.empty:
            requests.append(('association', {}))
        numeric = self.numeric_columns()
        if self.data is not None:
            pairs = [p for p in meta.get('pairs', []) if p.get('status') == 'ok'
                     and p.get('value') is not None and np.isfinite(p['value'])
                     and all(c in self.data for c in p['columns'])]
            pairs.sort(key=lambda p: -abs(p['value']))
            for pair in pairs[:3]:
                x, y = pair['columns']
                title = f"{x} vs {y} · {pair['method']} = {pair['value']:.3f}"
                if x in numeric and y in numeric:
                    requests.append(('scatter', {'x':x, 'y':y, 'title':title}))
                elif x not in numeric and y not in numeric:
                    requests.append(('category_heatmap', {'x':x, 'y':y, 'title':title}))
                else:
                    num, cat = (x,y) if x in numeric else (y,x)
                    requests.append(('box', {'x':num, 'group':cat, 'title':title}))
        names = list(columns) if columns is not None else list(self.columns or ({} if self.data is None else dict.fromkeys(self.data.columns)))
        for c in names:
            if c in numeric: requests.append(('histogram', {'x':c}))
            else:
                role = self.columns.get(c, {}).get('type', self._frame_roles.get(c))
                requests.append(('wordcloud' if role == 'text' else 'pie' if role == 'category' else 'bar', {'x':c}))
        charts = []
        for kind, kwargs in requests[:self.config.max_auto_charts]:
            chart = self.plot(kind, **kwargs)
            if output_dir:
                chart.save(Path(output_dir)/(safe_stem(kind+'_'+str(kwargs.get('x', 'overview'))+'_'+str(kwargs.get('y',kwargs.get('group',''))))+'.png'))
            charts.append(chart)
        return charts

    def column(self, name):
        """A bounded drill-down suitable for a future column-click handler."""
        if self.columns.get(name, {}).get('type') == 'datetime' or (
                self.data is not None and name in self.data and pd.api.types.is_datetime64_any_dtype(self.data[name])):
            return {'date_counts': self.plot('date_counts', x=name)}
        if name in self.numeric_columns():
            kinds = ['histogram', 'box', 'ecdf', 'raincloud']
        elif self.columns.get(name, {}).get('type', self._frame_roles.get(name)) == 'text':
            kinds = ['bar', 'text_length', 'word_frequency', 'wordcloud']
        else:
            kinds = ['pie', 'bar', 'lollipop', 'donut']
        return {kind: self.plot(kind, x=name) for kind in kinds}
