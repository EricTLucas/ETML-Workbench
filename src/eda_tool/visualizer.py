"""Public visualizer API and adapters for the original function names.

New API: Visualizer(profiler_or_results).plot('raincloud', x='income').
Legacy functions return Matplotlib Figures (not ChartResult) and optionally save
PNGs. They never call plt.show() or launch a browser. The old HTML renderer still
needs adaptation to the new profiler's sampled/nullable fields.
"""
from pathlib import Path
from .visualization import Visualizer, VisualizerConfig, ChartResult, register
from .visualization.core import safe_stem, section


def _viz(df, profile, config=None):
    # Prefer the profiler sample to preserve provenance, not an arbitrary df.
    interactions, _ = section(profile, 'interactions')
    has_sample = isinstance(interactions, dict) and interactions.get('sample') is not None
    return Visualizer(profile, data=None if has_sample else df, config=config)


def _finish(chart, output_dir, name):
    if output_dir:
        chart.save(Path(output_dir)/(safe_stem(name)+'.png'))
    return chart.figure


def visualize_dataset(df=None, profile=None, output_dir=None, *, config=None):
    """Bounded summary. Supports old (df, results, output_dir) and new (Profiler)."""
    if profile is not None:
        v = _viz(df, profile, config=config)
    else:
        v = Visualizer(df, config=config)
    figures = []
    for chart in v.summary():
        x = chart.request.get('x', '')
        kind = chart.kind
        name = {'missing_bar':'missing_barchart', 'missing_matrix':'missing_matrix_bars',
                'association':'correlation_heatmap'}.get(kind)
        if name is None:
            if kind=='histogram': name=f'hist_{x}'
            elif kind=='bar': name=f'cat_{x}'
            elif kind=='wordcloud': name=f'wordcloud_{x}'
            elif kind=='scatter': name=f"scatter_{x}_vs_{chart.request['y']}"
            else: name=f'{kind}_{x}'
        figures.append(_finish(chart, output_dir, name))
    return figures


def plotNumericColumn(df, profile, col, output_dir=None):
    return _finish(_viz(df,profile).plot('histogram',x=col,kde=False),output_dir,f'hist_{col}')


def plotCategoricalColumn(df, profile, col, output_dir=None):
    return _finish(_viz(df,profile).plot('bar',x=col),output_dir,f'cat_{col}')


def plotTextColumn(df, profile, col, output_dir=None):
    return _finish(_viz(df,profile).plot('wordcloud',x=col),output_dir,f'wordcloud_{col}')


def plotHeatmap(df, profile, correlations=None, output_dir=None):
    if correlations is not None:
        profile = {**profile, 'correlations':correlations}
    return _finish(_viz(df,profile).plot('association'),output_dir,'correlation_heatmap')


def plotInteractions(df, profile, pairs, sample=None, output_dir=None):
    """Explicit pairs, capped at 100; old per-pair samples are accepted."""
    pairs=list(pairs)
    if len(pairs)>100:
        raise ValueError('Request at most 100 plots at once')
    v=_viz(df,profile)
    figures=[]
    for a,b in pairs:
        if a==b:
            continue
        chosen=sample.get((a,b)) if isinstance(sample,dict) else None
        if chosen is None and isinstance(sample,dict):
            chosen=sample.get(f'{a}|{b}')
        renderer=Visualizer(profile,data=chosen) if chosen is not None else v
        figures.append(_finish(renderer.plot('scatter',x=a,y=b,trend=False),output_dir,f'scatter_{a}_vs_{b}'))
    return figures


def plotMissingMatrix(df, columns, n_rows, missing_dict, profile, output_dir=None):
    return _finish(_viz(df,profile).plot('missing_matrix'),output_dir,'missing_matrix_bars')


def plotMissingBarChart(columns, num_rows, missing_dict, output_dir=None):
    """Preserves the old function's nonmissing-count bar semantics."""
    profile={'columns':{c:{'num_missing':missing_dict.get(c,0)} for c in columns},
             'summary':{'rows':num_rows}}
    v=Visualizer(profile)
    result=v.plot('missing_bar',percent=False)
    ax=result.axes[0]
    names=sorted(columns,key=lambda c:missing_dict.get(c,0),reverse=True)[:v.config.max_columns]
    counts=[num_rows-missing_dict.get(c,0) for c in names]
    for patch, count in zip(ax.patches, counts):
        patch.set_width(count)
    for text in list(ax.texts):
        text.remove()
    ax.set_xlim(0,max(counts,default=1)*1.1 or 1)
    ax.set_xlabel('Nonmissing observations');ax.set_ylabel('Feature')
    result.figure.texts[0].set_text('Nonmissing values')
    return _finish(result,output_dir,'missing_barchart')


def plotMissingValues(df, profile, warnings=None, output_dir=None):
    v=_viz(df,profile)
    return (_finish(v.plot('missing_bar'),output_dir,'missing_barchart'),
            _finish(v.plot('missing_matrix'),output_dir,'missing_matrix_bars'))


__all__ = ['Visualizer','VisualizerConfig','ChartResult','register','visualize_dataset',
           'plotNumericColumn','plotCategoricalColumn','plotTextColumn','plotHeatmap',
           'plotInteractions','plotMissingMatrix','plotMissingBarChart','plotMissingValues']
