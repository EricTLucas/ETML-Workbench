"""Orchid EDA chart catalog; every public renderer returns provenance metadata."""
from collections import Counter
from statistics import NormalDist
import textwrap
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from .core import register, NoData, section


def _bins(v, bins):
    bins = v.config.bins if bins is None else bins
    if type(bins) is not int or not 2 <= bins <= 200:
        raise ValueError('bins must be an integer from 2 to 200')
    return bins


def _kde(values, bandwidth=None):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    h = bandwidth if bandwidth is not None else 1.06*np.std(values, ddof=1)*len(values)**(-.2)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('bandwidth must be positive and finite')
    grid = np.linspace(values.min()-3*h, values.max()+3*h, 256)
    density = np.zeros(len(grid))
    for start in range(0, len(values), 512):
        u = (grid[:, None]-values[None, start:start+512])/h
        density += np.exp(-.5*u*u).sum(axis=1)
    return grid, density/(len(values)*h*np.sqrt(2*np.pi))


def _legend(ax):
    ax.legend(frameon=False, fontsize=9, loc='best')


@register('histogram', 'Histogram, optionally with Gaussian KDE on the same density scale')
def histogram(v, ax, x, group=None, bins=None, kde=True, bandwidth=None):
    # Saved histograms work even when row-level samples were omitted.
    if v.data is None and group is None:
        p = v.columns.get(x, {})
        h = p.get('histogram', {})
        if not h.get('counts'):
            raise NoData('No stored histogram or row sample available')
        if bins is not None or bandwidth is not None:
            raise ValueError('Custom bins/bandwidth require row data')
        ax.stairs(h['counts'], h['edges'], fill=True, color=v.color(), alpha=.65)
        v.labels(ax, x, 'Observations in stored histogram')
        return {**h, 'rows_used': sum(h['counts']), 'kde_omitted': True}
    groups, frame, meta = v.groups(x, group)
    edges = np.histogram_bin_edges(frame[x], bins=_bins(v, bins))
    for i, (label, values) in enumerate(groups):
        color = v.group_color(group,label) if group else v.color(i)
        ax.hist(values, bins=edges, density=kde, color=color, alpha=.33 if kde or group else .8,
                label=label if group else None, rwidth=.94)
        if kde:
            smooth = _kde(values, bandwidth)
            if smooth:
                ax.plot(*smooth, color=color, lw=2)
    if group:
        _legend(ax)
    v.labels(ax, x, 'Density (per x unit)' if kde else 'Observations')
    return v.info(frame, **meta, kde=kde, bins=edges.tolist())


@register('density', 'Gaussian kernel density with soft fill')
def density(v, ax, x, group=None, bandwidth=None):
    groups, frame, meta = v.groups(x, group)
    for i, (label, values) in enumerate(groups):
        color = v.group_color(group,label) if group else v.color(i)
        smooth = _kde(values, bandwidth)
        if smooth:
            grid, d = smooth
            ax.fill_between(grid, d, color=color, alpha=.14)
            ax.plot(grid, d, color=color, lw=2, label=label)
        else:
            ax.axvline(values[0], color=color, label=f'{label}: constant / singleton')
    if group:
        _legend(ax)
    v.labels(ax, x, 'Density (per x unit)')
    return v.info(frame, **meta, estimator='Gaussian KDE; Silverman bandwidth unless overridden')


@register('ecdf', 'Empirical cumulative distribution; no smoothing')
def ecdf(v, ax, x, group=None):
    groups, frame, meta = v.groups(x, group)
    for i, (label, values) in enumerate(groups):
        values = np.sort(values)
        ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=v.group_color(group,label) if group else v.color(i), lw=2, label=label)
    ax.set_ylim(0, 1.02)
    if group:
        _legend(ax)
    v.labels(ax, x, 'Fraction at or below value')
    return v.info(frame, **meta)


def _distribution(v, ax, x, group, kind, bandwidth=None):
    groups, frame, meta = v.groups(x, group)
    rng = np.random.default_rng(v.config.seed)
    for i, (label, values) in enumerate(groups):
        color = v.group_color(group,label) if group else v.color(i)
        q1, med, q3 = np.quantile(values, [.25, .5, .75])
        if kind in {'violin', 'raincloud', 'ridgeline'}:
            smooth = _kde(values, bandwidth)
            if smooth:
                grid, dens = smooth
                height = dens/dens.max()*(.65 if kind == 'ridgeline' else .32)
                lower = i-height if kind == 'violin' else np.full(len(grid), i)
                ax.fill_between(grid, lower, i+height, color=color, alpha=.23)
                ax.plot(grid, i+height, color=color, lw=1.7)
                if kind == 'violin':
                    ax.plot(grid, lower, color=color, lw=1)
            else:
                ax.plot([med, med], [i-.12, i+.12], color=color, lw=2)
        if kind in {'box', 'violin', 'raincloud'}:
            offset = -.07 if kind == 'raincloud' else 0
            if kind == 'box':
                iqr = q3-q1
                inside = values[(values >= q1-1.5*iqr) & (values <= q3+1.5*iqr)]
                ax.plot([inside.min(), inside.max()], [i, i], color=color, lw=1)
                out = values[(values < q1-1.5*iqr) | (values > q3+1.5*iqr)]
                ax.scatter(out, np.full(len(out), i), color=color, s=9, alpha=.45)
            ax.plot([q1, q3], [i+offset]*2, color=color, lw=7, solid_capstyle='round')
            ax.scatter([med], [i+offset], s=18, color=v.config.foreground, zorder=5)
        if kind in {'raincloud', 'strip'}:
            shown = rng.choice(values, min(len(values), 400), replace=False)
            jitter = rng.uniform(-.25, -.13, len(shown)) if kind == 'raincloud' else rng.uniform(-.15, .15, len(shown))
            ax.scatter(shown, i+jitter, s=8, alpha=.45, color=color, rasterized=True)
    ax.set_yticks(range(len(groups)), [textwrap.fill(label, 22) for label, _ in groups])
    ax.set_ylim(-.5, len(groups)-.1 if kind == 'ridgeline' else len(groups)-.5)
    ax.grid(axis='y', visible=False)
    v.labels(ax, x, group or 'Distribution')
    return v.info(frame, **meta, glyph='median and IQR; box whiskers at extreme values within 1.5 IQR',
                  point_display_cap=400 if kind in {'raincloud', 'strip'} else None,
                  density_height='normalized per group for shape comparison')


@register('box', 'Horizontal box plots with 1.5-IQR whiskers and outliers')
def box(v, ax, x, group=None):
    return _distribution(v, ax, x, group, 'box')


@register('violin', 'Full violin densities with median and IQR')
def violin(v, ax, x, group=None, bandwidth=None):
    return _distribution(v, ax, x, group, 'violin', bandwidth)


@register('raincloud', 'Half violin + sampled observations + median/IQR (format B)')
def raincloud(v, ax, x, group=None, bandwidth=None):
    return _distribution(v, ax, x, group, 'raincloud', bandwidth)


@register('ridgeline', 'Stacked group density shapes (format D)')
def ridgeline(v, ax, x, group, bandwidth=None):
    return _distribution(v, ax, x, group, 'ridgeline', bandwidth)


@register('strip', 'Jittered observations by group')
def strip(v, ax, x, group=None):
    return _distribution(v, ax, x, group, 'strip')


@register('qq', 'Normal Q–Q diagnostic with quartile reference line')
def qq(v, ax, x):
    frame = v.frame([x], numeric=[x])
    values = np.sort(frame[x].to_numpy())
    theoretical = np.array([NormalDist().inv_cdf((i+.5)/len(values)) for i in range(len(values))])
    ax.scatter(theoretical, values, s=12, alpha=.5, color=v.color(), rasterized=True)
    q1, q3 = np.quantile(values, [.25, .75])
    z1, z3 = NormalDist().inv_cdf(.25), NormalDist().inv_cdf(.75)
    slope = (q3-q1)/(z3-z1)
    ends = np.array([theoretical.min(), theoretical.max()])
    ax.plot(ends, q1+slope*(ends-z1), color=v.color(1), lw=1.5)
    v.labels(ax, 'Theoretical standard normal quantile', x)
    return v.info(frame, reference='normal Q-Q; diagnostic, not a hypothesis test')


@register('outliers', 'Observations with 1.5-IQR outliers highlighted')
def outliers(v, ax, x):
    frame = v.frame([x], numeric=[x])
    values = frame[x].to_numpy()
    q1, q3 = np.quantile(values, [.25, .75]); iqr = q3-q1
    flag = (values < q1-1.5*iqr) | (values > q3+1.5*iqr)
    ax.scatter(np.flatnonzero(~flag), values[~flag], s=12, color=v.color(), alpha=.4)
    ax.scatter(np.flatnonzero(flag), values[flag], s=24, color=v.color(1), marker='x', label='IQR outlier')
    ax.axhspan(q1-1.5*iqr, q3+1.5*iqr, color=v.color(), alpha=.08)
    if flag.any():
        _legend(ax)
    v.labels(ax, 'Displayed observation position (not source row order)', x)
    return v.info(frame, outlier_count=int(flag.sum()), bounds=[q1-1.5*iqr, q3+1.5*iqr])


def _xy(v, x, y, group=None, size=None):
    cols = [x, y] + ([group] if group else []) + ([size] if size else [])
    frame = v.frame(cols, numeric=[x, y]+([size] if size else []))
    return frame


@register('scatter', 'Scatter with optional group colors and descriptive fitted line (format E)')
def scatter(v, ax, x, y, group=None, trend=True):
    frame = _xy(v, x, y, group)
    omitted = 0
    if group:
        counts = frame[group].value_counts()
        labels = counts.head(v.config.max_groups).index
        omitted = len(counts)-len(labels)
        markers = ['o', '^', 's', 'D', 'v', 'P']
        for i, label in enumerate(labels):
            subset = frame[frame[group] == label]
            ax.scatter(subset[x], subset[y], s=14, alpha=.45, color=v.group_color(group,label), marker=markers[i % len(markers)], label=str(label), rasterized=True)
        _legend(ax)
        frame = frame[frame[group].isin(labels)]
    else:
        ax.scatter(frame[x], frame[y], s=13, alpha=.3, color=v.color(), edgecolors='none', rasterized=True)
    fit = None
    if trend and len(frame) > 1 and frame[x].var() > 0:
        xv, yv = frame[x].to_numpy(), frame[y].to_numpy()
        mx, my = xv.mean(), yv.mean()
        slope = np.dot(xv-mx, yv-my)/np.dot(xv-mx, xv-mx)
        ends = np.array([xv.min(), xv.max()])
        ax.plot(ends, my+slope*(ends-mx), color=v.config.foreground, lw=1.5)
        fit = {'slope': float(slope), 'intercept': float(my-slope*mx)}
    v.labels(ax, x, y)
    return v.info(frame, omitted_groups=omitted, linear_fit=fit,
                  fit_description='pooled descriptive least-squares line; no confidence or causal claim')


@register('bubble', 'Scatter with marker area proportional to a nonnegative numeric column')
def bubble(v, ax, x, y, size):
    frame = _xy(v, x, y, size=size)
    if (frame[size] < 0).any():
        raise ValueError('Bubble sizes must be nonnegative')
    maximum = frame[size].max()
    area = frame[size]/maximum*250 if maximum else np.zeros(len(frame))
    ax.scatter(frame[x], frame[y], s=area, color=v.color(), alpha=.35, edgecolors=v.color(), linewidths=.4)
    v.labels(ax, x, y)
    ax.text(.02, .98, f'Marker area ∝ {size}', transform=ax.transAxes, va='top', color=v.config.muted, fontsize=9)
    return v.info(frame, size_column=size, max_size=float(maximum), zero_size='invisible')


@register('density2d', 'Rectangular 2D count bins (format F)')
def density2d(v, ax, x, y, bins=None):
    frame = _xy(v, x, y)
    h = ax.hist2d(frame[x], frame[y], bins=_bins(v, bins), cmap=v.cmap(), cmin=1)
    ax.figure.colorbar(h[3], ax=ax, label='Observations per bin')
    v.labels(ax, x, y)
    return v.info(frame, bin_counts=np.nan_to_num(h[0]).tolist())


@register('hexbin', 'Hexagonal density bins for large point clouds')
def hexbin(v, ax, x, y, bins=None):
    frame = _xy(v, x, y)
    image = ax.hexbin(frame[x], frame[y], gridsize=_bins(v, bins), mincnt=1, linewidths=0, cmap=v.cmap())
    ax.figure.colorbar(image, ax=ax, label='Observations per hexagon')
    v.labels(ax, x, y)
    return v.info(frame, binned_count=int(image.get_array().sum()))


@register('contour', 'Contours of a 2D binned count surface (not a KDE)')
def contour(v, ax, x, y, bins=None):
    frame = _xy(v, x, y)
    h, xe, ye = np.histogram2d(frame[x], frame[y], bins=_bins(v, bins))
    if np.ptp(h) == 0:
        raise NoData('No variation in the binned count surface')
    image = ax.contourf((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, h.T, levels=8, cmap=v.cmap())
    ax.figure.colorbar(image, ax=ax, label='Observations per bin')
    v.labels(ax, x, y)
    return v.info(frame, estimator='unsmoothed binned counts')


def _counts(v, x, top=None, other=False):
    top = v.config.max_groups if top is None else top
    if type(top) is not int or not 1 <= top <= 100:
        raise ValueError('top must be 1..100')
    column = v.columns.get(x, {})
    if column.get('methods', {}).get('frequencies', {}).get('method') == 'exact' and column.get('counts'):
        summary, _ = section(v.results, 'summary')
        total = summary.get('rows', 0) - column.get('num_missing', 0)
        items = sorted(column['counts'].items(), key=lambda item: item[1], reverse=True)[:top]
        labels = [str(label) for label, _ in items]
        values = np.array([count for _, count in items])
        omitted = max(0, total-int(values.sum()))
        if other and omitted:
            labels.append('[remaining categories]')
            values = np.append(values, omitted)
        return labels, values, {'method': 'exact', 'rows_used': total,
                                'source': 'full-data profile frequencies',
                                'total_nonmissing': total,
                                'omitted_observations': 0 if other else omitted}
    frame = v.frame([x])
    counts = frame[x].value_counts()
    displayed = counts.head(top)
    labels = [str(z) for z in displayed.index]
    values = displayed.to_numpy()
    omitted = int(counts.iloc[top:].sum())
    if other and omitted:
        labels.append('[remaining categories]'); values = np.append(values, omitted)
    return labels, values, v.info(frame, omitted_observations=omitted if not other else 0,
                                  total_nonmissing=int(counts.sum()))


def _category(v, ax, x, top, kind):
    labels, counts, meta = _counts(v, x, top, other=kind in {'pie', 'donut'})
    positions = np.arange(len(labels))
    wrapped = [textwrap.fill(s, 28) for s in labels]
    if kind in {'pie', 'donut'}:
        ax.grid(False)
        ax.pie(counts, labels=wrapped, colors=[v.color(i) for i in positions], autopct='%1.0f%%',
               startangle=90, wedgeprops={'width':.35} if kind == 'donut' else {},
               textprops={'fontsize':9, 'color':v.config.foreground})
        if kind == 'donut':
            ax.text(0, 0, f'n = {sum(counts):,}', ha='center', va='center', color=v.config.foreground)
    else:
        if kind == 'bar':
            ax.barh(positions, counts, color=v.color(), alpha=.8, height=.65)
        else:
            ax.hlines(positions, 0, counts, color=v.color(), alpha=.4, lw=2)
            ax.scatter(counts, positions, color=v.color(), s=40)
        ax.set_yticks(positions, wrapped)
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts)*1.18 or 1)
        for i, count in enumerate(counts):
            ax.annotate(f'{count:,}', (count, i), xytext=(6,0), textcoords='offset points', va='center', color=v.config.foreground, fontsize=9)
        v.labels(ax, 'Nonmissing observations', x)
    return meta


@register('bar', 'Horizontal category counts; top categories with omitted-count metadata')
def bar(v, ax, x, top=None):
    return _category(v, ax, x, top, 'bar')


@register('lollipop', 'Minimal category ranking using stems and dots')
def lollipop(v, ax, x, top=None):
    return _category(v, ax, x, top, 'lollipop')


@register('pie', 'Category composition with remaining categories included')
def pie(v, ax, x, top=None):
    return _category(v, ax, x, top, 'pie')


@register('donut', 'Category composition ring with remaining categories included')
def donut(v, ax, x, top=None):
    return _category(v, ax, x, top, 'donut')


@register('category_heatmap', 'Category-by-category contingency counts or row percentages')
def category_heatmap(v, ax, x, y, normalize=False):
    frame = v.frame([x,y])
    xs = frame[x].value_counts().head(v.config.max_groups).index
    ys = frame[y].value_counts().head(v.config.max_groups).index
    shown = frame[frame[x].isin(xs) & frame[y].isin(ys)]
    table = pd.crosstab(shown[x], shown[y]).reindex(index=xs, columns=ys, fill_value=0)
    if normalize:
        table = table.div(table.sum(axis=1).replace(0,np.nan), axis=0)*100
    im = ax.imshow(table.to_numpy(), aspect='auto', cmap=v.cmap(), vmin=0)
    ax.set_yticks(range(len(xs)), [str(z) for z in xs])
    ax.set_xticks(range(len(ys)), [str(z) for z in ys], rotation=35, ha='right')
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, label='Row % among displayed categories' if normalize else 'Observations')
    v.labels(ax, y, x)
    return v.info(shown, omitted_observations=len(frame)-len(shown), normalize=normalize)


@register('grouped_bar', 'Grouped category counts or within-group composition')
def grouped_bar(v, ax, x, group, normalize=False):
    frame = v.frame([x,group])
    xs = frame[x].value_counts().head(v.config.max_groups).index
    gs = frame[group].value_counts().head(v.config.max_groups).index
    shown = frame[frame[x].isin(xs) & frame[group].isin(gs)]
    table = pd.crosstab(shown[x], shown[group]).reindex(index=xs, columns=gs, fill_value=0)
    if normalize:
        table = table.div(table.sum(axis=0).replace(0,np.nan), axis=1)*100
    width = .8/max(1,len(gs))
    for i,g in enumerate(gs):
        ax.bar(np.arange(len(xs))+(i-(len(gs)-1)/2)*width, table[g], width=width*.95, color=v.group_color(group,g), label=str(g), alpha=.8)
    ax.set_xticks(range(len(xs)), [str(z) for z in xs], rotation=30, ha='right'); _legend(ax)
    v.labels(ax, x, '% of displayed group' if normalize else 'Observations')
    return v.info(shown, omitted_observations=len(frame)-len(shown), normalize=normalize)


@register('missing_bar', 'Full-data missingness by column when profile counts are available')
def missing_bar(v, ax, columns=None, percent=True):
    summary, _ = section(v.results, 'summary')
    if v.columns:
        counts = {c:p['num_missing'] for c,p in v.columns.items()}
        rows = summary.get('rows', 0)
        method = 'exact'
    elif v.data is not None:
        counts = v.data.isna().sum().to_dict(); rows = len(v.data); method = v.meta['method']
    else:
        raise NoData('Missingness counts unavailable')
    names = list(columns) if columns is not None else sorted(counts, key=counts.get, reverse=True)[:v.config.max_columns]
    if not names:
        raise NoData('No columns')
    if len(names)>v.config.max_columns:
        raise ValueError('Too many columns; increase max_columns')
    values = np.array([counts[c] for c in names], dtype=float)
    if percent:
        values = values/rows*100 if rows else values
    ax.barh(range(len(names)), values, color=v.color(), alpha=.8, height=.6)
    ax.set_yticks(range(len(names)), [textwrap.fill(str(z), 28) for z in names]); ax.invert_yaxis()
    ax.set_xlim(0,max(float(values.max())*1.22,1))
    for i,val in enumerate(values):
        ax.annotate(f'{val:.1f}%' if percent else f'{val:,.0f}', (val,i), xytext=(6,0), textcoords='offset points', va='center', fontsize=9)
    v.labels(ax, 'Missing values (%)' if percent else 'Missing observations', 'Feature')
    return {'method':method, 'rows_used':rows, 'omitted_columns':len(counts)-len(names)}


@register('missing_matrix', 'Raster missingness pattern from at most 1,000 displayed rows')
def missing_matrix(v, ax, columns=None):
    if v.data is None:
        raise NoData('A row sample is required for missingness patterns')
    names = list(columns) if columns is not None else list(v.data.columns[:v.config.max_columns])
    if len(names)>v.config.max_columns:
        raise ValueError('Too many columns; increase max_columns')
    frame = v.frame(names, dropna=False)
    if len(frame)>1000:
        frame = frame.sample(1000, random_state=v.config.seed)
    im = ax.imshow(frame.isna().to_numpy(), aspect='auto', interpolation='nearest',
                   cmap=ListedColormap([v.config.grid,v.color(1)]), vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), [str(z) for z in names], rotation=35, ha='right'); ax.grid(False)
    cb=ax.figure.colorbar(im, ax=ax, ticks=[0,1]); cb.ax.set_yticklabels(['Present','Missing'])
    v.labels(ax,'Feature','Displayed row position (sample order)')
    return v.info(frame, method='sampled' if len(frame)<v.meta['rows_available'] else v.meta['method'],
                  omitted_columns=len(v.data.columns)-len(names))


@register('missing_patterns', 'Most common missingness combinations on selected columns')
def missing_patterns(v, ax, columns=None):
    if v.data is None:
        raise NoData('A row sample is required')
    names = list(columns) if columns is not None else list(v.data.columns[:min(8,v.config.max_columns)])
    if len(names)>v.config.max_columns:
        raise ValueError('Too many columns')
    frame = v.frame(names, dropna=False)
    counts = Counter(tuple(row) for row in frame.isna().to_numpy()).most_common(v.config.max_groups)
    labels = [' + '.join(str(c) for c,flag in zip(names,row) if flag) or 'Complete' for row,n in counts]
    ax.barh(range(len(counts)), [n for row,n in counts], color=v.color(1), alpha=.7)
    ax.set_yticks(range(len(counts)), [textwrap.fill(s,35) for s in labels]); ax.invert_yaxis()
    v.labels(ax,'Observations','Missing fields')
    return v.info(frame, omitted_observations=len(frame)-sum(n for row,n in counts))


def _correlation_plot(v, ax, table, bubbles=False):
    if table.empty:
        raise NoData('No correlation data available')
    table = table.iloc[:v.config.max_columns,:v.config.max_columns]
    if bubbles:
        i,j = np.where(np.isfinite(table.to_numpy()))
        vals=table.to_numpy()[i,j]
        ax.scatter(j,i,s=np.abs(vals)*650,c=vals,cmap=v.cmap(True),vmin=-1,vmax=1,alpha=.75)
        for a,b,value in zip(i,j,vals):
            ax.text(b,a,f'{value:+.2f}',ha='center',va='center',fontsize=8)
        ax.set_xlim(-.5,len(table.columns)-.5);ax.set_ylim(len(table)-.5,-.5)
    else:
        image=ax.imshow(table.to_numpy(),vmin=-1,vmax=1,cmap=v.cmap(True),aspect='auto')
        ax.figure.colorbar(image,ax=ax,label='Association value')
        if len(table)<=8:
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    if pd.notna(table.iloc[i,j]):
                        ax.text(j,i,f'{table.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(len(table.columns)), [str(z) for z in table.columns], rotation=35,ha='right')
    ax.set_yticks(range(len(table)), [str(z) for z in table.index]);ax.grid(False)
    v.labels(ax,'Feature','Feature')


@register('correlation', 'Pearson or Spearman matrix recomputed on the available sample')
def correlation(v, ax, columns=None, method='pearson', bubbles=False):
    names = list(columns) if columns is not None else v.numeric_columns()[:v.config.max_columns]
    if not names:
        raise NoData('No numeric columns')
    if len(names)>v.config.max_columns:
        raise ValueError('Too many columns')
    frame=v.frame(names,numeric=names,dropna=False)
    if method not in {'pearson','spearman'}:
        raise ValueError('method must be pearson or spearman')
    # Pairwise ranks for Spearman: rank after pairwise missing removal.
    table=pd.DataFrame(np.nan,index=names,columns=names)
    pair_counts={}
    for a in names:
        for b in names:
            pair=frame.loc[:,list(dict.fromkeys([a,b]))].dropna()
            xs,ys=pair[a],pair[b]
            if method=='spearman': xs,ys=xs.rank(),ys.rank()
            table.loc[a,b]=xs.corr(ys) if len(pair)>1 and xs.nunique()>1 and ys.nunique()>1 else np.nan
            pair_counts[f'{a} / {b}']=len(pair)
    _correlation_plot(v,ax,table,bubbles)
    return v.info(frame,association_method=method,pair_counts=pair_counts,matrix=table)


@register('association', 'Stored profiler associations with mixed-method labeling')
def association(v, ax, columns=None, bubbles=False):
    table, meta=section(v.results,'correlations')
    if not isinstance(table,pd.DataFrame) or table.empty:
        raise NoData('Profiler association results unavailable')
    if columns is not None:
        if len(columns)>v.config.max_columns: raise ValueError('Too many columns')
        table=table.loc[columns,columns]
    _correlation_plot(v,ax,table,bubbles)
    ax.text(0,-.29,'Mixed associations: Pearson signed · V / eta unsigned · blank = uncomputed',transform=ax.transAxes,fontsize=8,color=v.config.muted)
    return {'method':'mixed','rows_used':None,'pair_methods':meta.get('pairs',[]),
            'omitted_columns':max(0,len(table)-v.config.max_columns)}


def _words(v,x):
    p=v.columns.get(x,{})
    if p.get('top words'):
        return Counter(p['top words']), {**p.get('methods',{}).get('words',{}), 'rows_used':None,
                                        'limited_to_profiler_top_words':True}
    frame=v.frame([x]);counts=Counter();remaining=v.config.max_text_chars;used=0
    for value in frame[x]:
        text=str(value)
        if len(text)>remaining:
            break  # never split tokens; truncate at row boundary
        counts.update(text.lower().split());remaining-=len(text);used+=1
    return counts,v.info(frame,rows_used=used,text_truncated=used<len(frame),
                         method='truncated_sample' if used<len(frame) else v.meta['method'])


@register('word_frequency', 'Ranked word frequencies from profile or bounded text sample')
def word_frequency(v, ax, x):
    counts,meta=_words(v,x)
    selected=counts.most_common(min(v.config.max_words,20))
    if not selected: raise NoData('No words available')
    labels,values=zip(*selected)
    ax.barh(range(len(labels)),values,color=v.color(),alpha=.75)
    ax.set_yticks(range(len(labels)),[textwrap.fill(str(z),25) for z in labels]);ax.invert_yaxis()
    v.labels(ax,'Token occurrences','Word')
    return meta


@register('wordcloud', 'Deterministic Orchid word cloud; requires optional wordcloud package')
def wordcloud(v, ax, x):
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError('Install wordcloud to generate word clouds: python -m pip install wordcloud') from exc
    counts,meta=_words(v,x)
    if not counts: raise NoData('No words available')
    cloud=WordCloud(width=1200,height=550,background_color=v.config.background,
                    max_words=v.config.max_words,random_state=v.config.seed,
                    prefer_horizontal=.95,relative_scaling=.5,
                    color_func=lambda word,**kwargs: v.color(sum(map(ord,word))%len(v.config.colors)))
    cloud.generate_from_frequencies(counts)
    ax.imshow(cloud.to_array(),interpolation='bilinear');ax.set_axis_off()
    return meta


@register('text_length', 'Distribution of characters per nonmissing row')
def text_length(v, ax, x, bins=None):
    frame=v.frame([x]);lengths=frame[x].astype(str).str.len()
    ax.hist(lengths,bins=_bins(v,bins),color=v.color(),alpha=.7,rwidth=.94)
    v.labels(ax,f'{x}: length (characters)','Observations')
    return v.info(frame)


@register('time_series', 'Date-sorted period aggregates from available observations')
def time_series(v, ax, x, y, frequency='D', agg='mean', area=False):
    if agg not in {'mean','median','sum','count','min','max'}:
        raise ValueError('Unsupported aggregation')
    frame=v.frame([x,y],numeric=[y])
    frame[x]=pd.to_datetime(frame[x],errors='raise')
    frame.dropna(subset=[x],inplace=True)
    if frame.empty: raise NoData('No valid dates')
    grouped=frame.set_index(x)[y].sort_index().resample(frequency)
    data=grouped.agg(agg)
    sizes=grouped.count()
    data.loc[sizes==0]=np.nan  # no artificial zero sums/counts for unobserved periods
    ax.plot(data.index,data.values,color=v.color(),lw=1.8,marker='o',markersize=3)
    if area: ax.fill_between(data.index,data.values,alpha=.15,color=v.color())
    ax.tick_params(axis='x',rotation=25)
    v.labels(ax,x,f'{agg}({y}) per {frequency} from available rows')
    return v.info(frame,aggregation=agg,frequency=frequency,population_estimate=False)


@register('scatter_matrix', 'Bounded pair plot with histograms on the diagonal')
def scatter_matrix(v, ax, columns=None):
    names=list(columns) if columns is not None else v.numeric_columns()[:v.config.max_matrix_columns]
    if not names: raise NoData('No numeric columns')
    if len(names)>v.config.max_matrix_columns: raise ValueError('Too many matrix columns')
    frame=v.frame(names,numeric=names,dropna=False)
    figure=ax.figure;figure.delaxes(ax);figure.set_size_inches(max(7,len(names)*2.2),max(6,len(names)*2.2))
    axes=figure.subplots(len(names),len(names),squeeze=False)
    for i,y in enumerate(names):
        for j,x in enumerate(names):
            a=axes[i,j]
            if i==j:a.hist(frame[x].dropna(),bins=20,color=v.color(),alpha=.6)
            else:a.scatter(frame[x],frame[y],s=4,alpha=.2,color=v.color(),rasterized=True)
            if i==len(names)-1:a.set_xlabel(str(x))
            if j==0:a.set_ylabel(str(y))
    return v.info(frame)


@register('parallel', 'Parallel coordinates; min-max scaled per feature on the sample')
def parallel(v, ax, columns=None):
    names=list(columns) if columns is not None else v.numeric_columns()[:v.config.max_matrix_columns]
    if len(names)<2: raise NoData('Choose at least two numeric columns')
    if len(names)>v.config.max_matrix_columns: raise ValueError('Too many columns')
    frame=v.frame(names,numeric=names)
    if len(frame)>500: frame=frame.sample(500,random_state=v.config.seed)
    span=frame.max()-frame.min();scaled=(frame-frame.min())/span.replace(0,1)
    ax.plot(range(len(names)),scaled.to_numpy().T,color=v.color(),alpha=.07,lw=.7)
    ax.plot(range(len(names)),scaled.median(),color=v.color(1),lw=2,label='Median of displayed rows')
    ax.set_xticks(range(len(names)),[str(z) for z in names],rotation=25,ha='right');_legend(ax)
    v.labels(ax,'Feature','Within-feature scaled value (0–1)')
    return v.info(frame,method='sampled' if len(frame)<v.meta['rows_available'] else v.meta['method'],
                  scaling='min-max of displayed rows; constant features at zero')


@register('joint', 'Scatter with marginal histograms on aligned axes')
def joint(v, ax, x, y, bins=None):
    frame=_xy(v,x,y)
    figure=ax.figure;figure.delaxes(ax)
    grid=figure.add_gridspec(4,4,hspace=.08,wspace=.08)
    main=figure.add_subplot(grid[1:,:3])
    top=figure.add_subplot(grid[0,:3],sharex=main)
    right=figure.add_subplot(grid[1:,3],sharey=main)
    main.scatter(frame[x],frame[y],s=12,color=v.color(),alpha=.3,rasterized=True)
    top.hist(frame[x],bins=_bins(v,bins),color=v.color(),alpha=.6,rwidth=.92)
    right.hist(frame[y],bins=_bins(v,bins),orientation='horizontal',color=v.color(2),alpha=.6,rwidth=.92)
    top.tick_params(labelbottom=False);right.tick_params(labelleft=False)
    top.set_ylabel('Count');right.set_xlabel('Count')
    v.labels(main,x,y)
    return v.info(frame)


@register('stacked_bar', 'Stacked category counts or composition with explicit category limits')
def stacked_bar(v, ax, x, group, normalize=True):
    frame=v.frame([x,group])
    xs=frame[x].value_counts().head(v.config.max_groups).index
    gs=frame[group].value_counts().head(v.config.max_groups).index
    shown=frame[frame[x].isin(xs)&frame[group].isin(gs)]
    table=pd.crosstab(shown[x],shown[group]).reindex(index=xs,columns=gs,fill_value=0)
    if normalize:table=table.div(table.sum(axis=1).replace(0,np.nan),axis=0)*100
    bottom=np.zeros(len(xs))
    for i,g in enumerate(gs):
        ax.bar(range(len(xs)),table[g],bottom=bottom,color=v.group_color(group,g),label=str(g),width=.65,alpha=.8)
        bottom+=table[g].to_numpy()
    ax.set_xticks(range(len(xs)),[str(z) for z in xs],rotation=30,ha='right');_legend(ax)
    v.labels(ax,x,'% among displayed categories' if normalize else 'Observations')
    return v.info(shown,omitted_observations=len(frame)-len(shown),normalize=normalize)


@register('line', 'Numeric-x line or area; repeated x values aggregated explicitly')
def line(v, ax, x, y, agg='mean', area=False):
    if agg not in {'mean','median','sum','min','max'}:raise ValueError('Unsupported aggregation')
    frame=_xy(v,x,y)
    data=frame.groupby(x,sort=True)[y].agg(agg)
    ax.plot(data.index,data.values,color=v.color(),lw=1.8,marker='o',markersize=3)
    if area:ax.fill_between(data.index,data.values,color=v.color(),alpha=.15)
    v.labels(ax,x,f'{agg}({y}) at each x')
    return v.info(frame,aggregation=agg,population_estimate=False)


@register('date_counts', 'Observation counts per calendar period')
def date_counts(v, ax, x, frequency='D'):
    frame=v.frame([x])
    dates=pd.to_datetime(frame[x],errors='raise').dropna()
    if dates.empty:raise NoData('No valid dates')
    counts=pd.Series(1,index=pd.DatetimeIndex(dates)).sort_index().resample(frequency).sum()
    ax.plot(counts.index,counts.values,color=v.color(),lw=1.8)
    ax.fill_between(counts.index,counts.values,color=v.color(),alpha=.18)
    ax.tick_params(axis='x',rotation=25)
    v.labels(ax,x,f'Available observations per {frequency}')
    return v.info(frame,frequency=frequency,population_estimate=False)
