"""Offline, self-contained HTML gallery. Escaped text; embedded PNGs; no scripts."""
from pathlib import Path
from html import escape
from io import BytesIO
import base64
import math
import numbers
from ..artifacts.runs import SavedChart


def _text(value):
    return escape(str(value), quote=True)


def _format(value, percent=False):
    if value is None: return 'Unavailable'
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)): return 'Unavailable'
        if percent: return f'{float(value):.1%}'
        if isinstance(value, numbers.Integral): return f'{int(value):,}'
        return f'{float(value):,.4g}'
    return _text(value)


def _section(profile, key):
    item = profile.get(key)
    return {} if item is None else item.data


class HtmlReport:
    def __init__(self, profile, charts=(), *, title='Exploratory data analysis', max_columns=50, max_warnings=30):
        self.profile = profile
        self.charts = list(charts)
        self.title = title
        if max_columns < 1 or max_warnings < 1: raise ValueError('Report limits must be positive')
        if len(self.charts)>100: raise ValueError('A gallery supports at most 100 charts')
        self.max_columns, self.max_warnings = max_columns, max_warnings

    def _chart(self, chart, index):
        if isinstance(chart, SavedChart):
            image = chart.path.read_bytes()
            if not image.startswith(b'\x89PNG\r\n\x1a\n'): raise ValueError('Saved gallery images must be PNG files')
            title = chart.title
        else:
            stream = BytesIO()
            chart.figure.savefig(stream, format='png', facecolor=chart.figure.get_facecolor(), bbox_inches='tight')
            image = stream.getvalue()
            title = chart.figure.texts[0].get_text() if chart.figure.texts else chart.kind
        encoded = base64.b64encode(image).decode('ascii')
        return f'''<figure id="chart-{index}">
<img src="data:image/png;base64,{encoded}" alt="{_text(title)}" loading="lazy">
</figure>'''

    def render(self):
        summary = _section(self.profile, 'summary')
        columns = _section(self.profile, 'columns')
        warnings = _section(self.profile, 'warnings')
        rows = []
        for name, p in list(columns.items())[:self.max_columns]:
            unique = _format(p.get('num_unique'))
            if p.get('num_unique') is None and p.get('unique_lower_bound') is not None:
                unique = f"At least {_format(p['unique_lower_bound'])}"
            cells = [_text(name), _text(p.get('type','unknown')), _format(p.get('pct_missing'),True),
                     unique, _format(p.get('mean')), _format(p.get('std'))]
            rows.append('<tr>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
        table = ''.join(rows) or '<tr><td colspan="6">No columns available</td></tr>'
        notices = [f'{col}: {message}' for col, entries in warnings.items() for message in entries.values()]
        warning_html = ''.join(f'<li>{_text(message)}</li>' for message in notices[:self.max_warnings])
        omitted = max(0,len(notices)-self.max_warnings)
        if omitted: warning_html += f'<li>{omitted} additional findings omitted.</li>'
        findings = f'<section><h2>Data quality findings</h2><ul>{warning_html}</ul></section>' if notices else ''
        gallery = ''.join(self._chart(chart,i) for i,chart in enumerate(self.charts,1))
        if not gallery: gallery = '<p class="caption">No charts included. Generate summary charts or add individual chart requests.</p>'
        duplicate = _format(summary.get('num_duplicates'))
        column_note = f'<p class="caption">Showing {len(rows)} of {len(columns)} columns.</p>' if len(columns)>len(rows) else ''
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'">
<title>{_text(self.title)}</title><style>
:root{{color-scheme:dark;--bg:#080B10;--surface:#10151F;--text:#E6EAF1;--muted:#A4ADBC;--line:#29313D;--accent:#C0A1FF}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:36px 24px}}h1{{font-size:clamp(24px,4vw,36px);line-height:1.2;font-weight:500;overflow-wrap:anywhere;margin:8px 0 12px}}
h2{{font-size:21px;font-weight:500;margin:0 0 14px}}h3{{font-size:17px;font-weight:500;margin:0;overflow-wrap:anywhere}}
.eyebrow{{color:var(--accent);font-size:12px;letter-spacing:.15em}}.caption{{color:var(--muted);font-size:13px;margin:6px 0 12px;overflow-wrap:anywhere}}
section{{margin:32px 0}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin:24px 0}}
.stat{{background:var(--surface);padding:16px;border-radius:10px}}.stat dt{{color:var(--muted);font-size:13px}}.stat dd{{margin:8px 0 0;font-size:25px;overflow-wrap:anywhere}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);overflow-wrap:anywhere;max-width:300px}}th{{font-weight:500;color:var(--muted)}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}figure{{margin:0;min-width:0;background:var(--surface);border-radius:12px;overflow:hidden}}img{{display:block;width:100%;height:auto}}li{{margin-bottom:6px;overflow-wrap:anywhere}}
footer{{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px}}@media(max-width:760px){{.gallery{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
@media print{{figure{{break-inside:avoid}}main{{padding:0}}.gallery{{display:block}}figure{{margin-bottom:20px}}}}
</style></head><body><main><header><div class="eyebrow">ETML WORKBENCH EDA</div><h1>{_text(self.title)}</h1>
<p class="caption">ETML report · full-data aggregates and sampled views are labeled separately.</p></header>
<dl class="stats"><div class="stat"><dt>Rows</dt><dd>{_format(summary.get('rows'))}</dd></div>
<div class="stat"><dt>Columns</dt><dd>{_format(summary.get('cols'))}</dd></div>
<div class="stat"><dt>Missing cells</dt><dd>{_format(summary.get('percent_missing_cells'),True)}</dd></div>
<div class="stat"><dt>Duplicate rows</dt><dd>{duplicate}</dd></div></dl>
<section><h2>Column overview</h2><p class="caption">Means and standard deviations use finite numeric values. Unavailable distinct counts are not guessed.</p>
<div class="table-wrap"><table><thead><tr><th>Column</th><th>Role</th><th>Missing</th><th>Distinct</th><th>Mean</th><th>Std. deviation</th></tr></thead><tbody>{table}</tbody></table></div>{column_note}</section>
{findings}<section><h2>Visualizations</h2><div class="gallery">{gallery}</div></section>
<footer>Self-contained export · no network resources or scripts · sampled counts are not population estimates.<br>
This report may contain dataset-derived values, words and plotted observations. It is not anonymized.</footer>
</main></body></html>'''

    def save(self, path, *, overwrite=False):
        path = Path(path)
        if path.suffix.lower() not in {'.html','.htm'}: raise ValueError('Report path must end in .html or .htm')
        content = self.render()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w' if overwrite else 'x', encoding='utf-8') as stream: stream.write(content)
        return path


def build_html_report(profile, output_dir=None, path='', *, charts=None, title=None):
    """Legacy string-returning adapter; optionally embed existing PNGs from a folder."""
    if charts is None:
        charts = []
        if output_dir is not None:
            directory = Path(output_dir).resolve()
            for image in sorted(directory.glob('*.png'))[:100]:
                if not image.resolve().is_relative_to(directory): raise ValueError('Image leaves report directory')
                charts.append(SavedChart('legacy',image,image.stem,{'method':'unknown'}))
    label = title or (Path(path).stem if isinstance(path,(str,Path)) and str(path) else 'Exploratory data analysis')
    return HtmlReport(profile,charts,title=label).render()
