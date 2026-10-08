"""Readable, offline HTML reports; charts never substitute for missing data."""
from html import escape
import json
import math

from app.services.evidence import diagnostic_groups, evidence_sections


CSS = """
:root{color-scheme:light;--ink:#172c42;--muted:#586a7c;--line:#dce3ec;--accent:#146c72}
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:var(--ink);font:16px/1.6 'Segoe UI',Arial,sans-serif}
main{max-width:1180px;margin:32px auto;padding:40px;background:white;border:1px solid var(--line);border-radius:16px}
h1{font-size:32px;line-height:1.2;margin:8px 0 16px;overflow-wrap:anywhere}h2{font-size:24px;line-height:1.3;margin:0 0 16px}
h3{font-size:19px;margin:24px 0 12px}h4{font-size:16px}p{margin:10px 0}.kicker{letter-spacing:2px;font-size:12px;font-weight:700;color:var(--accent)}
.muted,.caption{color:var(--muted)}.caption{font-size:14px}.meta{font-size:14px;overflow-wrap:anywhere}.notice{padding:18px 20px;background:#edf7f6;border-left:4px solid var(--accent);margin:24px 0}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{padding:18px;border:1px solid var(--line);border-radius:10px}.card strong{display:block;font-size:22px;margin:6px 0}.card span{font-size:14px;color:var(--muted)}
nav{display:flex;gap:10px 20px;flex-wrap:wrap;padding:16px 0;border-bottom:1px solid var(--line)}a{color:#126371;text-underline-offset:3px}section{padding-top:30px;margin-top:22px;border-top:1px solid var(--line);scroll-margin-top:15px}
.scroll{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}th{font-weight:650;background:#eff4f8;color:#233e58}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:nth-child(even){background:#fafbfd}td{overflow-wrap:anywhere}td.num,th.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.badge{font-size:11px;letter-spacing:.6px;font-weight:700;border:1px solid var(--line);padding:4px 8px;border-radius:6px;background:#f3f6fa;white-space:nowrap}.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.chart{border:1px solid var(--line);padding:16px;border-radius:10px;min-width:0}.chart h3{margin:0 0 8px}.chart svg{display:block;width:100%;height:auto}.chart-value{font-variant-numeric:tabular-nums}
details{border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}summary{cursor:pointer;font-weight:650}details details{background:#fafbfd}pre{background:#f4f6f9;border-radius:6px;padding:16px;font:12px/1.65 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto}code{font-size:12px;overflow-wrap:anywhere}li{margin:8px 0}.small{font-size:13px}.download-note{padding:12px 0;color:var(--muted)}
@media(max-width:800px){main{margin:0;padding:22px;border:0;border-radius:0}.cards,.charts{grid-template-columns:1fr}h1{font-size:28px}table{font-size:14px}}
@page{size:A4 landscape;margin:14mm}@media print{body{background:white;font-size:11pt}main{margin:0;padding:0;border:0;max-width:none}h1{font-size:24pt}h2{font-size:17pt}h3{font-size:13pt}table{font-size:10pt}nav,.download-note{display:none}section{margin-top:14px;padding-top:18px}.cards,.charts{grid-template-columns:repeat(3,minmax(0,1fr))}thead{display:table-header-group}tr,.card,.chart{break-inside:avoid}details{padding:8px}pre{max-height:none}a{color:inherit}}
"""

HEADERS = {
    'label': 'Profile / runtime', 'profile': 'Profile', 'accuracy_pct': 'Top-1 accuracy (%)',
    'correct_count': 'Correct', 'sample_count': 'Images', 'accuracy_delta_pp': 'Δ accuracy (pp)',
    'accuracy_delta_vs_fp32_pp': 'Δ vs FP32 (pp)', 'agreement_pct': 'Prediction agreement (%)',
    'latency_p50_ms': 'Median (ms)', 'latency_p95_ms': 'p95 (ms)', 'latency_stddev_ms': 'SD (ms)',
    'conversion_seconds': 'Conversion (s)', 'size_bytes': 'File size (MiB)',
    'output_mae': 'Output MAE', 'output_max_abs': 'Maximum error', 'tolerance_failure_count': 'Images outside tolerance',
    'candidate': 'Candidate', 'validation_accuracy_pct': 'Validation accuracy (%)',
    'test_accuracy_pct': 'Test accuracy (%)', 'validation_latency_ms': 'Validation median (ms)',
    'name': 'Boundary / operation', 'operation': 'Operator', 'status': 'Status', 'reason_code': 'Reason',
    'mae': 'MAE', 'max_abs': 'Maximum error', 'fault': 'Controlled test', 'expected': 'Expected outcome',
    'passed': 'Passed', 'first_observed_divergence': 'First observed difference',
    'accuracy_recovery_pp': 'Recovery (pp)', 'model_sha256': 'Model SHA-256', 'created_at': 'Recorded at',
    'latency_reduction_pct': 'Median latency reduction (%)', 'size_reduction_pct': 'File-size reduction (%)',
    'process_rss_sampled_peak_bytes': 'Sampled process RSS (MiB)', 'memory_evidence_status': 'Memory evidence',
    'arena_used_bytes': 'Arena used (KiB)', 'execution_origin': 'Measurement host', 'comparison_group': 'Dataset / input group',
    'published_size': 'Published file size', 'project_id': 'Project', 'job_id': 'Provider job',
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def txt(value):
    return escape('UNAVAILABLE' if value is None else str(value), quote=True)


def number(value, digits=3):
    return f'{value:,.{digits}f}' if finite(value) else 'UNAVAILABLE'


def cell(key, value):
    if value is None:
        return '<span class="muted">UNAVAILABLE</span>'
    if key in {'size_bytes', 'process_rss_sampled_peak_bytes'}:
        return number(value / 1048576, 3) + f'<br><span class="small muted">{number(value, 0)} bytes</span>' if finite(value) else txt(value)
    if key == 'arena_used_bytes' and finite(value):
        return number(value / 1024, 3) + f'<br><span class="small muted">{number(value, 0)} bytes</span>'
    if key == 'status':
        return txt({'pass':'Within tolerance','drift':'Needs review','unmapped':'Not compared','nonfinite':'Needs review'}.get(value, value))
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if finite(value):
        if key in {'mae','max_abs','output_mae','output_max_abs'} and 0 < abs(value) < 0.001:
            return f'{value:.3e}'
        return number(value, 0 if key in {'sample_count','correct_count','tolerance_failure_count'} else 3)
    if isinstance(value, (dict, list)):
        return txt(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if key in {'artifact_sha256', 'model_sha256', 'dataset_sha256'}:
        return '<code>' + txt(value) + '</code>'
    return txt(value)


def table(rows, keys, headers=None):
    if not rows:
        return '<p class="muted">No evidence recorded.</p>'
    headings = headers or HEADERS
    numeric = lambda k: ' class="num"' if any(finite(r.get(k)) for r in rows) else ''
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th' + numeric(k) + '>' + txt(headings.get(k, k.replace('_', ' ').title())) + '</th>' for k in keys
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<td' + numeric(k) + '>' + cell(k, row.get(k)) + '</td>' for k in keys
    ) + '</tr>' for row in rows) + '</tbody></table></div>'


def pretty(value):
    return '<pre>' + escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre>'


def details(title, body):
    return '<details><summary>' + txt(title) + '</summary>' + body + '</details>'


def document(title, body):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + txt(title) + '</title><style>' + CSS + '</style></head><body><main>' + body + '</main></body></html>'


def metric_charts(metrics, scope='Recorded host measurements'):
    """Numeric labels remain in the SVG; absent/invalid values get no bar."""
    if not metrics:
        return '<p class="muted">No measured metrics available for charts.</p>'
    charts = []
    specs = [('accuracy_pct', 'Top-1 accuracy', '%', 1, 100),
             ('latency_p50_ms', 'Median inference latency', 'ms', 1, None),
             ('size_bytes', 'Serialized model size', 'MiB', 1 / 1048576, None)]
    for key, title, unit, scale, ceiling in specs:
        usable = [r[key] * scale for r in metrics if finite(r.get(key)) and r[key] >= 0 and (key != 'accuracy_pct' or r[key] <= 100)]
        limit = ceiling or max(usable, default=1) or 1
        marks = []
        for index, row in enumerate(metrics):
            y = 16 + index * 62
            label = str(row.get('label') or row.get('profile') or 'Unnamed profile')
            short = label if len(label) <= 32 else label[:29] + '…'
            value = row.get(key)
            valid = finite(value) and value >= 0 and (key != 'accuracy_pct' or value <= 100)
            marks.append(f'<text x="0" y="{y}" font-size="13" fill="#334e67"><title>{txt(label)}</title>{txt(short)}</text>')
            if valid:
                width = max(0, min(185, value * scale / limit * 185))
                marks.append(f'<rect x="0" y="{y+9}" width="185" height="14" rx="3" fill="#edf1f6"/><rect x="0" y="{y+9}" width="{width:.3f}" height="14" rx="3" fill="#18797c"/>')
                marks.append(f'<text x="195" y="{y+21}" font-size="12" fill="#172c42">{number(value * scale, 3)} {unit}</text>')
            else:
                marks.append(f'<text x="0" y="{y+23}" font-size="12" fill="#68798b">UNAVAILABLE</text>')
        svg = f'<svg viewBox="0 0 310 {len(metrics)*62+2}" role="img" aria-label="{txt(title)}"><title>{txt(title)} · {txt(scope)}</title>' + ''.join(marks) + '</svg>'
        charts.append('<article class="chart"><h3>' + title + '</h3>' + svg + '</article>')
    return '<div class="charts">' + ''.join(charts) + '</div><p class="caption">' + txt(scope) + '. Accuracy axis: 0–100%. Latency and size use separate zero-based scales. See the numeric table for p95 and timing spread. Missing values have no bar.</p>'


def render_report(report):
    metrics = report.get('metrics', [])
    evidence = evidence_sections(report)
    selection = report.get('selection') or {}
    dataset = report.get('datasets', {}).get('test') or report.get('dataset', {})
    sample_count = dataset.get('image_count')
    resolution = report.get('accuracy_resolution_pp') or (100 / sample_count if finite(sample_count) and sample_count > 0 else None)
    measured = evidence['host']['status'] == 'MEASURED'
    host_scope = report.get('environment', {}).get('execution_origin') or report.get('environment', {}).get('platform') or 'Recorded host'
    body = '<header><p class="kicker">EDGELENS / EXPERIMENT REPORT</p><h1>' + txt(report.get('model', {}).get('name', 'Model experiment')) + '</h1><p class="meta">Run ' + txt(report.get('run_id')) + ' · ' + txt(report.get('created_at')) + '</p></header>'
    body += '<div class="notice"><strong>' + txt(report.get('summary', {}).get('conclusion', 'Review the recorded evidence below.')) + '</strong><p>Host: ' + evidence['host']['status'] + ' · Edge Impulse: ' + evidence['edge_impulse']['status'] + ' · ESP32: ' + evidence['esp32']['status'] + '</p></div>'
    body += '<div class="cards"><article class="card"><span>Held-out test images</span><strong>' + (number(sample_count, 0) if finite(sample_count) else 'UNAVAILABLE') + '</strong><span>True-label top-1 accuracy</span></article><article class="card"><span>Recorded profiles</span><strong>' + str(len(metrics)) + '</strong><span>' + txt(host_scope) + '</span></article><article class="card"><span>Accuracy resolution</span><strong>' + number(resolution, 3) + ' pp</strong><span>One changed prediction = 100 / N</span></article></div>'
    body += '<nav aria-label="Report sections"><a href="#metrics">Metrics & charts</a><a href="#selection">Conversion & selection</a><a href="#layers">Layer findings</a><a href="#edge">Edge evidence</a><a href="#reproduce">Reproduce</a></nav><p class="caption">MEASURED = execution on the recorded host; ESTIMATED = provider analysis; REFERENCE = an external published figure; UNAVAILABLE = no supported measurement. Accuracy differences use percentage points (pp), not relative percent.</p>'
    body += '<section id="metrics"><h2>1. Host results — ' + evidence['host']['status'] + '</h2><p>The main comparison uses labelled test images. File size, inference latency and conversion duration measure different things.</p>'
    body += table(metrics, ['label','accuracy_pct','correct_count','sample_count','latency_p50_ms','latency_p95_ms','size_bytes'])
    if measured:
        body += metric_charts(metrics, str(host_scope) + ' · each profile retains its own provenance')
    body += '<h3>Conversion fidelity and duration</h3>' + table(metrics, ['label','accuracy_delta_pp','agreement_pct','output_mae','output_max_abs','tolerance_failure_count','conversion_seconds'])
    body += '<p class="caption">Δ accuracy follows the metric’s declared reference; a positive value is not evidence of universal superiority. Agreement compares predictions with a reference, while accuracy compares them with true labels. Standalone ONNX/TFLite imports lack original PyTorch conversion-loss evidence.</p>'
    body += '<h3>Timing reliability</h3>' + table(metrics, ['label','latency_stddev_ms','latency_min_ms','latency_max_ms','timing_input_count','timing_evidence_status'])
    body += '<p class="caption">Median describes the centre; p95 describes slower invocations. Repeat independent runs before interpreting small differences. Serialized MiB are bytes / 1,048,576 and are not runtime RAM.</p>'
    body += details('Complete metric fields, raw latency samples and provenance', pretty(metrics)) + '</section>'

    body += '<section id="selection"><h2>2. Conversion and candidate selection</h2>'
    candidates = report.get('candidates', [])
    if candidates:
        rows = [{'candidate': c.get('name') or c.get('id'), 'status': c.get('status'),
                 'validation_accuracy_pct': (c.get('validation_metrics') or {}).get('accuracy_pct'),
                 'test_accuracy_pct': (c.get('test_metrics') or {}).get('accuracy_pct'),
                 'validation_latency_ms': (c.get('validation_metrics') or {}).get('latency_p50_ms'),
                 'eligible': (c.get('eligibility') or {}).get('eligible'),
                 'constraint_rejections': (c.get('eligibility') or {}).get('reasons'),
                 'failure': (c.get('failure') or {}).get('message')} for c in candidates]
        body += '<p>Calibration fits quantization ranges. Validation chooses configurations where the strategy supports selection. Held-out test results follow selection; validation-only candidates can have no test result.</p>' + table(rows, ['candidate','status','validation_accuracy_pct','test_accuracy_pct','eligible','constraint_rejections','failure'])
        body += details('Candidate settings, search, validation evidence and selection rationale', pretty({'selection':selection,'candidates':candidates,'search':report.get('search'),'datasets':report.get('datasets'),'diagnostics':report.get('diagnostics')}))
        if report.get('sensitivity'):
            body += '<h3>Controlled operation exclusions</h3>' + table(report['sensitivity'], ['node','operation','parent_candidate','status','accuracy_recovery_pp','output_mae_change','interpretation'])
    else:
        body += '<p>Selected: ' + txt(selection.get('selected')) + ' · ' + txt(selection.get('objective', 'No calibration-guided search in this run.')) + '</p>'
        body += details('Converter configurations and calibration rationale', pretty(selection))
    body += '</section><section id="layers"><h2>3. Layer and graph findings</h2><p>Within tolerance means a recorded numerical comparison passed. Needs review means observed drift/non-finite output. Not compared means no supported numerical comparison; it does not mean that layer is wrong. A first observed divergence is not proof of the cause.</p>'
    for group in diagnostic_groups(report):
        body += '<h3>Candidate: ' + txt(group['profile']) + '</h3><p>' + str(group['measured_count']) + ' numerical comparisons · ' + str(group['inventory_count']) + ' inventory entries · ' + str(group['unavailable_count']) + ' unavailable diagnostics.'
        if group['diagnostic_eligible']:
            body += ' Calibration operation coverage: ' + str(group['diagnostic_compared']) + ' / ' + str(group['diagnostic_eligible']) + ' eligible operations, on ' + str(group['diagnostic_sample_count']) + ' calibration images.'
        body += '</p>' + table(group['measured'], ['name','operation','status','mae','max_abs','sample_count','reason_code'])
        if group['unavailable']:
            body += '<h4>Unavailable or failed captures</h4>' + table(group['unavailable'], ['name','operation','status','reason_code','detail'])
        if group['inventory']:
            body += details('Operation inventory — ' + str(group['inventory_count']) + ' entries', '<p>Presence only; includes ' + str(group['quantization_helper_count']) + ' Q/DQ helpers.</p>' + table([{'operation':k,'count':v} for k,v in group['inventory_operations'].items()], ['operation','count']) + details('Full inventory entries', pretty(group['inventory'])))
        body += details('Full numerical comparison evidence', pretty(group['measured'] + group['unavailable']))
    for item in report.get('structural', []):
        body += '<h3>Structural graph: ' + txt(item.get('profile')) + '</h3><p>MATCHED / CHANGED / UNAVAILABLE are structural statuses, separate from numerical tolerance. Legal optimization can change a graph.</p>' + table((item.get('comparison') or {}).get('rows', []), ['name','operation','status','reason_code','changes']) + details('Graph shapes, connections and mapping evidence', pretty(item))
    faults = report.get('fault_validation') or {}
    if faults:
        body += '<h3>Controlled fault validation</h3>' + table(faults.get('cases', []), ['fault','expected','passed','first_observed_divergence','error']) + '<p>' + txt(faults.get('limitation')) + '</p>'
    body += details('Per-image predictions — ' + str(len(report.get('predictions', []))) + ' records', table(report.get('predictions', []), ['profile','image','label','prediction','reference_prediction','max_abs','within_tolerance'])) + '</section>'

    body += '<section id="edge"><h2>4. Edge evidence</h2><h3>ESP32 — ' + evidence['esp32']['status'] + '</h3><p>' + txt(evidence['esp32'].get('reason') or 'Physical reports retain their own device, artifact and input provenance.') + '</p>' + table(evidence['esp32']['records'], ['profile','source','latency_p50_ms','latency_p95_ms','arena_used_bytes','output_max_abs'])
    body += '<p class="caption">Arena usage is not total RAM. A one-image board output does not establish held-out dataset accuracy. Provider estimates and host timings do not become physical measurements.</p>' + details('Physical-device receipts', pretty(evidence['esp32']['records']))
    body += '<h3>Edge Impulse — ' + evidence['edge_impulse']['status'] + '</h3><p>' + txt(evidence['edge_impulse']['verification']) + '</p><p>TFLite evaluation links require the same uploaded artifact. ONNX is converted by the provider; the uploaded ONNX accuracy remains a host measurement and the provider-converted accuracy remains UNAVAILABLE.</p>' + table(evidence['edge_impulse']['records'], ['project_id','device','model_format','job_id','upload_evaluation_link','accuracy_link','created_at']) + details('Test estimate choice, provider response and model hashes', pretty({'choice':report.get('edge_estimate_request', {'enabled':False}), 'records':evidence['edge_impulse']['records']})) + '</section>'

    body += '<section id="reproduce"><h2>5. Reproduce and interpret</h2><p>Use the identical model/dataset hashes, labels, preprocessing, runtime versions and settings. Importing a Linux report preserves Linux timings; it does not re-measure them on Windows.</p>'
    body += details('Input contract, datasets, settings, CPU/runtime and model artifacts', pretty({k:report.get(k) for k in ['model','dataset','datasets','settings','environment','artifacts']}))
    body += '<h3>Memory measurement</h3><p>Sampled process RSS includes all loaded models, datasets and runtime state. It is neither model-only RAM nor a guaranteed true peak. A separate inference pass avoids adding the sampler to the recorded timing loop.</p>' + table(metrics, ['label','process_rss_sampled_peak_bytes','memory_evidence_status'])
    for key in ['methodology','limitations']:
        body += '<h3>' + key.title() + '</h3><ul>' + ''.join('<li>' + txt(x) + '</li>' for x in report.get(key, [])) + '</ul>'
    body += '</section><p class="download-note">Offline report · no external fonts, scripts or chart services. Open detailed panels before printing if you want them in the PDF. Use browser Print → Save as PDF, landscape, 100% scale.</p>'
    return document('EdgeLens — ' + str(report.get('model', {}).get('name', 'Experiment report')), body)
