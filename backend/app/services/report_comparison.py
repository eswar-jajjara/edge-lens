"""Combine saved host results with separately labelled published references."""
from datetime import datetime, timezone
import hashlib
import json
from urllib.parse import urlparse

from app.services.report_layout import document, txt, table, pretty, details, metric_charts, finite

CONTRACT = ('input_shape', 'layout', 'scale', 'mean', 'std', 'resize', 'resize_shorter', 'class_count')


def build_comparison(reports, references=None, title='EdgeLens conversion comparison'):
    if not isinstance(reports, list) or not 1 <= len(reports) <= 12:
        raise ValueError('Choose 1–12 exported EdgeLens report JSON files.')
    if not isinstance(title, str) or not title.strip() or len(title) > 180:
        raise ValueError('Use a comparison title of 1–180 characters.')
    references = [] if references is None else references
    if isinstance(references, dict):
        references = references.get('references')
    if not isinstance(references, list) or len(references) > 30:
        raise ValueError('Reference JSON must contain a references list with at most 30 entries.')
    rows, changes, experiments, seen = [], [], [], set()
    for report in reports:
        if not isinstance(report, dict) or report.get('source') != 'measured' or not report.get('run_id'):
            raise ValueError('Only measured EdgeLens exports with a run ID can enter this comparison. Illustrative demos are excluded.')
        if report['run_id'] in seen:
            raise ValueError('The same run was selected twice.')
        seen.add(report['run_id'])
        metrics = report.get('metrics')
        if not isinstance(metrics, list) or not metrics or len(metrics) > 40 or any(not isinstance(m, dict) for m in metrics):
            raise ValueError('Each selected run needs 1–40 host metric records.')
        dataset = report.get('datasets', {}).get('test') or report.get('dataset', {})
        model = report.get('model', {})
        environment = report.get('environment', {})
        contract = {k:model.get(k) for k in CONTRACT}
        comparison_identity = {'dataset_sha256':dataset.get('sha256'), 'preprocessed_sha256':dataset.get('preprocessed_sha256'), 'labels':dataset.get('labels'), 'contract':contract}
        group = hashlib.sha256(json.dumps(comparison_identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
        local_rows = []
        for metric in metrics:
            artifact = next((a for a in report.get('artifacts', []) if a.get('sha256') and a.get('sha256') == metric.get('artifact_sha256')), {})
            identity_status = 'HASH LINK RECORDED' if artifact else 'UNAVAILABLE'
            if not artifact and not metric.get('artifact_sha256'):
                profile_artifacts = [a for a in report.get('artifacts', [])
                                     if metric.get('profile') and a.get('profile') == metric['profile']
                                     and a.get('format') in {'pt2', 'pytorch_state_dict', 'onnx', 'tflite'}]
                if len(profile_artifacts) == 1:
                    artifact = profile_artifacts[0]
                    identity_status = 'UNAVAILABLE — framework from unique saved profile; metric hash link missing'
            framework = {'pt2':'PyTorch','pytorch_state_dict':'PyTorch','onnx':'ONNX Runtime','tflite':'LiteRT / TFLite'}.get(artifact.get('format'), 'Unspecified runtime')
            row = {k:metric.get(k) for k in ('accuracy_pct','sample_count','correct_count','latency_p50_ms','latency_p95_ms','latency_stddev_ms','size_bytes','conversion_seconds','output_mae','output_max_abs','artifact_sha256','dataset_sha256','evaluation_split')}
            row.update(run_id=report['run_id'], label=metric.get('label') or metric.get('profile'), profile=metric.get('profile'), framework=framework,
                       model=model.get('name'), dataset=dataset.get('name'), comparison_group=group,
                       artifact_identity_status=identity_status,
                       execution_origin=environment.get('execution_origin') or environment.get('platform') or 'UNAVAILABLE', evidence_status='MEASURED')
            local_rows.append(row)
        baseline = next((r for r in local_rows if r.get('profile') == 'standard'), None)
        if baseline is None:
            baseline = next((r for r in local_rows if r.get('profile') == 'fp32'), None)
        if baseline:
            for candidate in local_rows:
                if candidate is baseline:
                    continue
                same_test = bool(baseline.get('dataset_sha256')) and baseline.get('dataset_sha256') == candidate.get('dataset_sha256') and baseline.get('sample_count') == candidate.get('sample_count') and baseline.get('evaluation_split') == candidate.get('evaluation_split') == 'test'
                delta = candidate['accuracy_pct'] - baseline['accuracy_pct'] if same_test and finite(candidate.get('accuracy_pct')) and finite(baseline.get('accuracy_pct')) else None
                def reduction(key):
                    base, value = baseline.get(key), candidate.get(key)
                    return (base-value)/base*100 if finite(base) and finite(value) and base > 0 and value >= 0 else None
                changes.append({'run_id':report['run_id'], 'baseline':baseline['label'], 'candidate':candidate['label'],
                                'accuracy_delta_pp':delta, 'latency_reduction_pct':reduction('latency_p50_ms'),
                                'size_reduction_pct':reduction('size_bytes'),
                                'interpretation':'Observed same-run difference; positive reduction means smaller/faster. Accuracy needs matching held-out identity. Timing differences need independent repetitions; no significance or universal superiority is established.'})
        rows.extend(local_rows)
        experiments.append({'run_id':report['run_id'],'created_at':report.get('created_at'),'model':model,'dataset':dataset,
                            'datasets':report.get('datasets'),'settings':report.get('settings'),'environment':environment,
                            'artifacts':report.get('artifacts'),
                            'summary':report.get('summary'),'rows':local_rows,
                            'limitations':report.get('limitations', []),'selection':report.get('selection')})
    clean_refs = []
    for ref in references:
        if not isinstance(ref, dict):
            raise ValueError('Each published reference must be an object.')
        url = ref.get('source_url', '')
        if not isinstance(url, str) or len(url) > 2000:
            raise ValueError('Every reference needs a valid HTTPS source URL.')
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Use HTTPS source URLs without embedded credentials.')
        for key in ('accuracy_pct','latency_p50_ms','size_bytes'):
            value = ref.get(key)
            if value is not None and (not finite(value) or value < 0 or (key == 'accuracy_pct' and value > 100)):
                raise ValueError('Reference metrics must be finite, non-negative numbers; accuracy is 0–100.')
        if not ref.get('framework') or not ref.get('model'):
            raise ValueError('Each reference needs its framework and exact model/checkpoint name.')
        allowed = ('framework','model','weights','dataset','preprocessing','hardware','latency_method','accuracy_pct','latency_p50_ms','size_bytes','published_size','source_url','accessed_at','notes')
        clean_refs.append({**{k:ref.get(k) for k in allowed}, 'evidence_status':'REFERENCE',
                           'comparison_status':'CONTEXT ONLY — published conditions are not verified identical to these local runs.'})
    return {'schema_version':1, 'kind':'edgelens_cross_framework_comparison','title':title.strip(),
            'created_at':datetime.now(timezone.utc).isoformat(), 'measured_rows':rows, 'same_run_changes':changes,
            'references':clean_refs, 'experiments':experiments,
            'conclusion':'Compare observed trade-offs under matching conditions. Published figures are contextual references; this report does not establish universal converter superiority.',
            'physical_device_status':'UNAVAILABLE unless supplied separately in an original measured hardware report.'}


def comparison_html(comparison):
    body = '<p class="kicker">EDGELENS / FRAMEWORK COMPARISON</p><h1>' + txt(comparison['title']) + '</h1><p class="meta">Prepared ' + txt(comparison['created_at']) + '</p>'
    body += '<div class="notice"><strong>' + txt(comparison['conclusion']) + '</strong><p>MEASURED host results, REFERENCE internet figures and ESTIMATED edge profiles are different evidence classes.</p></div>'
    body += '<h2>1. Your measured framework results</h2>' + table(comparison['measured_rows'], ['framework','label','accuracy_pct','sample_count','latency_p50_ms','latency_p95_ms','size_bytes','execution_origin','comparison_group'])
    body += '<p class="caption">Rows with different dataset/preprocessing groups need separate accuracy interpretation. Rows from different CPUs, operating systems, runtime versions or thread settings must not establish a latency advantage. Model size includes the saved file’s serialization format. Missing values stay UNAVAILABLE.</p>'
    body += '<p class="caption">Older exports can name a runtime through a unique saved artifact profile while lacking explicit metric-to-artifact or metric-to-dataset hash links. This does not establish a verified evaluation link. Inspect artifact_identity_status in JSON/provenance and rerun those models for complete evidence.</p>'
    body += '<h2>2. Observed changes within each run</h2>' + table(comparison['same_run_changes'], ['baseline','candidate','accuracy_delta_pp','latency_reduction_pct','size_reduction_pct'])
    body += '<p class="caption">The baseline is the standard profile, or the FP32 control in precision experiments. Positive accuracy delta is percentage points. Positive latency/size reduction means lower median/bytes. A faster ONNX runtime compared with PyTorch is a runtime result; it does not by itself prove that EdgeLens’s conversion strategy improved the standard converter.</p>'
    for experiment in comparison['experiments']:
        body += '<h3>Charts for run ' + txt(experiment['run_id']) + '</h3>' + metric_charts(experiment['rows'], 'This saved run only; check its measurement origin and settings')
    body += '<section><h2>3. Published internet references — REFERENCE</h2>' + table(comparison['references'], ['framework','model','accuracy_pct','latency_p50_ms','size_bytes','published_size','dataset','hardware'])
    body += '<p class="caption">No published-reference advantage is calculated automatically. A creator’s complete ImageNet validation score is not the same experiment as a 500-image ImageNet-V2 subset. Rounded published MB, checkpoint bytes, ONNX bytes and TFLite bytes are separate quantities.</p>'
    for ref in comparison['references']:
        body += '<h3>' + txt(ref['framework']) + ' · ' + txt(ref['model']) + '</h3><p><a href="' + txt(ref['source_url']) + '" rel="noopener noreferrer">Published source</a> · Checked ' + txt(ref.get('accessed_at')) + '</p>' + pretty(ref)
    body += '</section><section><h2>4. Conclusions to write after testing</h2><ol><li>State the exact baseline, model hash, dataset hash and precision.</li><li>Report true-label accuracy and its delta, median/p95 latency with repeatability, and actual serialized bytes.</li><li>Describe the trade-off: equal, better or worse results are all valid. Name constraints that led the tool to retain FP32 or reject a candidate.</li><li>Support diagnostic/reporting benefits with saved fault checks and actual findings. Do not infer that another tool lacks a feature without checking its documentation.</li><li>Keep host MEASURED, provider ESTIMATED, published REFERENCE and missing UNAVAILABLE evidence separate.</li></ol></section>'
    body += '<h2>5. Reproducibility</h2>' + ''.join(details('Run ' + str(x['run_id']) + ' — settings, input contract and provenance', pretty(x)) for x in comparison['experiments'])
    body += '<p class="download-note">Offline comparison. Open provenance panels before printing if required. Print → Save as PDF → landscape.</p>'
    return document(comparison['title'], body)


def comparison_csv(comparison):
    import csv
    import io
    from app.services.reports import _cell
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    for title, rows in [('MEASURED host results',comparison['measured_rows']),('Observed same-run changes',comparison['same_run_changes']),('REFERENCE published figures — context only',comparison['references'])]:
        writer.writerow([title])
        keys = list(rows[0]) if rows else []
        writer.writerow(keys)
        for row in rows:
            writer.writerow([_cell(row.get(k)) for k in keys])
        writer.writerow([])
    return stream.getvalue()
