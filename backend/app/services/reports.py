"""Self-contained developer reports, with separate host/device/provider evidence."""
import csv
from html import escape
import io
import json

METRICS = ["label", "accuracy_pct", "correct_count", "sample_count", "accuracy_delta_pp", "accuracy_delta_vs_fp32_pp", "reference_profile", "agreement_pct", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "conversion_seconds", "size_bytes", "output_mae", "output_max_abs", "tolerance_failure_count"]
LAYERS = ["profile", "name", "operation", "status", "mae", "max_abs", "sample_index", "expected_shape", "actual_shape", "max_error_index", "expected_value", "actual_value", "detail"]
PREDICTIONS = ["profile", "sample_index", "image", "label", "prediction", "reference_prediction", "max_abs", "within_tolerance"]
SELECTION = ["id", "export_optimize", "runtime_optimize", "tolerance_failure_count", "output_mae", "output_max_abs", "export_seconds"]
HARDWARE = ["profile", "source", "model_sha256", "latency_p50_ms", "latency_p95_ms", "arena_used_bytes", "output_mae", "output_max_abs", "within_tolerance"]
CANDIDATES = ["id", "name", "precision", "status", "failure", "artifact_sha256", "configuration", "inventory", "incremental_build_seconds", "dependency_build_seconds"]


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def report_csv(report):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    def row(values):
        writer.writerow([_cell(x) for x in values])
    row(["EdgeLens measured host CPU report", report.get("run_id")])
    for key in ("model", "dataset", "datasets", "summary", "environment", "settings", "accuracy_resolution_pp", "search", "test_data_used_for_selection"):
        row([key, report.get(key)])
    selection = report.get("selection") or {}
    row(["selection", {k: v for k, v in selection.items() if k != "candidates"}])
    sections = [("Host CPU metrics", report.get("metrics", []), METRICS),
                ("Precision candidates (including failures)", report.get("candidates", []), CANDIDATES),
                ("Validation metrics", [c["validation_metrics"] for c in report.get("candidates", []) if c.get("validation_metrics")], METRICS),
                ("Calibration selection", selection.get("candidates", []), SELECTION),
                ("Layer diagnostics", report.get("layers", []), LAYERS),
                ("Per-image results", report.get("predictions", []), PREDICTIONS),
                ("Physical device reports", [x for x in report.get("edge_results", []) if x["kind"] == "hardware"], HARDWARE)]
    for name, rows, keys in sections:
        row([]); row([name]); row(keys)
        for value in rows:
            row([value.get(k) for k in keys])
    for key in ("artifacts", "edge_results", "methodology", "limitations"):
        row([]); row([key])
        for item in report.get(key, []):
            row([item])
    for metric in report.get("metrics", []):
        row(["Raw host latency samples (ms)", metric.get("profile"), metric.get("latency_samples_ms")])
        row(["Metric provenance", metric.get("profile"), metric.get("provenance")])
    for candidate in report.get("candidates", []):
        row(["Full candidate record", candidate])
    return stream.getvalue()


def report_html(report):
    def text(value):
        return escape("—" if value is None else str(value))
    def pretty(value):
        return "<pre>" + escape(json.dumps(value, indent=2, ensure_ascii=False)) + "</pre>"
    def table(rows, keys):
        if not rows:
            return "<p class='muted'>No evidence recorded.</p>"
        return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{text(k)}</th>" for k in keys) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{text(row.get(k))}</td>" for k in keys) + "</tr>" for row in rows) + "</tbody></table></div>"
    hardware = [x for x in report.get("edge_results", []) if x["kind"] == "hardware"]
    provider = [x for x in report.get("edge_results", []) if x["kind"] == "edge_impulse_result"]
    selection = report.get("selection") or {}
    layers = report.get("layers", [])
    compared = sum(x.get("mae") is not None for x in layers)
    drifts = [x for x in layers if x.get("status") in ("drift", "nonfinite")]
    resolution = report.get("accuracy_resolution_pp") or (100 / report["dataset"]["image_count"] if report.get("dataset", {}).get("image_count") else None)
    detail = "".join(f"<h2>{text(section.title())}</h2><ul>" + "".join(f"<li>{text(x)}</li>" for x in report.get(section, [])) + "</ul>" for section in ("methodology", "limitations"))
    candidates = report.get("candidates", [])
    if candidates:
        candidate_rows = [{"candidate": c["name"], "status": c["status"], "validation_accuracy_pct": (c.get("validation_metrics") or {}).get("accuracy_pct"),
                           "test_accuracy_pct": (c.get("test_metrics") or {}).get("accuracy_pct"),
                           "failure": (c.get("failure") or {}).get("message"), "artifact_sha256": c.get("artifact_sha256")} for c in candidates]
        selection_section = "<h2>2. FP32 and static INT8 candidates</h2><p>Fixed configurations; no automatic winner or test-based tuning. Calibration sets INT8 ranges; validation and held-out test results remain separate. QDQ graph coverage does not establish fully integer execution.</p>" + table(candidate_rows, ["candidate", "status", "validation_accuracy_pct", "test_accuracy_pct", "failure", "artifact_sha256"]) + "<h3>Dataset roles and hashes</h3>" + pretty(report.get("datasets", {})) + "<h3>Configuration, coverage, failures and provenance</h3>" + pretty(candidates)
    else:
        selection_section = f"<h2>2. Converter selection</h2><p>{text(selection.get('objective', 'No calibration-guided search in this run.'))}</p><p>Selected: {text(selection.get('selected'))} · Total search seconds: {text(selection.get('total_strategy_seconds'))}. Test data is held out.</p>" + table(selection.get("candidates", []), SELECTION) + pretty(selection.get("calibration", {}))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EdgeLens developer report</title>
<style>body{{font:14px/1.65 system-ui,sans-serif;color:#19352e;margin:3rem auto;max-width:1300px;padding:0 1.5rem}}h1{{font-size:2.5rem;letter-spacing:-1px}}h2{{margin-top:2.4rem}}.notice{{padding:1rem;background:#edf8f1;border-left:4px solid #28634b}}.muted{{color:#687c76}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce6e0;vertical-align:top}}th{{background:#edf3ef}}td{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8f7;padding:1rem;font-size:12px}}li{{margin-bottom:.5rem}}details{{margin:1rem 0}}summary{{cursor:pointer;font-weight:600}}@media print{{body{{margin:0}}thead{{display:table-header-group}}tr{{break-inside:avoid}}}}</style>
<p class="muted">EDGELENS / DEVELOPER EXPERIMENT RECORD</p><h1>{text(report.get('model', {}).get('name'))}</h1>
<p>{text(report.get('run_id'))} · {text(report.get('created_at'))}</p><div class="notice"><strong>{text(report.get('summary', {}).get('conclusion'))}</strong><p>Host CPU measurements · {len(hardware)} physical-device reports · {len(provider)} Edge Impulse analyses. These are separate evidence sources.</p></div>
<h2>1. Conversion and host performance</h2><p>Accuracy uses true labels. Agreement and numerical error use the declared PyTorch or FP32 ONNX reference when supplied. Original PyTorch conversion loss is unavailable for ONNX-only uploads. Accuracy deltas are percentage points; one changed prediction = {text(resolution)} pp on this dataset. Missing comparisons remain empty.</p>
{table(report.get('metrics', []), METRICS)}<p>Serialized file size is not runtime RAM. Small timing differences require independent repetitions.</p>
{selection_section}
<h2>3. Layer diagnostics</h2><p>{compared} of {len(layers)} entries compared numerically; {len(drifts)} require review. Unmapped operations are unverified. An observed difference at a boundary is evidence, not proof of its root cause.</p>
{table(layers, LAYERS)}
<h2>4. Per-image predictions</h2>{table(report.get('predictions', []), PREDICTIONS)}
<h2>5. Physical ESP32 results</h2>{table(hardware, HARDWARE)}<p>Device-reported Invoke-only samples and output on one fixed image. USB capture and imported JSON retain separate provenance. Neither carries cryptographic hardware attestation. Arena usage is not total RAM; one-image agreement is not dataset accuracy.</p>{pretty(hardware)}
<h2>6. Edge Impulse reference analysis</h2><p>Provider resource/timing analysis is not exact latency from your USB-connected ESP32.</p>{pretty(provider)}
<h2>7. Reproduce this experiment</h2><p>Use the identical model and dataset hashes, preprocessing, tolerances, runtime versions and settings. Exported PT2 must come from a trusted source. Download the corresponding artifacts from the saved run.</p>
{pretty({'model': report.get('model'), 'dataset': report.get('dataset'), 'datasets': report.get('datasets'), 'settings': report.get('settings'), 'environment': report.get('environment'), 'artifacts': report.get('artifacts')})}
<h3>Metric provenance</h3>{pretty({x.get('profile'): x.get('provenance') for x in report.get('metrics', [])})}
<h2>Raw host latency samples (ms)</h2>{pretty({x.get('profile'): x.get('latency_samples_ms') for x in report.get('metrics', [])})}{detail}</html>'''
