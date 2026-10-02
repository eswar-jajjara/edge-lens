"""Self-contained developer reports, with separate host/device/provider evidence."""
import csv
from html import escape
import io
import json
from app.services.evidence import diagnostic_groups, evidence_sections

METRICS = ["label", "accuracy_pct", "correct_count", "sample_count", "accuracy_delta_pp", "accuracy_delta_vs_fp32_pp", "reference_profile", "agreement_pct", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_min_ms", "latency_max_ms", "latency_p25_ms", "latency_p75_ms", "latency_stddev_ms", "timing_input_count", "timing_evidence_status", "process_rss_sampled_peak_bytes", "memory_evidence_status", "conversion_seconds", "size_bytes", "output_mae", "output_max_abs", "tolerance_failure_count", "artifact_sha256", "dataset_sha256", "evaluation_split"]
LAYERS = ["profile", "name", "operation", "status", "reason_code", "mae", "max_abs", "nrmse", "sample_count", "scope", "sample_index", "expected_shape", "actual_shape", "max_error_index", "expected_value", "actual_value", "detail"]
SENSITIVITY = ["node", "operation", "candidate_id", "parent_candidate", "status", "accuracy_recovery_pp", "output_mae_change", "excluded_nodes", "evidence_split", "interpretation"]
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
    row(["EdgeLens experiment report", report.get("run_id")])
    row(["Evidence sections", evidence_sections(report)])
    row(["Diagnostic coverage by candidate", [{k: v for k, v in g.items() if k not in {"measured", "inventory", "unavailable"}} for g in diagnostic_groups(report)]])
    for key in ("model", "dataset", "datasets", "summary", "environment", "settings", "edge_estimate_request", "accuracy_resolution_pp", "search", "diagnostics", "structural", "conversion_candidates", "calibration", "test_data_used_for_selection"):
        row([key, report.get(key)])
    selection = report.get("selection") or {}
    row(["selection", {k: v for k, v in selection.items() if k != "candidates"}])
    sections = [("Host CPU metrics", report.get("metrics", []), METRICS),
                ("Precision candidates (including failures)", report.get("candidates", []), CANDIDATES),
                ("Validation metrics", [c["validation_metrics"] for c in report.get("candidates", []) if c.get("validation_metrics")], METRICS),
                ("Calibration selection", selection.get("candidates", []), SELECTION),
                ("Layer diagnostics", report.get("layers", []), LAYERS),
                ("Controlled validation sensitivity", report.get("sensitivity", []), SENSITIVITY),
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
    evidence = evidence_sections(report)
    provider = evidence["edge_impulse"]["records"]
    selection = report.get("selection") or {}
    layers = report.get("layers", [])
    compared = sum(x.get("mae") is not None for x in layers)
    drifts = [x for x in layers if x.get("status") in ("drift", "nonfinite")]
    layer_sections = ""
    for group in diagnostic_groups(report):
        layer_sections += f"<h3>Candidate: {text(group['profile'])}</h3><p>{group['measured_count']} numerical comparisons · {group['inventory_count']} inventory entries · {group['unavailable_count']} unavailable diagnostics."
        if group["diagnostic_eligible"]:
            layer_sections += f" Calibration operation coverage: {group['diagnostic_compared']} / {group['diagnostic_eligible']} eligible operations, on {group['diagnostic_sample_count']} calibration images."
        layer_sections += "</p><h4>Measured comparisons</h4>" + table(group["measured"], LAYERS)
        if group["unavailable"]:
            layer_sections += "<h4>Unavailable or failed diagnostic captures</h4>" + table(group["unavailable"], LAYERS)
        if group["inventory"]:
            layer_sections += f"<details><summary>Operation inventory — {group['inventory_count']} entries, including {group['quantization_helper_count']} Q/DQ helpers</summary><p>Inventory establishes presence only. Not compared means no numerical comparison was recorded; it does not mean a failed conversion.</p>" + table([{"operation": op, "count": count} for op, count in group["inventory_operations"].items()], ["operation", "count"]) + "<details><summary>Full inventory entries</summary>" + table(group["inventory"], LAYERS) + "</details></details>"
    resolution = report.get("accuracy_resolution_pp") or (100 / report["dataset"]["image_count"] if report.get("dataset", {}).get("image_count") else None)
    detail = "".join(f"<h2>{text(section.title())}</h2><ul>" + "".join(f"<li>{text(x)}</li>" for x in report.get(section, [])) + "</ul>" for section in ("methodology", "limitations"))
    candidates = report.get("candidates", [])
    if candidates:
        candidate_rows = [{"candidate": c["name"], "status": c["status"], "validation_accuracy_pct": (c.get("validation_metrics") or {}).get("accuracy_pct"),
                           "validation_latency_ms": (c.get("validation_metrics") or {}).get("latency_p50_ms"),
                           "eligible": (c.get("eligibility") or {}).get("eligible"), "constraint_rejections": (c.get("eligibility") or {}).get("reasons"),
                           "pareto_efficient": c.get("pareto_efficient"),
                           "test_accuracy_pct": (c.get("test_metrics") or {}).get("accuracy_pct"),
                           "failure": (c.get("failure") or {}).get("message"), "artifact_sha256": c.get("artifact_sha256")} for c in candidates]
        selection_section = "<h2>2. FP32 and static INT8 candidates</h2><p>Fixed configurations; no automatic winner or test-based tuning. Calibration sets INT8 ranges; validation and held-out test results remain separate. QDQ graph coverage does not establish fully integer execution.</p>" + table(candidate_rows, ["candidate", "status", "validation_accuracy_pct", "test_accuracy_pct", "failure", "artifact_sha256"]) + "<h3>Dataset roles and hashes</h3>" + pretty(report.get("datasets", {})) + "<h3>Configuration, coverage, failures and provenance</h3>" + pretty(candidates)
        if report.get("experiment_type") == "deployment_optimization":
            selection_section = "<h2>2. Deployment configuration search</h2><p>Calibration fits ranges and collects operation diagnostics. Validation alone determines constraints, Pareto membership and the selected objective. Selection is frozen before held-out evaluation. Unselected candidates retain validation evidence only.</p>" + pretty(selection) + table(candidate_rows, ["candidate", "status", "validation_accuracy_pct", "validation_latency_ms", "eligible", "constraint_rejections", "pareto_efficient", "test_accuracy_pct", "failure"]) + "<h3>Controlled one-operation exclusion experiments</h3>" + table(report.get("sensitivity", []), SENSITIVITY) + "<h3>Bounded activation diagnostics</h3>" + pretty(report.get("diagnostics", {})) + "<h3>Search budget</h3>" + pretty(report.get("search", {})) + "<h3>Dataset roles and hashes</h3>" + pretty(report.get("datasets", {})) + "<details><summary>Full candidate configurations, coverage and provenance</summary>" + pretty(candidates) + "</details>"
    else:
        selection_section = f"<h2>2. Converter selection</h2><p>{text(selection.get('objective', 'No calibration-guided search in this run.'))}</p><p>Selected: {text(selection.get('selected'))} · Total search seconds: {text(selection.get('total_strategy_seconds'))}. Test data is held out.</p>" + table(selection.get("candidates", []), SELECTION) + pretty(selection.get("calibration", {}))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EdgeLens developer report</title>
<style>body{{font:14px/1.65 system-ui,sans-serif;color:#19352e;margin:3rem auto;max-width:1300px;padding:0 1.5rem}}h1{{font-size:2.5rem;letter-spacing:-1px}}h2{{margin-top:2.4rem}}.notice{{padding:1rem;background:#edf8f1;border-left:4px solid #28634b}}.muted{{color:#687c76}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce6e0;vertical-align:top}}th{{background:#edf3ef}}td{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8f7;padding:1rem;font-size:12px}}li{{margin-bottom:.5rem}}details{{margin:1rem 0}}summary{{cursor:pointer;font-weight:600}}@media print{{body{{margin:0}}thead{{display:table-header-group}}tr{{break-inside:avoid}}}}</style>
<p class="muted">EDGELENS / DEVELOPER EXPERIMENT RECORD</p><h1>{text(report.get('model', {}).get('name'))}</h1>
<p>{text(report.get('run_id'))} · {text(report.get('created_at'))}</p><div class="notice"><strong>{text(report.get('summary', {}).get('conclusion'))}</strong><p>Host CPU measurements · {len(hardware)} physical-device reports · {len(provider)} Edge Impulse analyses. These are separate evidence sources.</p></div>
<h2>1. Laptop / host — {evidence['host']['status']}</h2><p>Accuracy uses true labels. Agreement and numerical error use the declared PyTorch or FP32 ONNX reference when supplied. Original PyTorch conversion loss is unavailable for ONNX-only uploads. Accuracy deltas are percentage points; one changed prediction = {text(resolution)} pp on this dataset. Missing comparisons remain empty.</p>
{table(report.get('metrics', []), METRICS)}<p>Serialized file size is not runtime RAM. Small timing differences require independent repetitions.</p>
{selection_section}
<h2>3. Layer diagnostics</h2><p>{compared} numerical comparisons recorded across candidates; {len(drifts)} require review. Inventory and unavailable captures are shown separately below. Counts across candidates are repeated graph entries, not unique model layers. An observed difference at a boundary is evidence, not proof of its root cause.</p>
{layer_sections}
<h3>Structural graph comparison · separate from numerical fidelity</h3><p>MEASURED inventories list operators, tensor shapes and connections. MATCHED means only the declared boundary checks matched; it is not a numerical pass. CHANGED can be a legal converter optimization. UNAVAILABLE mappings remain explicit.</p>
{''.join('<h4>' + text(s.get('profile')) + '</h4>' + table((s.get('comparison') or {}).get('rows', []), ['name', 'operation', 'status', 'reason_code', 'changes', 'expected_shape', 'actual_shape', 'expected_parents', 'actual_parents']) + '<details><summary>Graph inventory and mapping evidence</summary>' + pretty(s) + '</details>' for s in report.get('structural', []))}
<h2>4. Per-image predictions</h2>{table(report.get('predictions', []), PREDICTIONS)}
<h2>5. ESP32 — {evidence['esp32']['status']}</h2>{table(hardware, HARDWARE)}<p>{text(evidence["esp32"]["reason"]) if not hardware else ""}</p><p>Device-reported Invoke-only samples and output on one fixed image. USB capture and imported JSON retain separate provenance. Neither carries cryptographic hardware attestation. Arena usage is not total RAM; one-image agreement is not dataset accuracy.</p>{pretty(hardware)}
<h2>6. Edge Impulse — {evidence['edge_impulse']['status']}</h2><h3>Test estimate choice and destination</h3>{pretty(report.get('edge_estimate_request', {'enabled': False, 'note': 'No estimate choice recorded in this older test.'}))}<p>{text(evidence["edge_impulse"]["verification"])}</p><p>A Yes request is not a completed estimate. ESTIMATED provider resources and timing are separate from physical measurements. Accuracy is linked only when the uploaded SHA-256 matches an explicitly evaluated held-out artifact. Older unverified records retain UNAVAILABLE accuracy links.</p>{pretty(provider)}
<h2>7. Reproduce this experiment</h2><p>Use the identical model and dataset hashes, preprocessing, tolerances, runtime versions and settings. Exported PT2 must come from a trusted source. Download the corresponding artifacts from the saved run.</p>
{pretty({'model': report.get('model'), 'dataset': report.get('dataset'), 'datasets': report.get('datasets'), 'settings': report.get('settings'), 'environment': report.get('environment'), 'artifacts': report.get('artifacts')})}
<h3>Metric provenance</h3>{pretty({x.get('profile'): x.get('provenance') for x in report.get('metrics', [])})}
<h2>Host memory measurement method</h2><p>Sampled process RSS includes all loaded models, inputs and runtime state. A separate inference pass samples memory without perturbing the recorded timing pass. This is neither model-only RAM nor a guaranteed true peak.</p>{pretty({x.get("profile"): {k:v for k,v in x.items() if k.startswith("memory_") or k.startswith("process_rss_") or k.startswith("timing_")} for x in report.get("metrics", [])})}<h2>Raw host latency samples (ms)</h2>{pretty({x.get('profile'): x.get('latency_samples_ms') for x in report.get('metrics', [])})}{detail}</html>'''
