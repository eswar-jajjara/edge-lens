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
    for key in ("model", "dataset", "datasets", "summary", "environment", "settings", "edge_estimate_request", "accuracy_resolution_pp", "search", "diagnostics", "structural", "fault_validation", "conversion_candidates", "calibration", "test_data_used_for_selection"):
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
    from app.services.report_layout import render_report
    return render_report(report)
