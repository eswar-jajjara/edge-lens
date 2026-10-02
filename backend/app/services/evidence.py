"""Evidence views derived from saved results; never infer missing measurements."""
from collections import Counter
import math
import re


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def artifact_evaluation(report, artifact):
    """Only explicit held-out metric hashes establish an accuracy/resource link."""
    dataset = report.get("datasets", {}).get("test") or report.get("dataset", {})
    if (not re.fullmatch(r"[a-f0-9]{64}", str(artifact.get("sha256", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(dataset.get("sha256", "")))
            or type(dataset.get("image_count")) is not int or dataset["image_count"] < 1):
        return {"status": "UNAVAILABLE", "reason": "An evaluated artifact, dataset hash and positive held-out image count are required."}
    matches = [m for m in report.get("metrics", [])
               if m.get("artifact_sha256") == artifact.get("sha256")
               and m.get("evaluation_split") == "test"
               and m.get("dataset_sha256") == dataset.get("sha256")
               and m.get("sample_count") == dataset.get("image_count")
               and finite(m.get("accuracy_pct"))]
    if not artifact.get("sha256") or not dataset.get("sha256") or not matches:
        return {"status": "UNAVAILABLE", "reason": "No explicit matching artifact hash and complete held-out evaluation. Re-run evaluation with the current engine."}
    metric = matches[0]
    return {"status": "MEASURED", "artifact_sha256": artifact["sha256"],
            "metric_profile": metric["profile"], "dataset_sha256": dataset["sha256"],
            "preprocessed_sha256": dataset.get("preprocessed_sha256"),
            "accuracy_pct": metric["accuracy_pct"], "sample_count": metric["sample_count"],
            "preprocessing": {k: report.get("model", {}).get(k) for k in
                              ("input_shape", "layout", "scale", "mean", "std", "resize", "resize_shorter", "class_count")},
            "labels": dataset.get("labels")}


def diagnostic_groups(report):
    groups = {}
    for row in report.get("layers", []):
        profile = row.get("profile", "unknown")
        group = groups.setdefault(profile, {"profile": profile, "measured": [], "inventory": [], "unavailable": []})
        measured = finite(row.get("mae")) and finite(row.get("max_abs"))
        # A failed/non-finite comparison is an attempted diagnostic, not inventory.
        category = "measured" if measured else "unavailable" if row.get("scope") == "calibration_diagnostic" or row.get("status") == "nonfinite" else "inventory"
        group[category].append(row)
    result = []
    for group in groups.values():
        calibration = [r for r in group["measured"] if r.get("scope") == "calibration_diagnostic"]
        unavailable = [r for r in group["unavailable"] if r.get("scope") == "calibration_diagnostic"]
        group.update(measured_count=len(group["measured"]), inventory_count=len(group["inventory"]),
                     unavailable_count=len(group["unavailable"]),
                     quantization_helper_count=sum(r.get("operation") in {"QuantizeLinear", "DequantizeLinear"} for r in group["inventory"]),
                     diagnostic_compared=len(calibration), diagnostic_eligible=len(calibration) + len(unavailable),
                     diagnostic_sample_count=max((r.get("sample_count", 0) for r in calibration), default=0),
                     inventory_operations=dict(Counter(r.get("operation", "unknown") for r in group["inventory"])))
        result.append(group)
    return result


def evidence_sections(report):
    artifacts = report.get("artifacts", [])
    provider = []
    for record in report.get("edge_results", []):
        if record.get("kind") != "edge_impulse_result":
            continue
        artifact = next((a for a in artifacts if a.get("sha256") == record.get("model_sha256")), None)
        evaluation = artifact_evaluation(report, artifact) if artifact else {"status": "UNAVAILABLE", "reason": "Profiled artifact hash does not match this run."}
        captured = record.get("evaluated_artifact") or {}
        linked = evaluation["status"] == "MEASURED" and all(captured.get(k) == evaluation.get(k) for k in
                    ("artifact_sha256", "dataset_sha256", "preprocessed_sha256", "preprocessing", "labels", "sample_count", "accuracy_pct"))
        is_onnx = record.get("model_format") == "onnx" or record.get("provider_protocol") == "onnx_byom" or (artifact or {}).get("format") == "onnx"
        provider.append({**record, "evidence_status": "ESTIMATED", "upload_evaluation_link": "VERIFIED" if linked else "UNAVAILABLE",
                         "accuracy_link": "VERIFIED" if linked and not is_onnx else "UNAVAILABLE",
                         "evaluation": evaluation if linked and not is_onnx else {"status": "UNAVAILABLE", "reason":
                            "Edge Impulse converts the uploaded ONNX model. Accuracy of that converted artifact has not been evaluated; the uploaded ONNX accuracy remains a separate host measurement."
                            if is_onnx else "The provider record lacks matching evaluated-artifact provenance."},
                         "uploaded_artifact_evaluation": evaluation if linked else {"status": "UNAVAILABLE"}})
    hardware = [r for r in report.get("edge_results", []) if r.get("kind") == "hardware"]
    return {"host": {"status": "MEASURED" if report.get("source") == "measured" and report.get("metrics") else "UNAVAILABLE", "scope": "host_cpu"},
            "edge_impulse": {"status": "ESTIMATED" if provider else "UNAVAILABLE", "records": provider,
                             "verification": "Provider responses recorded; see each record's provenance." if provider else "Tested with mocks only; no live profile is recorded in this run."},
            "esp32": {"status": "MEASURED" if hardware else "UNAVAILABLE", "records": hardware,
                      "reason": None if hardware else "No physical ESP32 benchmark has been performed for this run."}}
