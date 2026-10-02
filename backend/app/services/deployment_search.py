"""Bounded candidate exploration using the existing builder/evaluator and validation only."""
from datetime import datetime, timezone
import math
import time

from app.services import benchmark as b
from app.services import quantization as q
from app.services.quantization_diagnostics import compare_activations


def select_candidates(candidates, constraints, baseline_accuracy):
    """Explicit hard constraints and Pareto analysis; never read test metrics."""
    eligible, measured = [], []
    for candidate in candidates:
        metric = candidate.get("validation_metrics") or {}
        values = [metric.get(key) for key in ("accuracy_pct", "size_bytes", "latency_p50_ms")]
        reasons = []
        if candidate["status"] == "failed":
            reasons.append("Candidate failed; no deployment recommendation.")
        if any(value is None or not math.isfinite(value) for value in values) or baseline_accuracy is None:
            reasons.append("Required validation accuracy, model size or host latency is unavailable.")
        if not reasons:
            loss = baseline_accuracy - metric["accuracy_pct"]
            for name, value, limit in (("Accuracy loss (percentage points)", loss, constraints.get("max_accuracy_loss_pp")),
                                       ("Serialized size (MiB)", metric["size_bytes"] / 1048576, constraints.get("max_size_mib")),
                                       ("Validation median host latency (ms)", metric["latency_p50_ms"], constraints.get("max_host_latency_ms"))):
                if limit is not None and value > limit + 1e-9:
                    reasons.append(f"{name}: {value:.6g} exceeds {limit:.6g}.")
            measured.append(candidate)
        candidate["eligibility"] = {"eligible": not reasons, "reasons": reasons, "evidence_split": "validation"}
        if not reasons:
            eligible.append(candidate)
    def dominates(left, right):
        x, y = left["validation_metrics"], right["validation_metrics"]
        a = (-x["accuracy_pct"], x["size_bytes"], x["latency_p50_ms"])
        z = (-y["accuracy_pct"], y["size_bytes"], y["latency_p50_ms"])
        return all(i <= j for i, j in zip(a, z)) and any(i < j for i, j in zip(a, z))
    pareto = [candidate["id"] for candidate in measured if not any(dominates(other, candidate) for other in measured if other is not candidate)]
    for candidate in candidates:
        candidate["pareto_efficient"] = candidate["id"] in pareto
    objective = constraints.get("objective", "tradeoffs")
    selected, order = None, []
    if objective != "tradeoffs" and eligible:
        order = {"latency": ["latency_p50_ms", "accuracy_desc", "size_bytes", "id"],
                 "size": ["size_bytes", "accuracy_desc", "latency_p50_ms", "id"],
                 "accuracy": ["accuracy_desc", "size_bytes", "latency_p50_ms", "id"]}[objective]
        def key(candidate):
            metric = candidate["validation_metrics"]
            return tuple(candidate["id"] if field == "id" else -metric["accuracy_pct"] if field == "accuracy_desc" else metric[field] for field in order)
        selected = min(eligible, key=key)["id"]
    return {"algorithm": "Validation constraints and explicit objective; no composite score", "selected": selected,
            "objective": objective, "constraints": constraints, "sort_order": order, "pareto_candidates": pareto,
            "eligible_candidates": [candidate["id"] for candidate in eligible],
            "decision": "selected" if selected else "tradeoffs_only" if objective == "tradeoffs" and eligible else "no_feasible_candidate",
            "reference_accuracy_pct": baseline_accuracy, "reference_profile": "fp32", "evidence_split": "validation",
            "test_data_used_for_selection": False, "frozen_at": datetime.now(timezone.utc).isoformat()}


def explore(*, baseline_path, candidates, profiles, artifacts, output_dir, calibration_arrays, validation_arrays,
            settings, search, constraints, initial_config, register, evaluate, fail):
    """No test arrays, test labels or test outputs enter this function."""
    started = time.perf_counter()
    deadline = started + search["max_seconds"]
    stopped = []
    sensitivity = []
    diagnostics = {"status": "unavailable", "rows": [], "reason": "No successful quantized candidate."}
    caches = {}
    fixed = candidates[1]["configuration"]
    def budget():
        if len(candidates) >= search["max_candidates"]:
            stopped.append("candidate_limit")
            return False
        if time.perf_counter() >= deadline:
            stopped.append("soft_time_budget")
            return False
        return True
    def validation(candidate):
        if candidate["status"] != "built":
            return
        profile = next(value for value in profiles if value.get("candidate") is candidate)
        try:
            candidate["validation_metrics"] = evaluate(profile)
        except Exception as exc:
            fail(candidate, "validation_evaluation", exc)
    for candidate in candidates:
        validation(candidate)
        cache = output_dir / "calibration-ranges.json"
        if candidate["id"] == "static_int8" and cache.exists():
            caches[initial_config["calibration_method"]] = cache

    def build(identifier, name, config, parent=None):
        if not budget():
            return None
        candidate = {"id": identifier, "name": name, "candidate_type": "selective_quantization" if config.get("nodes_to_exclude") else "static_quantization",
                     "precision": "Selective INT8 / FP32 weights" if config.get("nodes_to_exclude") else "INT8 QDQ with floating-point operations",
                     "format": "onnx", "status": "pending", "failure": None, "artifact_sha256": None,
                     "configuration": {**fixed, **config}, "validation_metrics": None, "test_metrics": None, "parent_candidate": parent}
        candidates.append(candidate)
        path = output_dir / identifier / "model.onnx"
        try:
            seconds, inventory = q.build_static_int8(baseline_path, path, calibration_arrays, config,
                                            calibration_cache=caches.get(config["calibration_method"]))
            candidate["incremental_build_seconds"] = seconds
            candidate["dependency_build_seconds"] = candidates[0].get("conversion_seconds") or 0
            register(candidate, path, seconds + candidate["dependency_build_seconds"])
            candidate["inventory"].update(inventory)
            cache = path.with_name("calibration-ranges.json")
            if cache.exists():
                caches.setdefault(config["calibration_method"], cache)
                artifact = b._artifact(identifier + "_calibration_ranges", "json", cache)
                artifact["role"] = "calibration_statistics"
                artifacts.append(artifact)
                candidate["calibration_cache_sha256"] = artifact["sha256"]
            validation(candidate)
        except Exception as exc:
            fail(candidate, "quantization_or_runtime_validation", exc)
        return candidate

    if candidates[0]["status"] == "built":
        for method in search["calibration_methods"]:
            for per_channel in ([True, False] if search["try_per_tensor"] else [True]):
                if method == initial_config["calibration_method"] and per_channel == initial_config["per_channel"]:
                    continue
                if not budget():
                    break
                build(f"int8_{method.lower()}_{'channel' if per_channel else 'tensor'}", f"INT8 {method} · {'per channel' if per_channel else 'per tensor'}",
                      {"calibration_method": method, "per_channel": per_channel, "nodes_to_exclude": []})
    quantized = [candidate for candidate in candidates if candidate["id"] != "fp32" and candidate["status"] == "built" and candidate.get("validation_metrics")]
    rank = lambda candidate: (-candidate["validation_metrics"]["accuracy_pct"], candidate["validation_metrics"]["output_mae"], candidate["id"])
    if quantized and time.perf_counter() < deadline:
        parent = min(quantized, key=rank)
        parent_profile = next(value for value in profiles if value.get("candidate") is parent)
        try:
            diagnostics = compare_activations(baseline_path, parent_profile["path"], calibration_arrays, settings, search["diagnostic_samples"])
            diagnostics["profile"] = parent["id"]
        except Exception as exc:
            diagnostics = {"status": "unavailable", "profile": parent["id"], "rows": [], "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
        ranked = sorted([row for row in diagnostics["rows"] if row.get("nrmse") is not None], key=lambda row: (-row["nrmse"], row["name"]))
        improving = []
        for row in ranked[:search["sensitivity_probes"]]:
            config = {key: parent["configuration"][key] for key in ("calibration_method", "per_channel")}
            candidate = build(f"exclude_{len(sensitivity) + 1}", f"Selective INT8 · exclude {row['name']}", config | {"nodes_to_exclude": [row["name"]]}, parent["id"])
            if candidate is None:
                break
            metric = candidate.get("validation_metrics")
            control = {"node": row["name"], "operation": row["operation"], "candidate_id": candidate["id"], "parent_candidate": parent["id"],
                       "status": "measured" if metric else "failed", "evidence_split": "validation", "observed_nrmse": row["nrmse"],
                       "accuracy_recovery_pp": metric["accuracy_pct"] - parent["validation_metrics"]["accuracy_pct"] if metric else None,
                       "output_mae_change": metric["output_mae"] - parent["validation_metrics"]["output_mae"] if metric else None,
                       "excluded_nodes": [row["name"]], "interpretation": "One operation excluded; other quantizer settings and calibration ranges held fixed. Measured effect under this protocol, not a universal root-cause claim."}
            sensitivity.append(control)
            if metric and candidate["status"] == "built" and rank(candidate)[:2] < rank(parent)[:2]:
                improving.append(row["name"])
        for count in range(2, len(improving) + 1):
            config = {key: parent["configuration"][key] for key in ("calibration_method", "per_channel")}
            if build(f"selective_{count}", f"Selective INT8 · {count} FP32 exclusions", config | {"nodes_to_exclude": improving[:count]}, parent["id"]) is None:
                break
    elif quantized:
        diagnostics["reason"] = "Soft time budget exhausted before diagnostic capture."

    valid_profiles = [profile for profile in profiles if profile.get("candidate", {}).get("status") == "built" and profile["candidate"].get("validation_metrics")]
    failures = {}
    timing = b._time_profiles([(profile["profile"], profile["infer"], validation_arrays[:16]) for profile in valid_profiles], settings, failures) if valid_profiles else {}
    for profile in valid_profiles:
        candidate = profile["candidate"]
        if candidate["id"] in failures:
            fail(candidate, "validation_host_benchmark", failures[candidate["id"]])
        else:
            candidate["validation_metrics"].update(timing[candidate["id"]])
    baseline_accuracy = (candidates[0].get("validation_metrics") or {}).get("accuracy_pct")
    selection = select_candidates(candidates, constraints, baseline_accuracy)
    return {"selection": selection, "sensitivity": sensitivity, "diagnostics": diagnostics,
            "search": {"candidate_limit": search["max_candidates"], "attempted_candidates": len(candidates), "automatic_selection": selection["selected"] is not None,
                       "elapsed_seconds": time.perf_counter() - started, "max_seconds": search["max_seconds"],
                       "budget_kind": "Soft budget checked between operations; an in-progress build or evaluation completes.",
                       "stopped_by": sorted(set(stopped)), "settings": search}}
