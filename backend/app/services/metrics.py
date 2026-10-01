"""Small, dependency-free statistics shared by the optional ML worker."""
import math
from collections.abc import Sequence


def percentile(values: Sequence[float], percent: float) -> float:
    """Linearly interpolated percentile, including single-observation samples."""
    if not values or not 0 <= percent <= 100:
        raise ValueError("Percentile requires observations and a percentage from 0 to 100")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Observations must be finite")
    rank = (len(ordered) - 1) * percent / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def classification_metrics(reference: Sequence[int], candidate: Sequence[int],
                           labels: Sequence[int]) -> dict:
    if not labels or len(reference) != len(labels) or len(candidate) != len(labels):
        raise ValueError("Predictions and labels must have the same nonzero length")
    count = len(labels)
    correct = sum(predicted == label for predicted, label in zip(candidate, labels))
    reference_correct = sum(predicted == label for predicted, label in zip(reference, labels))
    agreements = sum(left == right for left, right in zip(reference, candidate))
    return {
        "accuracy_pct": 100 * correct / count,
        "accuracy_delta_pp": 100 * (correct - reference_correct) / count,
        "agreement_pct": 100 * agreements / count,
        "correct_count": correct,
        "sample_count": count,
    }


def comparison_summary(standard: dict, dashboard: dict) -> dict:
    """Signed measured differences: positive accuracy is better; latency is not."""
    delta = dashboard["accuracy_pct"] - standard["accuracy_pct"]
    if delta == 0:
        conclusion = "Equal measured top-1 accuracy on this dataset; no accuracy improvement demonstrated."
    elif delta > 0:
        conclusion = "Dashboard configuration has higher measured top-1 accuracy on this dataset only."
    else:
        conclusion = "Standard conversion has higher measured top-1 accuracy on this dataset."
    return {
        "conclusion": conclusion,
        "accuracy_delta_pp": delta,
        "latency_delta_ms": dashboard["latency_p50_ms"] - standard["latency_p50_ms"],
        "conversion_delta_seconds": dashboard["conversion_seconds"] - standard["conversion_seconds"],
    }
