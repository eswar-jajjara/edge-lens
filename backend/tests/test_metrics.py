import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.metrics import classification_metrics, comparison_summary, percentile
from app.services.benchmark import RuntimeUnavailable, run_benchmark, runtime_capabilities, _scope_names, _time_profiles


class MetricTests(unittest.TestCase):
    def test_agreement_is_not_accuracy(self):
        result = classification_metrics([2, 1, 0], [2, 1, 0], [0, 1, 2])
        self.assertAlmostEqual(result["accuracy_pct"], 100 / 3)
        self.assertEqual(result["agreement_pct"], 100)
        self.assertEqual(result["accuracy_delta_pp"], 0)

    def test_accuracy_difference_is_signed_percentage_points(self):
        result = classification_metrics([0, 0], [1, 0], [1, 0])
        self.assertEqual(result["accuracy_delta_pp"], 50)
        self.assertEqual(result["agreement_pct"], 50)
        self.assertEqual(result["sample_count"], 2)

    def test_percentiles_do_not_require_large_samples(self):
        self.assertEqual(percentile([3], 95), 3)
        self.assertEqual(percentile([10, 20, 30, 40, 50], 50), 30)
        self.assertEqual(percentile([10, 20, 30, 40, 50], 95), 48)

    def test_invalid_observations_fail(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                percentile(values, 50)
        with self.assertRaises(ValueError):
            classification_metrics([0], [0, 1], [0])

    def test_summary_does_not_promote_equal_or_worse_result(self):
        standard = {"accuracy_pct": 75, "latency_p50_ms": 10, "conversion_seconds": 5}
        self.assertIn("no accuracy improvement", comparison_summary(standard, dict(standard))["conclusion"])
        worse = {"accuracy_pct": 50, "latency_p50_ms": 12, "conversion_seconds": 6}
        result = comparison_summary(standard, worse)
        self.assertEqual(result["accuracy_delta_pp"], -25)
        self.assertEqual(result["latency_delta_ms"], 2)
        self.assertIn("Standard conversion", result["conclusion"])

    def test_missing_runtime_is_actionable_and_never_returns_demo_data(self):
        with patch("app.services.benchmark._present", return_value=False):
            with self.assertRaisesRegex(RuntimeUnavailable, "requirements-ml.txt"):
                run_benchmark({"model_id": "resnet18", "format": "onnx", "target": "raspberry_pi"}, {}, None)

    def test_tflite_requires_linux_even_when_packages_are_detected(self):
        with patch("app.services.benchmark._present", return_value=True), patch("app.services.benchmark.platform.system", return_value="Windows"):
            caps = runtime_capabilities()
        self.assertTrue(caps["onnx"]["available"])
        self.assertFalse(caps["tflite"]["available"])
        self.assertEqual(set(caps), {"onnx", "tflite"})

    def test_timing_excludes_warmup_and_rotates_profile_order(self):
        calls = []
        profiles = [(name, lambda _, name=name: calls.append(name), 0) for name in ("a", "b", "c")]
        ticks = iter(range(0, 18_000_000, 1_000_000))
        with patch("app.services.benchmark.time.perf_counter_ns", side_effect=lambda: next(ticks)):
            measured = _time_profiles(profiles, {"warmup_runs": 1, "measured_runs": 3})
        self.assertEqual(calls[:3], ["a", "b", "c"])
        self.assertEqual(calls[3:], ["a", "b", "c", "b", "c", "a", "c", "a", "b"])
        self.assertEqual(measured["a"]["latency_samples_ms"], [1, 1, 1])

    def test_mapping_excludes_fx_node_name_to_avoid_false_leaf_matches(self):
        node = SimpleNamespace(metadata_props=[SimpleNamespace(key="pkg.torch.onnx.name_scopes", value="['', 'block', 'relu']")])
        self.assertEqual(_scope_names(node), ["", "block"])
        node.metadata_props[0].value = "['relu']"
        self.assertEqual(_scope_names(node), [])


if __name__ == "__main__":
    unittest.main()
