import unittest
from unittest.mock import patch
from app.services.host_measurements import measure_memory
from app.services.benchmark import _time_profiles


class HostMeasurementTests(unittest.TestCase):
    def test_shared_inputs_rotate_and_memory_pass_is_outside_raw_timing(self):
        seen = []
        with patch('app.services.host_measurements.measure_memory', return_value={'memory_evidence_status': 'MEASURED'}):
            values = _time_profiles([('a', lambda x: seen.append(('a', x)), [10, 20]), ('b', lambda x: seen.append(('b', x)), [10, 20])], {'warmup_runs': 1, 'measured_runs': 4})
        self.assertEqual(seen[2:], [('a', 10), ('b', 10), ('b', 20), ('a', 20), ('a', 10), ('b', 10), ('b', 20), ('a', 20)])
        self.assertEqual(values['a']['timing_sample_indices'], [0, 1, 0, 1])
        self.assertEqual(len(values['a']['latency_samples_ms']), 4)
        self.assertGreaterEqual(values['a']['latency_max_ms'], values['a']['latency_min_ms'])

    def test_memory_failure_does_not_fabricate_zero(self):
        with patch('app.services.host_measurements.rss_bytes', side_effect=OSError('not supported')):
            value = measure_memory(lambda _: None, None)
        self.assertEqual(value['memory_evidence_status'], 'UNAVAILABLE')
        self.assertNotIn('process_rss_sampled_peak_bytes', value)
