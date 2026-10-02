import copy
from pathlib import Path
import tempfile
import unittest
import importlib.util
import os
from app.services.structural import compare_onnx


class StructuralTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX runtime not installed')
    def test_known_faults_and_unchanged_control(self):
        from app.services.fault_validation import run_fault_suite
        with tempfile.TemporaryDirectory(dir=os.environ.get('EDGELENS_TEST_TMP')) as tmp:
            result = run_fault_suite(Path(tmp) / 'faults')
            self.assertTrue(result['passed'], result['cases'])
            rows = {r['fault']: r for r in result['cases']}
            self.assertEqual(rows['changed_weights']['first_observed_divergence'], 'classifier')
            self.assertEqual(rows['replaced_operator']['first_observed_divergence'], 'activation')
            self.assertEqual(rows['wrong_preprocessing']['first_observed_divergence'], 'scale')
            self.assertIsNotNone(rows['wrong_shape']['error'])
            self.assertIsNotNone(rows['removed_operator']['error'])

    def test_ambiguous_identity_never_matches(self):
        node = {'name': 'same', 'operation': 'Relu', 'domain': '', 'inputs': [], 'outputs': [], 'parents': [], 'output_shapes': [], 'attributes_sha256': '', 'weights': {}}
        value = compare_onnx({'nodes': [node, copy.deepcopy(node)]}, {'nodes': [node]})
        self.assertEqual(value['matched'], 0)
        self.assertTrue(all(r['reason_code'] == 'ambiguous_node_identity' for r in value['rows']))
