import copy
from pathlib import Path
import tempfile
import unittest
import importlib.util
import os
from uuid import uuid4
from app.services.structural import compare_onnx


class StructuralTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX runtime not installed')
    def test_known_faults_and_unchanged_control(self):
        from app.services.fault_validation import run_fault_suite
        directory = Path(os.environ.get('EDGELENS_TEST_TMP', tempfile.gettempdir())) / ('faults-' + uuid4().hex)
        result = run_fault_suite(directory)
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

    @unittest.skipUnless(os.environ.get('EDGELENS_TEST_CUSTOM') == '1', 'Full ML environment required')
    def test_real_pooling_flatten_boundaries_and_renamed_input(self):
        import torch
        import numpy as np
        from app.services.developer_benchmark import _export, _diagnostics, onnx_session
        from app.services.structural import fx_inventory, onnx_inventory, compare_fx_onnx
        directory = Path(os.environ.get('EDGELENS_TEST_TMP', tempfile.gettempdir())) / ('mapping-' + uuid4().hex)
        directory.mkdir(parents=True)
        module = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2)).eval()
        sample = torch.ones(1, 3, 8, 8)
        ep = torch.export.export(module, (sample,))
        path = directory / 'unoptimized.onnx'; _export(ep, sample, path, False)
        result = compare_fx_onnx(fx_inventory(ep), onnx_inventory(path))
        self.assertEqual(result['matched'], 5, result['rows'])
        self.assertEqual(result['changed'], 0, result['rows'])
        infer, _ = onnx_session(path, 1)
        with torch.no_grad():
            rows = _diagnostics(ep, sample.numpy(), [{'profile': 'diagnostic', 'path': path, 'export_optimize': False, 'infer': infer}], {}, {'threads': 1, 'atol': 1e-4, 'rtol': 1e-3})
        self.assertTrue(all(row['status'] == 'pass' for row in rows), rows)

    def test_reused_scope_does_not_create_a_pooling_mapping(self):
        from app.services.structural import fx_boundaries
        sources = [{'name': name, 'operation': 'aten.adaptive_avg_pool2d.default', 'kind': 'call_function', 'module_scopes': ['', 'pool'], 'output_shape': [1, 3, 1, 1]} for name in ('pool_1', 'pool_2')]
        target = {'name': 'mean', 'origin': 'mean', 'module_scopes': ['', 'pool'], 'operation': 'ReduceMean', 'domain': '', 'output_shapes': [[1, 3, 1, 1]], 'outputs': ['pooled']}
        result = fx_boundaries({'nodes': sources}, {'nodes': [target]})
        self.assertTrue(all(item['target'] is None for item in result.values()))

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX runtime not installed')
    def test_self_test_api_saves_results_and_rejects_active_benchmarks(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        from app.core.config import Settings
        directory = Path(os.environ.get('EDGELENS_TEST_TMP', tempfile.gettempdir())) / ('fault-api-' + uuid4().hex)
        app = create_app(Settings(environment='test', allow_custom_models=True, data_dir=directory))
        with TestClient(app) as client:
            response = client.post('/api/v1/diagnostics/self-test')
            self.assertEqual(response.status_code, 201, response.text)
            run = response.json()
            self.assertTrue(run['report']['fault_validation']['passed'])
            exported = client.get('/api/v1/runs/' + run['id'] + '/report.html')
            self.assertIn('Controlled fault validation', exported.text)
            app.state.repository.create_run({'model_id': 'waiting'})
            self.assertEqual(client.post('/api/v1/diagnostics/self-test').status_code, 409)
