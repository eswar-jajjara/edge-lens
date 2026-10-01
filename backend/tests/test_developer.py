"""Real tiny classifiers verify conversion plumbing; synthetic accuracy is not project evidence."""
import io
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4
import zipfile
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from app.repositories import Repository
from app.services.models import ModelSpec
from app.services.developer_benchmark import rank_fidelity, disjoint_inputs
from app.services.hardware import validate_observation


def tiny_tflite(quantized=False):
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as s
    model = s.ModelT()
    model.version = 3
    model.description = 'EdgeLens synthetic test fixture - not accuracy evidence'
    opcode = s.OperatorCodeT(); opcode.builtinCode = s.BuiltinOperator.FULLY_CONNECTED; opcode.version = 1
    opcode.deprecatedBuiltinCode = s.BuiltinOperator.FULLY_CONNECTED
    model.operatorCodes = [opcode]
    buffers = [s.BufferT() for _ in range(3)]
    buffers[1].data = np.array([.05] * 12 + [-.05] * 12, dtype='<f4').view(np.uint8)
    buffers[2].data = np.array([0., 1.], dtype='<f4').view(np.uint8)
    model.buffers = buffers
    graph = s.SubGraphT(); graph.name = 'synthetic_classifier'
    tensors = []
    for name, shape, buffer in [('image', [1, 2, 2, 3], 0), ('weight', [2, 12], 1), ('bias', [2], 2), ('logits', [1, 2], 0)]:
        tensor = s.TensorT(); tensor.name = name; tensor.shape = np.array(shape, dtype=np.int32)
        tensor.type = s.TensorType.FLOAT32; tensor.buffer = buffer; tensors.append(tensor)
    graph.tensors = tensors; graph.inputs = np.array([0], dtype=np.int32); graph.outputs = np.array([3], dtype=np.int32)
    if quantized:
        buffers[1].data = np.array([1] * 12 + [-1] * 12, dtype=np.int8).view(np.uint8)
        buffers[2].data = np.array([0, 5100], dtype='<i4').view(np.uint8)
        for index, tensor in enumerate(tensors):
            tensor.type = s.TensorType.INT32 if index == 2 else s.TensorType.INT8
            quantization = s.QuantizationParametersT()
            quantization.scale = np.array([[(1/255), .05, .05/255, .01][index]], dtype=np.float32)
            quantization.zeroPoint = np.array([-128 if index == 0 else 0], dtype=np.int64)
            tensor.quantization = quantization
    op = s.OperatorT(); op.opcodeIndex = 0; op.inputs = np.array([0, 1, 2], dtype=np.int32); op.outputs = np.array([3], dtype=np.int32)
    op.builtinOptionsType = s.BuiltinOptions.FullyConnectedOptions; op.builtinOptions = s.FullyConnectedOptionsT()
    graph.operators = [op]; model.subgraphs = [graph]
    builder = flatbuffers.Builder(1024); offset = model.Pack(builder); builder.Finish(offset, file_identifier=b'TFL3')
    return bytes(builder.Output())


def images_zip(offset=0):
    from PIL import Image
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('labels.json', json.dumps({'dark': 0, 'light': 1}))
        for i, folder in enumerate(('dark', 'light')):
            image = Image.new('RGB', (2, 2), (30 + offset + i * 80, 20 + offset, 10 + offset))
            raw = io.BytesIO(); image.save(raw, format='PNG')
            archive.writestr(folder + '/image.png', raw.getvalue())
    return stream.getvalue()


@unittest.skipUnless(os.environ.get('EDGELENS_TEST_CUSTOM') == '1', 'Set EDGELENS_TEST_CUSTOM=1 for actual tiny ML exports')
class DeveloperIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.root = Path(__file__).resolve().parents[1] / 'data' / ('custom_test_' + uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.pt2 = cls.root / 'test.pt2'
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2), torch.nn.Sigmoid()).eval()
        torch.manual_seed(4)
        ep = torch.export.export(model, (torch.zeros(1, 3, 2, 2),))
        torch.export.save(ep, str(cls.pt2))
        cls.app = create_app(Settings(environment='test', data_dir=cls.root, allow_custom_models=True))
        cls.context = TestClient(cls.app); cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        shutil.rmtree(cls.root)

    def upload(self, content, format='pt2'):
        spec = {'name': 'Synthetic unit-test classifier', 'format': format, 'input_shape': [1, 3, 2, 2] if format != 'tflite' else [1, 2, 2, 3],
                'layout': 'NCHW' if format != 'tflite' else 'NHWC', 'class_count': 2, 'trusted_source': True}
        response = self.client.post('/api/v1/models', content=content, headers={'X-Model-Spec': quote(json.dumps(spec))})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertNotIn('path', response.json())
        return response.json()['id']

    def dataset(self, offset):
        response = self.client.post('/api/v1/datasets', content=images_zip(offset), headers={'Content-Type': 'application/zip'})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['id']

    def run_model(self, model, dataset, format='onnx', calibration=None):
        payload = {'model_id': model, 'format': format, 'target': 'esp32', 'dataset_id': dataset,
                   'settings': {'warmup_runs': 1, 'measured_runs': 3, 'threads': 1}, 'strategy': 'fidelity_search' if calibration else 'fixed_profiles', 'calibration_dataset_id': calibration}
        response = self.client.post('/api/v1/runs', json=payload)
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        for _ in range(600):
            run = self.client.get('/api/v1/runs/' + run_id).json()
            if run['status'] in ('completed', 'failed'): break
            time.sleep(.1)
        self.assertEqual(run['status'], 'completed', run)
        return run

    def test_01_pt2_search_and_imported_onnx(self):
        model = self.upload(self.pt2.read_bytes())
        dataset, calibration = self.dataset(0), self.dataset(3)
        run = self.run_model(model, dataset, calibration=calibration)
        report = run['report']
        self.assertEqual(len(report['metrics']), 3)
        self.assertEqual(report['accuracy_resolution_pp'], 50)
        self.assertFalse(report['selection']['test_data_used_for_selection'])
        candidates = report['selection']['candidates']
        self.assertEqual(len(candidates), 3)
        winner = min(candidates, key=rank_fidelity)
        self.assertEqual(winner['id'], report['selection']['selected'])
        self.assertTrue(any(x['name'] == 'output.logits' and x['status'] in ('pass', 'drift') for x in report['layers']))
        self.assertTrue(any(x['name'] != 'output.logits' for x in report['layers']))
        self.assertTrue(any(x['name'] == 'linear' and x['status'] in ('pass', 'drift') for x in report['layers']))
        self.assertEqual(len(report['predictions']), 6)
        onnx_bytes = self.client.get(f'/api/v1/runs/{run["id"]}/artifacts/2').content
        uploaded = self.upload(onnx_bytes, 'onnx')
        standalone = self.run_model(uploaded, dataset)
        self.assertEqual(len(standalone['report']['metrics']), 1)
        self.assertIsNone(standalone['report']['metrics'][0]['output_mae'])
        self.assertIn('No original PyTorch', standalone['report']['summary']['conclusion'])
        for extension in ('.html', '.csv', ''):
            result = self.client.get(f'/api/v1/runs/{run["id"]}/report{extension}')
            self.assertEqual(result.status_code, 200)
            self.assertNotIn(str(self.root), result.text)

    def test_02_tflite_firmware_and_report_provenance(self):
        model = self.upload(tiny_tflite(), 'tflite')
        run = self.run_model(model, self.dataset(6), 'tflite')
        result = self.client.post(f'/api/v1/runs/{run["id"]}/edge/packages', json={'artifact_index': 0, 'arena_kib': 96})
        self.assertEqual(result.status_code, 201, result.text)
        package = result.json()
        self.assertEqual(package['operators'], ['FULLY_CONNECTED'])
        archive_bytes = self.client.get(f'/api/v1/edge/packages/{package["id"]}/download').content
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            source = archive.read('main/main.cc').decode()
            self.assertNotIn('@RESOLVER@', source)
            self.assertIn('resolver.AddFullyConnected()', source)
        observation = {'protocol': 'edgelens.esp32.v1', 'package_id': package['id'], 'model_sha256': package['model_sha256'], 'input_sha256': package['input_sha256'],
                       'chip': 'esp32', 'idf_version': 'test_fixture', 'cpu_freq_mhz': 240, 'arena_used_bytes': 1024, 'arena_capacity_bytes': 96 * 1024,
                       'warmup_runs': 1, 'samples_us': [100, 110, 90], 'output': manifest['expected_output']}
        bad = self.client.post(f'/api/v1/edge/packages/{package["id"]}/import', json=observation | {'model_sha256': '0' * 64})
        self.assertEqual(bad.status_code, 422)
        good = self.client.post(f'/api/v1/edge/packages/{package["id"]}/import', json=observation)
        self.assertEqual(good.status_code, 201, good.text)
        self.assertEqual(good.json()['source'], 'imported_device_report')
        self.assertAlmostEqual(good.json()['latency_p50_ms'], .1)
        self.assertTrue(good.json()['within_tolerance'])
        secret = 'test_secret_not_a_real_key'
        with patch('app.services.edge_impulse.start_profile', return_value=123) as submit:
            body = {'artifact_index': 0, 'project_id': 7, 'device': 'test-target', 'api_key': secret, 'consent_upload': True}
            job = self.client.post(f'/api/v1/runs/{run["id"]}/edge/impulse', json=body)
            self.assertEqual(job.status_code, 201, job.text)
            self.assertEqual(submit.call_args.args[-1], secret)
        with patch('app.services.edge_impulse.profile_result', return_value={'hasPerformance': True, 'timePerInferenceMs': 900}):
            result = self.client.post(f'/api/v1/edge/impulse/{job.json()["id"]}/refresh', json={'api_key': secret})
            self.assertEqual(result.status_code, 200, result.text)
        final = self.client.get(f'/api/v1/runs/{run["id"]}').json()['report']
        self.assertEqual(len(final['edge_results']), 2)
        self.assertEqual(final['edge_results'][1]['measurement_scope'], 'provider_analysis')
        self.assertNotIn(secret, json.dumps(self.app.state.repository.edge_records(run['id'])))
        self.assertEqual(final['edge_results'][0]['latency_p50_ms'], .1)

    def test_03_overlap_rejected_and_no_report(self):
        model = self.upload(self.pt2.read_bytes())
        dataset = self.dataset(10)
        response = self.client.post('/api/v1/runs', json={'model_id': model, 'format': 'onnx', 'target': 'esp32', 'dataset_id': dataset,
                                                        'strategy': 'fidelity_search', 'calibration_dataset_id': dataset})
        self.assertEqual(response.status_code, 422)

    def test_04_int8_tflite_input_quantization_and_firmware(self):
        model = self.upload(tiny_tflite(quantized=True), 'tflite')
        run = self.run_model(model, self.dataset(9), 'tflite')
        self.assertEqual(len(run['report']['predictions']), 2)
        result = self.client.post(f'/api/v1/runs/{run["id"]}/edge/packages', json={'artifact_index': 0})
        self.assertEqual(result.status_code, 201, result.text)
        self.assertEqual(result.json()['input_size_bytes'], 12)


class DeveloperUnitTests(unittest.TestCase):
    def test_device_tolerance_uses_reference_scale(self):
        package = {'id': 'pkg_' + 'a'*32, 'model_sha256': 'b'*64, 'input_sha256': 'c'*64, 'warmup_runs': 1,
                   'arena_capacity_bytes': 1024, 'measured_runs': 3, 'expected_output': [1., 0.], 'profile': 'imported',
                   'artifact_index': 0, 'model_size_bytes': 100, 'atol': 0, 'rtol': .5}
        observation = {'protocol': 'edgelens.esp32.v1', 'package_id': package['id'], 'model_sha256': package['model_sha256'],
                       'input_sha256': package['input_sha256'], 'chip': 'esp32', 'idf_version': 'fixture', 'cpu_freq_mhz': 240,
                       'arena_used_bytes': 512, 'arena_capacity_bytes': 1024, 'warmup_runs': 1, 'samples_us': [1, 2, 3], 'output': [2., 0.]}
        result = validate_observation(observation, package, 'imported_device_report')
        self.assertFalse(result['within_tolerance'])

    def test_trust_and_signature_required(self):
        for changes in ({'trusted_source': False}, {'input_shape': [2, 3, 2, 2]}, {'std': [0, 1, 1]}, {'class_count': 1}):
            with self.assertRaises(ValueError):
                ModelSpec.model_validate({'name': 'test', 'format': 'pt2', 'input_shape': [1, 3, 2, 2], 'class_count': 2, 'trusted_source': True} | changes)

    def test_fidelity_ties_prefer_first_and_never_use_accuracy(self):
        standard = {'tolerance_failure_count': 0, 'output_max_abs': .1, 'output_mae': .01, 'accuracy_pct': 0}
        candidate = standard | {'accuracy_pct': 100}
        self.assertIs(min([standard, candidate], key=rank_fidelity), standard)
        self.assertLess(rank_fidelity(standard | {'output_max_abs': .09}), rank_fidelity(standard))

    def test_overlap_checks_pixels_not_filenames(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError, 'overlap'):
            disjoint_inputs([np.zeros((1, 3, 2, 2))], [np.zeros((1, 3, 2, 2))])

    def test_disabled_model_execution_rejects_upload(self):
        root = Path(__file__).resolve().parents[1] / 'data' / ('policy_' + uuid4().hex)
        try:
            with TestClient(create_app(Settings(environment='test', data_dir=root))) as client:
                self.assertEqual(client.post('/api/v1/models', content=b'x').status_code, 403)
        finally:
            if root.exists(): shutil.rmtree(root)

    def test_migration_preserves_existing_runs_and_normalized_rows(self):
        root = Path(__file__).resolve().parents[1] / 'data' / ('migration_' + uuid4().hex)
        try:
            repo = Repository(root)
            run = repo.create_run({'model_id': 'legacy'})
            repo.update_run(run['id'], 'completed', report={'metrics': [{'profile': 'standard', 'accuracy_pct': 42}], 'layers': [], 'artifacts': []})
            with closing(sqlite3.connect(repo.db_path)) as connection, connection:
                connection.execute('DROP TABLE custom_models'); connection.execute('DROP TABLE edge_records'); connection.execute('PRAGMA user_version=1')
            migrated = Repository(root)
            self.assertEqual(migrated.get_run(run['id'])['report']['metrics'][0]['accuracy_pct'], 42)
            self.assertEqual(migrated.list_models(), [])
            with closing(sqlite3.connect(repo.db_path)) as connection, connection:
                self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 2)
                self.assertEqual(connection.execute('SELECT accuracy_pct FROM run_metrics').fetchone()[0], 42)
                connection.execute('PRAGMA user_version=3')
            with self.assertRaisesRegex(RuntimeError, 'newer EdgeLens'):
                Repository(root)
        finally:
            if root.exists(): shutil.rmtree(root)


if __name__ == '__main__': unittest.main()
