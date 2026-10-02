"""Controlled validation-only search, diagnostics, constraints and migration checks."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.repositories import Repository
from app.schemas.runs import CreateRunRequest
from app.services.deployment_search import select_candidates


def candidate(identifier, accuracy, size, latency, status='completed'):
    return {'id': identifier, 'status': status, 'validation_metrics': {'accuracy_pct': accuracy, 'size_bytes': size,
            'latency_p50_ms': latency}, 'test_metrics': {'accuracy_pct': 1000 if identifier == 'bad' else 0}}


class SearchContractTests(unittest.TestCase):
    def test_constraints_select_validation_evidence_and_ignore_test(self):
        values = [candidate('fp32', 90, 10 * 1048576, 5), candidate('good', 89.5, 3 * 1048576, 3),
                  candidate('bad', 85, 1048576, 1), candidate('missing', 90, 1048576, None)]
        selection = select_candidates(values, {'objective': 'size', 'max_accuracy_loss_pp': 1}, 90)
        self.assertEqual(selection['selected'], 'good')
        self.assertEqual(selection['evidence_split'], 'validation')
        self.assertFalse(selection['test_data_used_for_selection'])
        self.assertFalse(values[2]['eligibility']['eligible'])
        self.assertIn('unavailable', values[3]['eligibility']['reasons'][0])

    def test_no_feasible_candidate_no_arbitrary_winner_and_pareto(self):
        values = [candidate('fp32', 90, 10 * 1048576, 5), candidate('small', 89, 2 * 1048576, 3),
                  candidate('dominated', 88, 3 * 1048576, 4), candidate('failed', 100, 1, 1, 'failed')]
        selection = select_candidates(values, {'objective': 'latency', 'max_size_mib': 1}, 90)
        self.assertIsNone(selection['selected'])
        self.assertEqual(selection['decision'], 'no_feasible_candidate')
        self.assertEqual(set(selection['pareto_candidates']), {'fp32', 'small'})
        self.assertIsNone(select_candidates(values, {'objective': 'tradeoffs'}, 90)['selected'])

    def test_bounded_strict_schema(self):
        base = dict(model_id='model_a', strategy='deployment_search', format='onnx', target='esp32',
                    calibration_dataset_id='cal', validation_dataset_id='val', dataset_id='test')
        self.assertEqual(CreateRunRequest(**base).search.max_candidates, 16)
        for change in ({'search': {'max_candidates': 25}}, {'search': {'diagnostic_samples': 0}},
                       {'search': {'calibration_methods': ['MinMax', 'MinMax']}}, {'dataset_id': 'val'},
                       {'constraints': {'max_size_mib': -1}}, {'constraints': {'objective': 'random'}},
                       {'constraints': {'max_accuracy_loss_pp': float('nan')}}, {'constraints': {'max_device_latency_ms': 5}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                CreateRunRequest(**(base | change))

    def test_upgrade_schema3_preserves_history_and_atomic_search_records(self):
        root = Path(__file__).resolve().parents[1] / 'data' / ('search_migration_' + uuid4().hex)
        root.mkdir(parents=True)
        try:
            repository = Repository(root)
            old = repository.create_run({'old': True})
            repository.update_run(old['id'], 'completed', report={'metrics': [], 'layers': [], 'artifacts': [], 'summary': {'conclusion': 'keep'}})
            with closing(sqlite3.connect(repository.db_path)) as connection:
                connection.execute('DROP TABLE sensitivity_results')
                connection.execute('DROP TABLE deployment_selections')
                connection.execute('PRAGMA user_version=3')
                connection.commit()
            upgraded = Repository(root)
            self.assertEqual(upgraded.get_run(old['id'])['report']['summary']['conclusion'], 'keep')
            run = upgraded.create_run({})
            report = {'experiment_type': 'deployment_optimization', 'selection': {'selected': None, 'objective': 'size',
                'decision': 'no_feasible_candidate', 'constraints': {}}, 'diagnostics': {'status': 'partial'},
                'sensitivity': [{'node': 'conv', 'candidate_id': 'exclude_1', 'parent_candidate': 'static_int8', 'accuracy_recovery_pp': -2}]}
            upgraded.update_run(run['id'], 'completed', report=report)
            with closing(sqlite3.connect(upgraded.db_path)) as connection:
                self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 4)
                self.assertEqual(connection.execute('SELECT accuracy_recovery_pp FROM sensitivity_results').fetchone()[0], -2)
                self.assertIsNone(connection.execute('SELECT selected_candidate FROM deployment_selections').fetchone()[0])
            invalid = dict(report, sensitivity=[{'node': {'invalid': 'SQL type'}}])
            with self.assertRaises(sqlite3.ProgrammingError):
                upgraded.update_run(run['id'], 'failed', report=invalid)
            self.assertEqual(upgraded.get_run(run['id'])['report'], report)
        finally:
            shutil.rmtree(root)


@unittest.skipUnless(os.environ.get('EDGELENS_TEST_CUSTOM') == '1', 'Set EDGELENS_TEST_CUSTOM=1 for actual search and diagnostics')
class SearchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_quantization import PrecisionIntegrationTests
        PrecisionIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        from tests.test_quantization import PrecisionIntegrationTests
        PrecisionIntegrationTests.tearDownClass.__func__(cls)

    def test_zero_probes_keeps_diagnostics_and_the_two_control_candidates(self):
        from tests.test_quantization import PrecisionIntegrationTests
        self.model = lambda *args, **kw: PrecisionIntegrationTests.model(self, *args, **kw)
        self.dataset = lambda *args, **kw: PrecisionIntegrationTests.dataset(self, *args, **kw)
        payload = PrecisionIntegrationTests.payload(self)
        payload.update(strategy='deployment_search', search={'max_candidates': 2,
            'sensitivity_probes': 0, 'diagnostic_samples': 1, 'max_seconds': 120})
        report = PrecisionIntegrationTests.finish(self, payload)['report']
        self.assertEqual(len(report['candidates']), 2)
        self.assertEqual(report['sensitivity'], [])
        self.assertGreater(report['diagnostics']['compared_operations'], 0)
        self.assertEqual(report['selection']['decision'], 'tradeoffs_only')
        self.assertFalse(report['selection']['test_data_used_for_selection'])

    def test_real_search_freezes_before_test_and_retains_operation_controls(self):
        from tests.test_quantization import PrecisionIntegrationTests
        from app.services.deployment_search import select_candidates as selector
        self.model = lambda *args, **kw: PrecisionIntegrationTests.model(self, *args, **kw)
        self.dataset = lambda *args, **kw: PrecisionIntegrationTests.dataset(self, *args, **kw)
        payload = PrecisionIntegrationTests.payload(self)
        payload.update(strategy='deployment_search', search={'max_candidates': 4, 'sensitivity_probes': 1,
            'diagnostic_samples': 2, 'calibration_methods': ['MinMax'], 'max_seconds': 120},
            constraints={'objective': 'size', 'max_accuracy_loss_pp': 0})
        def check_freeze(candidates, constraints, baseline):
            self.assertTrue(all(candidate.get('test_metrics') is None for candidate in candidates))
            return selector(candidates, constraints, baseline)
        with patch('app.services.deployment_search.select_candidates', side_effect=check_freeze):
            run = PrecisionIntegrationTests.finish(self, payload)
        self.assertEqual(run['status'], 'completed', run)
        report = run['report']
        self.assertEqual(report['experiment_type'], 'deployment_optimization')
        self.assertFalse(report['test_data_used_for_selection'])
        self.assertLessEqual(len(report['candidates']), 4)
        self.assertGreater(report['diagnostics']['compared_operations'], 0)
        self.assertEqual(len(report['sensitivity']), 1)
        chosen = report['selection']['selected']
        self.assertIsNotNone(chosen)
        for candidate in report['candidates']:
            if candidate.get('validation_metrics'):
                self.assertEqual(candidate['validation_metrics']['provenance']['latency_p50_ms']['scope'], 'host_cpu')
            if candidate['id'] not in {'fp32', 'static_int8', chosen}:
                self.assertIsNone(candidate['test_metrics'])
        control = next(candidate for candidate in report['candidates'] if candidate['id'].startswith('exclude_'))
        if control['status'] == 'completed':
            self.assertEqual(control['inventory']['excluded_operations'][0]['weight_dtype'], 'FLOAT')
        with closing(sqlite3.connect(self.app.state.repository.db_path)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM sensitivity_results WHERE run_id=?', (run['id'],)).fetchone()[0], 1)
            self.assertEqual(connection.execute('SELECT selected_candidate FROM deployment_selections WHERE run_id=?', (run['id'],)).fetchone()[0], chosen)
        for format in ('html', 'csv'):
            response = self.client.get(f'/api/v1/runs/{run["id"]}/report.{format}')
            self.assertEqual(response.status_code, 200)
            self.assertIn('validation', response.text)
            self.assertNotIn(str(self.root), response.text)


if __name__ == '__main__':
    unittest.main()
