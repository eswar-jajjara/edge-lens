"""Deleting a finished test cannot erase uploaded inputs or active work."""
import sqlite3
from contextlib import closing
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from test_edge_evidence import workspace_temp, fixture


class HistoryDeleteTests(unittest.TestCase):
    def setUp(self):
        self.temp = workspace_temp(); self.root = self.temp.__enter__()
        self.context = TestClient(create_app(Settings(environment='test', data_dir=self.root, allow_custom_models=True)))
        self.client = self.context.__enter__(); self.repo = self.client.app.state.repository
        self.run = self.repo.create_run({'model_id': 'fixture'})
        self.directory = self.root / 'runs' / self.run['id']; self.directory.mkdir(parents=True)
        (self.directory / 'artifact.tflite').write_bytes(b'artifact')
        self.repo.update_run(self.run['id'], 'completed', report=fixture())
        self.repo.save_edge_record(self.run['id'], 'edge_impulse_result', {'fixture': True})
        self.endpoint = '/api/v1/runs/' + self.run['id']

    def tearDown(self):
        self.context.__exit__(None, None, None); self.temp.__exit__(None, None, None)

    def test_deletion_cascades_report_rows_and_keeps_inputs_and_other_tests(self):
        model = {'id': 'model_keep', 'created_at': '2026-10-02'}; self.repo.save_model(model)
        dataset = {'id': 'ds_keep', 'name': 'Keep images', 'image_count': 2, 'class_count': 2, 'path': str(self.root / 'images.zip')}
        self.repo.save_dataset(dataset)
        other = self.repo.create_run({'model_id': 'fixture'}); self.repo.update_run(other['id'], 'failed')
        response = self.client.delete(self.endpoint)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['artifacts_removed']); self.assertFalse(self.directory.exists())
        self.assertIsNone(self.repo.get_run(self.run['id']))
        self.assertEqual(self.repo.get_model('model_keep'), model)
        self.assertIsNotNone(self.repo.get_dataset('ds_keep')); self.assertIsNotNone(self.repo.get_run(other['id']))
        with closing(sqlite3.connect(self.repo.db_path)) as connection, connection:
            for table in ('run_metrics', 'layer_results', 'model_artifacts', 'edge_records', 'run_candidates', 'run_datasets', 'sensitivity_results', 'deployment_selections'):
                self.assertEqual(connection.execute(f'SELECT COUNT(*) FROM {table} WHERE run_id=?', (self.run['id'],)).fetchone()[0], 0)
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertEqual(self.client.get(self.endpoint + '/report.html').status_code, 404)
        self.assertEqual(self.client.delete(self.endpoint).status_code, 404)

    def test_queued_and_running_jobs_are_kept(self):
        for status in ('queued', 'running'):
            self.repo.update_run(self.run['id'], status)
            self.assertEqual(self.client.delete(self.endpoint).status_code, 409)
            self.assertTrue(self.directory.exists()); self.assertIsNotNone(self.repo.get_run(self.run['id']))

    def test_files_are_restored_when_the_database_delete_fails(self):
        with closing(sqlite3.connect(self.repo.db_path)) as connection, connection:
            connection.executescript("CREATE TRIGGER refuse_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'test rollback'); END;")
        with self.assertRaises(sqlite3.IntegrityError): self.repo.delete_run(self.run['id'])
        self.assertTrue((self.directory / 'artifact.tflite').is_file()); self.assertIsNotNone(self.repo.get_run(self.run['id']))

    def test_path_escape_and_remote_workspace_deletion_are_rejected(self):
        with self.assertRaises(KeyError): self.repo.delete_run('../models')
        self.client.app.state.config = Settings(environment='test', data_dir=self.root, allow_custom_models=False)
        self.assertEqual(self.client.delete(self.endpoint).status_code, 403)
        self.assertTrue(self.directory.exists())


if __name__ == '__main__': unittest.main()
