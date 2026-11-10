"""Viewer archive compatibility and persisted status tests; no inference needed."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('visual_server', Path(__file__).resolve().parents[1] / 'docker/visual/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.status('shape')

    def tearDown(self):
        self.temp.cleanup()

    def status(self, mode, state='complete'):
        (self.path / 'status.json').write_text(json.dumps(dict(id='20261105T000000Z', status=state, mode=mode)))

    def csv(self, path, rows):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_shape_report_archive_and_transform(self):
        folder = self.path / 'evaluation' / 'timestamp'
        archive = folder / 'table1/chair/0000.npz'
        archive.parent.mkdir(parents=True)
        source = np.array([[0., 0., 0.], [1., 0., 0.]])
        pose = np.eye(4)
        pose[0, 3] = 2
        np.savez(archive, source=source, target=source + [2, 0, 0], ransac=pose, corsair=pose, refined=pose)
        # Audited reports already contain an archive column.
        self.csv(folder / 'per_case.csv', [dict(method='CORSAIR + ICP', category='chair', query_id='chair.npy',
                 table=1, case_index=0, archive='table1/chair/0000.npz')])
        payload = server.cloud_payload(self.path, 0)
        np.testing.assert_allclose(payload['stages']['CORSAIR + ICP'], source + [2, 0, 0])
        self.assertEqual(len(server.cases(self.path)), 1)
        with self.assertRaises(ValueError):
            server.cloud_payload(self.path, -1)

    def test_partial_archive_uses_observation_as_source(self):
        self.status('complete')
        source = np.array([[1., 2., 3.]])
        np.savez(self.path / 'partial.npz', **{'7_observed': source, '7_retrieved_cad': source,
                 '7_retrieved_ransac': np.eye(4), '7_retrieved_before': np.eye(4), '7_retrieved_after': np.eye(4)})
        for table in (1, 2, 3):
            self.csv(self.path / f'table{table}-per_case.csv', [dict(method='CORSAIR + ICP', category='chair',
                     query_index=7, archive='partial.npz', archive_prefix='7_retrieved')])
        np.testing.assert_allclose(server.cloud_payload(self.path, 2)['stages']['Input'], source)

    def test_restart_marks_running_as_interrupted(self):
        self.status('shape', 'running')
        self.assertEqual(server.read_status(self.path)['status'], 'interrupted')
        self.assertEqual(server.cases(self.path), [])

    def test_full_run_retains_results_and_records_reference_mismatch(self):
        (self.path / 'status.json').write_text(json.dumps(dict(
            id='20261105T000000Z', mode='complete', models=100, status='running')))
        with patch.object(server.subprocess, 'call', side_effect=[0, 1]):
            server.evaluate(self.path, ['evaluation-command'])
        status = server.read_status(self.path)
        self.assertEqual(status['status'], 'complete')
        self.assertFalse(status['reference_match'])


if __name__ == '__main__':
    unittest.main()
