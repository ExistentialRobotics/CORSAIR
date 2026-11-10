"""A reproduction check must detect changed results and changed case selection."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from reproduction.reproduce import compare, REFERENCE


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        for path in REFERENCE.glob('*.csv'):
            shutil.copyfile(path, self.output / path.name)
        (self.output / 'manifest.json').write_text(json.dumps(dict(status='complete', arguments=dict(models=100))))

    def tearDown(self):
        self.temp.cleanup()

    def test_reference_matches_with_reordered_rows(self):
        path = self.output / 'table1-per_case.csv'
        rows = pd.read_csv(path).iloc[::-1]
        rows.to_csv(path, index=False)
        self.assertTrue(compare(self.output))

    def test_changed_metric_and_retrieved_id_fail(self):
        path = self.output / 'table3-per_case.csv'
        rows = pd.read_csv(path)
        rows.loc[0, 'rre_deg'] += .1
        rows.loc[0, 'retrieved_id'] = 'different-cad'
        rows.to_csv(path, index=False)
        self.assertFalse(compare(self.output))
        report = json.loads((self.output / 'reproduction-check.json').read_text())
        self.assertEqual({row['field'] for row in report['differences']}, {'rre_deg', 'retrieved_id'})

    def test_subset_is_not_full_reproduction(self):
        (self.output / 'manifest.json').write_text(json.dumps(dict(status='complete', arguments=dict(models=1))))
        with self.assertRaisesRegex(ValueError, '100-model'):
            compare(self.output)

    def test_complete_only_verifies_chairs_and_tables_without_partial_files(self):
        (self.output / 'manifest.json').write_text(json.dumps(dict(
            status='complete', arguments=dict(models=100, complete_only=True))))
        for table in (2, 3):
            for suffix in ('.csv', '-per_case.csv'):
                (self.output / f'table{table}{suffix}').unlink()
        self.assertTrue(compare(self.output, quiet=True))
        report = json.loads((self.output / 'reproduction-check.json').read_text())
        self.assertEqual(report['evaluation_tables'], [1])
        self.assertEqual(set(report['rows_checked']), {'table1.csv', 'table1-per_case.csv'})
        path = self.output / 'table1-per_case.csv'
        rows = pd.read_csv(path)
        rows.loc[0, 'rre_deg'] += .1
        rows.to_csv(path, index=False)
        self.assertFalse(compare(self.output, quiet=True))


if __name__ == '__main__':
    unittest.main()
