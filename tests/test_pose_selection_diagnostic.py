"""Check that oracle analysis sees raw proposals without changing selection."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from reproduction.diagnose_pose_selection import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_rejected_proposal_and_metric_tradeoff(self):
        rows = []
        for case, initial, selected, proposal in [
            ('rejected', (6., .03), (6., .03), (2., .01)),
            ('tradeoff', (1., .03), (2., .01), (2., .01)),
        ]:
            for method, errors in [('CORSAIR', initial), ('CORSAIR + ICP', selected)]:
                rows.append(dict(dataset='test', category='chair', case_id=case, method=method,
                    rre_deg=errors[0], rte=errors[1], proposal_rre_deg=proposal[0], proposal_rte=proposal[1]))
        original = pd.DataFrame(rows)
        frame = original.copy(deep=True)
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            summary = diagnose(frame, Path(folder)).set_index('oracle_metric')
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(summary.loc['rre_deg', 'icp_better'], 1)
        self.assertEqual(summary.loc['rre_deg', 'corsair_better'], 1)
        self.assertEqual(summary.loc['rre_deg', 'oracle_joint_pct'], 50.)
        self.assertEqual(summary.loc['rte', 'oracle_joint_pct'], 100.)
        self.assertEqual(summary.loc['rte', 'rmse_selection_joint_pct'], 50.)


if __name__ == '__main__':
    unittest.main()
