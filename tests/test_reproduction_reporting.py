"""Console summaries must reflect saved metrics and preserve failure diagnostics."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from reproduction.reporting import print_final_metrics, run_logged
from reproduction.reproduce import REFERENCE


class ReportingTests(unittest.TestCase):
    def test_final_metrics_use_exported_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for name in ('table1.csv', 'table2.csv', 'table3.csv'):
                shutil.copyfile(REFERENCE / name, output / name)
            # Prove the summary reads the current run, not reference constants.
            path = output / 'table1.csv'
            path.write_text(path.read_text().replace(',97.0\n', ',96.0\n'))
            console = io.StringIO()
            with redirect_stdout(console):
                print_final_metrics(output)
            lines = console.getvalue().splitlines()
            self.assertEqual(lines, ['CORSAIR + ICP: 94.50%'])

    def test_missing_category_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = (REFERENCE / 'table1.csv').read_text().splitlines()
            (output / 'table1.csv').write_text('\n'.join(row for row in rows if not row.startswith('Table,')))
            with self.assertRaisesRegex(ValueError, 'complete chairs and tables'):
                print_final_metrics(output)

    def test_child_output_is_logged_and_verbose_streams_it(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose), tempfile.TemporaryDirectory() as folder:
                log = Path(folder) / 'evaluation.log'
                console = io.StringIO()
                with redirect_stdout(console):
                    run_logged([sys.executable, '-c', 'import sys; print("progress"); print("diagnostic", file=sys.stderr)'],
                               log, cwd=folder, verbose=verbose)
                self.assertIn('progress', log.read_text())
                self.assertIn('diagnostic', log.read_text())
                self.assertEqual(console.getvalue(), log.read_text() if verbose else '')

    def test_quiet_failure_shows_diagnostics_and_keeps_exit_code(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'evaluation.log'
            console = io.StringIO()
            with redirect_stderr(console), self.assertRaises(subprocess.CalledProcessError) as raised:
                run_logged([sys.executable, '-c', 'import sys; print("registration failed"); sys.exit(7)'],
                           log, cwd=folder)
            self.assertEqual(raised.exception.returncode, 7)
            self.assertIn('registration failed', console.getvalue())
            self.assertIn(str(log), console.getvalue())
            self.assertIn('registration failed', log.read_text())

    def test_nested_progress_reaches_console_without_per_case_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'outer.log'
            inner_log = Path(folder) / 'inner.log'
            worker = ('from reproduction.reporting import emit_progress; import time; '
                      'emit_progress("Features", 0, 2); time.sleep(.15); '
                      'emit_progress("Features", 1, 2); time.sleep(.15); '
                      'emit_progress("Features", 2, 2); '
                      'emit_progress("Registration", 0, 1); '
                      'print("per-case details"); time.sleep(.15); '
                      'emit_progress("Registration", 1, 1)')
            orchestrator = ('from reproduction.reporting import run_logged; '
                            f'run_logged({[sys.executable, "-c", worker]!r}, '
                            f'{str(inner_log)!r}, cwd={str(root)!r})')
            console, progress = io.StringIO(), io.StringIO()
            with redirect_stdout(console), redirect_stderr(progress):
                run_logged([sys.executable, '-c', orchestrator], log, cwd=root, progress=True)
            self.assertEqual(console.getvalue(), '')
            self.assertIn('Features', progress.getvalue())
            self.assertIn('2/2', progress.getvalue())
            self.assertIn('Registration', progress.getvalue())
            self.assertIn('1/1', progress.getvalue())
            self.assertIn('<', progress.getvalue())  # tqdm elapsed < remaining
            self.assertNotIn('CORSAIR_PROGRESS', progress.getvalue())
            self.assertIn('per-case details', inner_log.read_text())


if __name__ == '__main__':
    unittest.main()
