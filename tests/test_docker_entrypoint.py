"""The plain image invocation must run Table 1, with the UI opt-in."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('docker_entrypoint', ROOT / 'docker/visual/entrypoint.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


class EntrypointTests(unittest.TestCase):
    def command(self, args):
        with patch.object(entrypoint.os, 'execv') as execute:
            entrypoint.main(args)
        executable, command = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        return command

    def test_default_runs_reproduction_in_unique_results_directory(self):
        with patch.dict(entrypoint.os.environ, CORSAIR_RESULTS='/results'):
            command = self.command([])
        self.assertEqual(command[:3], [sys.executable, '-u', str(ROOT / 'reproduction/reproduce.py')])
        self.assertEqual(command[3], '--output')
        self.assertTrue(command[4].startswith('/results/table1-'))

    def test_evaluation_arguments_are_forwarded(self):
        command = self.command(['--workers', '12', '--verbose', '--output=/results/custom'])
        self.assertEqual(command[3:], ['--workers', '12', '--verbose', '--output=/results/custom'])

    def test_check_only_does_not_add_an_output(self):
        command = self.command(['--check-only', '/results/existing'])
        self.assertEqual(command[3:], ['--check-only', '/results/existing'])

    def test_browser_requires_serve(self):
        command = self.command(['serve', '--port', '8081'])
        self.assertEqual(command, [sys.executable, '-u', str(ROOT / 'docker/visual/server.py'), '--port', '8081'])


if __name__ == '__main__':
    unittest.main()
