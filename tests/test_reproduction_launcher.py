"""Exercise the host launcher with a fake Docker command, without inference."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / 'docker/visual/reproduce.bash'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        docker = self.folder / 'docker'
        docker.write_text('''#!/usr/bin/env bash
case "$1" in
    create) echo test-data-container ;;
    image) echo '{}' ;;
    run) sleep 2; echo 'Final metrics'; exit "${FAKE_DOCKER_EXIT:-0}" ;;
    rm) echo "$*" >> "$CORSAIR_OUTPUT/cleanup.log" ;;
esac
''')
        docker.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.folder}:{os.environ["PATH"]}',
                        CORSAIR_OUTPUT=str(self.folder / 'results'))

    def tearDown(self):
        self.temp.cleanup()

    def test_redirected_output_and_failure_keep_status_and_cleanup(self):
        self.env['FAKE_DOCKER_EXIT'] = '7'
        result = subprocess.run(['bash', str(LAUNCHER)], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7)
        self.assertRegex(result.stderr, r'Elapsed: 00:00:0[2-9]\n')
        self.assertNotIn('\r', result.stderr)
        self.assertNotIn('Results and logs:', result.stdout)
        self.assertFalse((self.folder / 'results/cleanup.log').exists())

    def test_success_prints_total_elapsed(self):
        result = subprocess.run(['bash', str(LAUNCHER)], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Final metrics', result.stdout)
        self.assertIn('Results and logs:', result.stdout)
        self.assertRegex(result.stderr, r'Elapsed: 00:00:0[2-9]\n')


if __name__ == '__main__':
    unittest.main()
