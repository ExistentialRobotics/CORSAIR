"""Run Table 1 by default; start the optional browser with `serve`."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'serve':
        command = [sys.executable, '-u', str(ROOT / 'docker/visual/server.py'), *args[1:]]
    else:
        if not any(arg.partition('=')[0] in ('--output', '--check-only') for arg in args):
            results = Path(os.environ.get('CORSAIR_RESULTS', '/results'))
            name = datetime.now(timezone.utc).strftime('table1-%Y%m%dT%H%M%S%fZ')
            args += ['--output', str(results / name)]
        command = [sys.executable, '-u', str(ROOT / 'reproduction/reproduce.py'), *args]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
