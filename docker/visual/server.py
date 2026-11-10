"""Local CORSAIR evaluation UI. Runs one evaluation at a time; no GUI required."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get('CORSAIR_RESULTS', '/results')).resolve()
LOCK = threading.Lock()
ACTIVE = None


def run_path(name):
    if not re.fullmatch(r'[0-9TZ-]+', name):
        raise ValueError('Invalid run ID')
    path = RESULTS / name
    if not path.is_dir():
        raise ValueError('Unknown run')
    return path


def read_status(path):
    status = json.loads((path / 'status.json').read_text())
    if status['status'] == 'running' and status['id'] != ACTIVE:
        status['status'] = 'interrupted'
    return status


def cases(path):
    status = read_status(path)
    if status['status'] != 'complete':
        return []
    records = []
    if status['mode'] == 'complete':
        for table in (1, 2, 3):
            with (path / f'table{table}-per_case.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row['method'] == 'CORSAIR + ICP':
                    records.append(dict(label=f'Table {table} · {row["category"]} · ' + row.get('query_id', row.get('query_index', '')),
                                        **row))
    else:
        folder, = (path / 'evaluation').iterdir()
        with (folder / 'per_case.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['method'] == 'CORSAIR + ICP':
                    archive = folder / f'table{row["table"]}' / row['category'] / f'{int(row["case_index"]):04d}.npz'
                    records.append(row | dict(label=f'{row["category"]} · {row["query_id"]}',
                                              archive=str(archive.relative_to(path))))
    return records


def cloud_payload(path, index):
    rows = cases(path)
    if index < 0 or index >= len(rows):
        raise ValueError('Unknown case')
    row = rows[index]
    archive = (path / row['archive']).resolve()
    if not archive.is_relative_to(path):
        raise ValueError('Invalid archive path')
    with np.load(archive, allow_pickle=False) as arrays:
        prefix = row.get('archive_prefix')
        if prefix:
            idx = row['query_index']
            source, target = arrays[f'{idx}_observed'], arrays[f'{prefix}_cad']
            transforms = {name: arrays[f'{prefix}_{key}'] for name, key in
                          [('RANSAC', 'ransac'), ('CORSAIR', 'before'), ('CORSAIR + ICP', 'after')]}
        else:
            source, target = arrays['source'], arrays['target']
            transforms = {name: arrays[key] for name, key in
                          [('RANSAC', 'ransac'), ('CORSAIR', 'corsair'), ('CORSAIR + ICP', 'refined')]}
        source = source[::max(1, int(np.ceil(len(source) / 3500)))].copy()
        target = target[::max(1, int(np.ceil(len(target) / 3500)))].copy()
        stages = {'Input': source}
        stages.update({name: source @ pose[:3, :3].T + pose[:3, 3] for name, pose in transforms.items()})
    if not all(np.isfinite(x).all() for x in [target, *stages.values()]):
        raise ValueError('This case has a failed registration; see the report')
    return dict(target=target.tolist(), stages={name: points.tolist() for name, points in stages.items()}, metrics=row)


def evaluate(path, command):
    global ACTIVE
    try:
        with (path / 'run.log').open('w') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status = json.loads((path / 'status.json').read_text())
        status.update(status='complete' if code == 0 else 'failed', exit_code=code)
        if code == 0 and status['mode'] == 'complete' and status['models'] == 100:
            with (path / 'run.log').open('a') as log:
                check = subprocess.call([sys.executable, '-u', str(ROOT / 'reproduction/reproduce.py'),
                                         '--check-only', str(path)], cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT)
            status['reference_match'] = check == 0
    except Exception as error:
        status = json.loads((path / 'status.json').read_text())
        status.update(status='failed', error=str(error))
    finally:
        with LOCK:
            (path / 'status.json').write_text(json.dumps(status))
            ACTIVE = None


class Handler(SimpleHTTPRequestHandler):
    def json(self, payload, status=200):
        data = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parts = unquote(urlsplit(self.path).path).strip('/').split('/')
        try:
            if parts == ['api', 'runs']:
                return self.json([read_status(p) for p in sorted(RESULTS.iterdir(), reverse=True)
                                  if p.is_dir() and (p / 'status.json').exists()])
            if len(parts) >= 3 and parts[:2] == ['api', 'runs']:
                path = run_path(parts[2])
                if len(parts) == 4 and parts[3] == 'log':
                    log = path / 'run.log'
                    if not log.exists():
                        return self.json(dict(log=''))
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size - 48000))
                        return self.json(dict(log=stream.read().decode(errors='replace')))
                if len(parts) == 4 and parts[3] == 'cases':
                    return self.json(cases(path))
                if len(parts) == 5 and parts[3] == 'cloud':
                    return self.json(cloud_payload(path, int(parts[4])))
            if parts[0] == 'results':
                file = (RESULTS / '/'.join(parts[1:])).resolve()
                if not file.is_relative_to(RESULTS) or not file.is_file():
                    return self.send_error(404)
                self.path = '/' + str(file.relative_to(RESULTS))
                old = self.directory
                self.directory = str(RESULTS)
                try:
                    return super().do_GET()
                finally:
                    self.directory = old
            if parts not in [[''], ['index.html']]:
                return self.send_error(404)
            return super().do_GET()
        except (ValueError, OSError, KeyError) as error:
            return self.json(dict(error=str(error)), 400)

    def do_POST(self):
        global ACTIVE
        # The UI is local; reject cross-origin requests from other web pages.
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.json(dict(error='Origin rejected'), 403)
        if self.path != '/api/runs':
            return self.send_error(404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            mode = request.get('mode', 'shape')
            if mode not in ('shape', 'complete'):
                raise ValueError('Unknown mode')
            values = {key: request.get(key, default) for key, default in
                      [('models', 1), ('workers', 2), ('iterations', 10)]}
            for key, value in values.items():
                if type(value) is not int or not 1 <= value <= (32 if key == 'workers' else 100):
                    raise ValueError(f'Invalid {key}')
            if mode == 'complete' and not all((Path('/data') / name).exists() for name in
                    ['Scan2CAD_pc', 'Scan2CAD_annotations', 'ShapeNetCore.v2.PC15k']):
                raise ValueError('Full evaluation needs the three dataset volumes mounted under /data. See docker/visual/README.md.')
            with LOCK:
                if ACTIVE is not None:
                    return self.json(dict(error='An evaluation is already running'), 409)
                name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                path = RESULTS / name
                path.mkdir()
                status = dict(id=name, status='running', mode=mode, **values)
                (path / 'status.json').write_text(json.dumps(status))
                script = 'run_complete.py' if mode == 'complete' else 'run_pose_tables.py'
                command = [sys.executable, '-u', str(ROOT / 'reproduction' / script),
                           '--output', str(path if mode == 'complete' else path / 'evaluation'),
                           '--models', str(values['models']), '--workers', str(values['workers']),
                           '--icp-iterations', str(values['iterations'])]
                if mode == 'shape':
                    command += ['--tables', '1']
                ACTIVE = name
                threading.Thread(target=evaluate, args=(path, command), daemon=True).start()
            return self.json(status, 202)
        except (ValueError, OSError) as error:
            return self.json(dict(error=str(error)), 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('0.0.0.0', args.port), partial(Handler, directory=str(Path(__file__).parent)))
    print(f'CORSAIR + ICP visualization: http://localhost:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
