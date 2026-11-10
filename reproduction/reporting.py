"""Concise console results and persistent logs for Docker evaluation commands."""
import csv
import json
from collections import deque
from pathlib import Path
import subprocess
import sys

PROGRESS_PREFIX = 'CORSAIR_PROGRESS '


def emit_progress(stage, completed, total):
    print(PROGRESS_PREFIX + json.dumps(dict(stage=stage, completed=completed, total=total)), flush=True)


class EvaluationProgress:
    """Estimate each phase separately because their task costs differ."""
    def __init__(self):
        self.bar = None

    def update(self, event):
        from tqdm import tqdm
        if self.bar is None or self.bar.desc != event['stage']:
            self.close()
            self.bar = tqdm(total=event['total'], desc=event['stage'], unit='task',
                            dynamic_ncols=True, file=sys.stderr)
        self.bar.update(event['completed'] - self.bar.n)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def run_logged(command, log_path, *, cwd, verbose=False, env=None, progress=False):
    """Save child output, stream it in verbose mode, and surface failures."""
    tail = deque(maxlen=30)
    display = EvaluationProgress()
    try:
        with Path(log_path).open('a') as log:
            with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    if line.startswith(PROGRESS_PREFIX):
                        if progress:
                            display.update(json.loads(line[len(PROGRESS_PREFIX):]))
                        else:
                            # Forward events through the orchestration process.
                            print(line, end='', flush=True)
                    elif verbose:
                        if display.bar is not None:
                            display.bar.write(line.rstrip('\n'), file=sys.stdout)
                        else:
                            print(line, end='', flush=True)
                code = process.wait()
    finally:
        display.close()
    if code:
        if not verbose:
            print(''.join(tail), end='', file=sys.stderr, flush=True)
        print(f'Evaluation failed; full log: {log_path}', file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(code, command)


def combined_icp_accuracy(output):
    """Return the equal-weight mean of complete chair/table ICP joint success."""
    with (Path(output) / 'table1.csv').open(newline='') as source:
        rows = [row for row in csv.DictReader(source) if row['Method'] == 'CORSAIR + ICP']
    if len(rows) != 2 or {row['Category'].lower() for row in rows} != {'chair', 'table'}:
        raise ValueError('Expected one CORSAIR + ICP result each for complete chairs and tables')
    return sum(float(row['Both ≤5°/0.02']) for row in rows) / 2


def print_final_metrics(output):
    print(f'CORSAIR + ICP: {combined_icp_accuracy(output):.2f}%', flush=True)
