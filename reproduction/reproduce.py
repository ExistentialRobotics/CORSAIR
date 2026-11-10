"""Evaluate 100 complete chairs and tables and verify against the reference.

    python reproduction/reproduce.py --output /results/reproduction --workers 12
    python reproduction/reproduce.py --check-only /results/existing-run

Reference CSVs are historical observations, never inputs to inference or ICP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.reporting import print_final_metrics, run_logged

REFERENCE = Path(__file__).with_name('reference_tables')


def compare(output, verbose=False, quiet=False):
    provenance = json.loads((REFERENCE / 'provenance.json').read_text())
    for name, expected in provenance['reference_sha256'].items():
        if hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reference checksum mismatch: {name}')
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest['arguments']['models'] != 100:
        raise ValueError('Verification requires a completed 100-model full evaluation')
    differences, checked = [], {}
    tables = [1] if manifest['arguments'].get('complete_only', False) else [1, 2, 3]
    for table in tables:
        name = f'table{table}'
        for suffix, keys, fields in [
            ('.csv', ['Category', 'Method'], None),
            ('-per_case.csv', ['category', 'case_index' if table == 1 else 'query_index', 'method'],
             ['rre_deg', 'rte', 'guard_accepted'] +
             (['query_id', 'cad_id'] if table == 1 else ['file', 'gt_id', 'retrieved_id']))]:
            file = name + suffix
            actual = pd.read_csv(output / file)
            expected = pd.read_csv(REFERENCE / file)
            if actual.duplicated(keys).any():
                differences.append(dict(file=file, error='Duplicate case/method keys'))
                continue
            actual, expected = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
            if not actual.index.equals(expected.index):
                differences.append(dict(file=file, error='Category/method/query selections differ'))
                continue
            fields = fields or list(expected.columns)
            for field in fields:
                if field not in actual:
                    differences.append(dict(file=file, field=field, error='Missing column'))
                    continue
                a, b = actual[field], expected[field]
                if pd.api.types.is_numeric_dtype(b) and field != 'guard_accepted':
                    tolerance = 1e-8 if suffix != '.csv' else 0
                    match = np.isclose(a, b, rtol=0, atol=tolerance, equal_nan=True)
                else:
                    match = a.fillna('').eq(b.fillna('')).to_numpy()
                for index in np.flatnonzero(~match):
                    differences.append(dict(file=file, case=str(expected.index[index]), field=field,
                                            expected=str(b.iloc[index]), actual=str(a.iloc[index])))
            checked[file] = len(expected)
    result = dict(passed=not differences, reference_run=provenance['reference_run'],
                  evaluation_tables=tables,
                  rows_checked=checked, per_case_absolute_tolerance=1e-8,
                  summary_absolute_tolerance=0, differences=differences)
    (output / 'reproduction-check.json').write_text(json.dumps(result, indent=2) + '\n')
    if verbose:
        print(json.dumps(result, indent=2), flush=True)
    elif not quiet or not result['passed']:
        status = 'PASSED' if result['passed'] else f'FAILED ({len(differences)} differences)'
        print(f'Reference check: {status}. Details: {output / "reproduction-check.json"}', flush=True)
    return result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--check-only', type=Path)
    parser.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument('--verbose', action='store_true', help='Show per-case progress, detailed tables, and verification JSON')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    output = (args.check_only or args.output).resolve()
    if args.output:
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CORSAIR_DEDUP_CORRESPONDENCES='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CORSAIR_FORWARD_PROGRESS='1')
        run_logged([sys.executable, '-u', str(ROOT / 'reproduction/run_complete.py'),
                        '--output', str(output), '--complete-only', '--models', '100', '--workers', str(args.workers),
                        '--icp-iterations', '10', *(['--verbose'] if args.verbose else [])],
                   output / 'reproduction.log', cwd=ROOT, env=env, verbose=args.verbose, progress=True)
    passed = compare(output, verbose=args.verbose, quiet=not args.verbose)
    if passed:
        print_final_metrics(output)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
