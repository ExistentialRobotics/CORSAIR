"""Run fresh inference/registration and export three audited method tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from utils.pose_metrics import pose_errors
from utils.pose_refinement import guard_pose_update
from reproduction.run_partial_chairs import score
from reproduction.reporting import print_final_metrics, combined_icp_accuracy, run_logged, emit_progress

METHODS = [('RANSAC', 'ransac'), ('CORSAIR', 'before'), ('CORSAIR + ICP', 'after')]


def summarize(frame):
    rows = []
    for category in sorted(frame.category.unique()):
        for method, _ in METHODS:
            group = frame[(frame.category == category) & (frame.method == method)]
            if group.empty:
                raise ValueError(f'Missing method {category}/{method}')
            row = dict(Category=category.title(), Method=method, N=len(group))
            for limit in [5, 15, 45]:
                row[f'RRE≤{limit}°'] = round(float(100 * (group.rre_deg <= limit).mean()), 2)
            for limit in [.02, .05, .1]:
                row[f'RTE≤{limit:.2f}'] = round(float(100 * (group.rte <= limit).mean()), 2)
            row['Both ≤5°/0.02'] = round(float(100 * ((group.rre_deg <= 5) & (group.rte <= .02)).mean()), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def report(shape_run, partial_run, output, expected, verbose=False):
    shape = pd.read_csv(shape_run / 'per_case.csv')
    for index, row in shape.iterrows():
        archive = shape_run / f'table{row["table"]}' / row['category'] / f'{int(row["case_index"]):04d}.npz'
        shape.loc[index, 'archive'] = str(archive.relative_to(output))
        if row['method'] == 'CORSAIR + ICP':
            with np.load(archive) as arrays:
                proposal_errors = min((pose_errors(arrays['proposal'], truth) for truth in arrays['truth_variants']),
                                      key=lambda errors: errors[0])
            shape.loc[index, 'proposal_rre_deg'], shape.loc[index, 'proposal_rte'] = proposal_errors
    for category in ['chair', 'table']:
        for method, _ in METHODS:
            assert len(shape[(shape.category == category) & (shape.method == method)]) == expected
    exports = [('table1', 'Complete ShapeNet pairs — correct CAD supplied', shape)]
    verified, accepted = 0, dict(known_pair=0, retrieved=0)
    if partial_run is not None:
        records = []
        frame = pd.read_csv(partial_run / 'per_chair.csv')
        if frame.duplicated(['query_index', 'mode']).any():
            raise ValueError('Duplicate partial query/mode rows')
        if set(frame['mode']) != {'known_pair', 'retrieved'}:
            raise ValueError('Both partial evaluation modes are required')
        known = set(frame.loc[frame['mode'] == 'known_pair', 'query_index'])
        retrieved = set(frame.loc[frame['mode'] == 'retrieved', 'query_index'])
        if known != retrieved or len(known) != expected:
            raise ValueError('Partial evaluation modes must use the same expected query set')
        with np.load(partial_run / 'poses_and_clouds.npz') as arrays:
            for row in frame.itertuples():
                idx, mode = int(row.query_index), row.mode
                prefix = f'{idx}_{mode}'
                selected, details = guard_pose_update(arrays[f'{idx}_observed'], arrays[f'{prefix}_cad'],
                    arrays[f'{prefix}_before'], arrays[f'{prefix}_proposal'], require_overlap=False)
                np.testing.assert_allclose(selected, arrays[f'{prefix}_after'], atol=0, rtol=0)
                assert details['accepted'] == bool(row.guard_accepted)
                accepted[mode] += int(details['accepted'])
                for method, stage in METHODS:
                    rre, rte = score(arrays[f'{prefix}_{stage}'], arrays[f'{idx}_transform'], int(row.truth_symmetry))
                    np.testing.assert_allclose([rre, rte],
                        [getattr(row, f'{stage}_rre_deg'), getattr(row, f'{stage}_rte')], rtol=1e-8, atol=1e-10)
                    records.append(dict(mode=mode, category='chair', method=method, query_index=idx,
                        file=row.file, gt_id=row.gt_id, retrieved_id=row.retrieved_id,
                        archive=str((partial_run / 'poses_and_clouds.npz').relative_to(output)), archive_prefix=prefix,
                        proposal_rre_deg=row.proposal_rre_deg, proposal_rte=row.proposal_rte,
                        rre_deg=rre, rte=rte, guard_accepted=details['accepted'] if stage == 'after' else False))
                verified += 1
        partial = pd.DataFrame(records)
        for number, mode in [(2, 'known_pair'), (3, 'retrieved')]:
            subset = partial[partial['mode'] == mode]
            assert len(subset) == 3 * expected
            title = 'Partial Scan2CAD chairs — ' + ('known CAD' if mode == 'known_pair' else 'top-1 retrieved CAD')
            exports.append((f'table{number}', title, subset))
    html, markdown = [], []
    for name, title, data in exports:
        data = data.copy()
        data['method'] = pd.Categorical(data.method, categories=[method for method, _ in METHODS], ordered=True)
        case_column = 'case_index' if name == 'table1' else 'query_index'
        data = data.sort_values(['category', case_column, 'method'])
        table = summarize(data)
        table.to_csv(output / f'{name}.csv', index=False)
        data.to_csv(output / f'{name}-per_case.csv', index=False)
        html.append(f'<h2>{name.title()}: {title}</h2>{table.to_html(index=False)}')
        columns = list(table.columns)
        text = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
        text += '\n'.join('| ' + ' | '.join(str(value) for value in row) + ' |' for row in table.itertuples(index=False, name=None))
        markdown.append(f'## {name.title()}: {title}\n\n{text}\n')
        if verbose:
            print(f'\n{title}\n{table.to_string(index=False)}', flush=True)
    introduction = ('Fresh inference and registration; no prediction caches. Methods are ordered RANSAC, CORSAIR, CORSAIR + ICP. '
        'ICP uses up to 10 point-to-point iterations by default and accepts a pose only if full-source nearest-neighbor RMSE decreases '
        'by more than 1e-12. There is no overlap condition. GT is used for synthetic observations, upstream Scan2CAD '
        'normalization/segmentation, and scoring; never for accepting ICP. RTE uses unit-radius-normalized coordinates. '
        'All success entries are percentages; joint success requires RRE≤5° and RTE≤0.02 on the same case. '
        'The partial known-CAD and retrieved-CAD tables use the same held-out queries and retrieval checkpoint. '
        'Table 1 uses the released pose checkpoints; see protocol files for hashes and exact selections.')
    if partial_run is None:
        accuracy = combined_icp_accuracy(output)
        introduction = ('Complete ShapeNet chairs and tables, with correct CAD supplied. '
            'Joint success requires RRE≤5° and normalized RTE≤0.02 on the same case. '
            'The final CORSAIR + ICP percentage is the arithmetic mean of chair and table success. '
            'Detailed category and method results are retained below.')
        html = [f'<h2>CORSAIR + ICP: {accuracy:.2f}%</h2><details><summary>Detailed results</summary>'
                + ''.join(html) + '</details>']
        (output / 'summary.json').write_text(json.dumps(dict(method='CORSAIR + ICP',
            joint_success_pct=accuracy, aggregation='arithmetic mean of chair and table percentages',
            models_per_category=expected, categories=['chair', 'table']), indent=2) + '\n')
    (output / 'tables.html').write_text('<!doctype html><meta charset="utf-8"><title>CORSAIR evaluation</title>'
        '<style>body{font:16px system-ui;margin:40px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}</style>'
        '<h1>CORSAIR evaluation</h1><p>' + introduction + '</p>' + ''.join(html))
    (output / 'tables.md').write_text('# CORSAIR evaluation\n\n' + introduction + '\n\n' + '\n'.join(markdown))
    shape_audit = json.loads((shape_run / 'audit.json').read_text())
    audit = dict(shape_pairs_verified=shape_audit['verified_pairs'], partial_registrations_verified=verified,
        partial_icp_accepted=accepted, models_per_category=expected, icp_acceptance='rmse_only',
        shape_run=str(shape_run.relative_to(output)), partial_run=str(partial_run.relative_to(output)) if partial_run is not None else None)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=int, default=100)
    parser.add_argument('--complete-only', action='store_true', help='Evaluate complete chairs and tables only')
    parser.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument('--icp-iterations', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', help='Show per-case progress and detailed tables')
    args = parser.parse_args()
    if min(args.models, args.workers, args.icp_iterations) < 1:
        parser.error('All numeric arguments must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Output already contains a run')
    source_paths = sorted(set(p for folder in ['reproduction', 'utils', 'model', 'datasets', 'configs', 'tests']
        for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts))
    source_paths += [ROOT / 'docker/Dockerfile.pose', ROOT / 'requirements-pose.txt', ROOT / 'README.md', ROOT / '.dockerignore']
    input_paths = sorted((ROOT / 'docker/data/ShapeNetCore.v2.PC15k').glob('*/test/*.npy'))
    input_paths += [ROOT / 'ckpts' / name for name in
                    (['scannet_pose_chair_best', 'scannet_pose_table_best'] +
                     ([] if args.complete_only else ['scannet_ret_chair']))]
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), status='running', arguments=vars(args) | {'output': str(output)},
        command=sys.argv, icp_acceptance='rmse_only', fresh_inference=True,
        evaluation_tables=[1] if args.complete_only else [1, 2, 3],
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        environment={name: os.environ.get(name) for name in
                     ['CORSAIR_DEDUP_CORRESPONDENCES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        python=sys.version, packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    common = ['--models', str(args.models), '--workers', str(args.workers), '--icp-iterations', str(args.icp_iterations)]
    try:
        stages = [('run_pose_tables.py', 'shape', ['--tables', '1'])]
        if not args.complete_only:
            stages.append(('run_partial_chairs.py', 'partial', []))
        for script, folder, extra in stages:
            run_logged([sys.executable, '-u', str(ROOT / 'reproduction' / script),
                '--output', str(output / folder), *extra, *common],
                output / 'evaluation.log', cwd=ROOT, verbose=args.verbose,
                progress=os.environ.get('CORSAIR_FORWARD_PROGRESS') != '1')
        shape_run, = (output / 'shape').iterdir()
        partial_run = None
        if not args.complete_only:
            partial_run, = (output / 'partial').iterdir()
        emit_progress('Audit and export', 0, 2)
        manifest['audit'] = report(shape_run, partial_run, output, args.models, verbose=args.verbose)
        emit_progress('Audit and export', 1, 2)
        if not args.complete_only:
            run_logged([sys.executable, '-u', str(ROOT / 'reproduction/diagnose_pose_selection.py'),
                        '--complete-run', str(output)], output / 'evaluation.log', cwd=ROOT, verbose=args.verbose)
        emit_progress('Audit and export', 2, 2)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['finished'] = datetime.now(timezone.utc).isoformat()
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if not args.verbose:
        print_final_metrics(output)
    if args.verbose:
        print(f'Completed and audited results: {output / "tables.html"}', flush=True)


if __name__ == '__main__':
    main()
