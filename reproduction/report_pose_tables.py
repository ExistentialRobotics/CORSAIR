"""Collect completed pose runs, verify saved poses, and render separate tables."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from utils.pose_metrics import pose_errors
from utils.pose_refinement import guard_pose_update
from reproduction.run_pose_tables import write_tables


def plot_examples(frame, output, number, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 9))
    for row_index, category in enumerate(['chair', 'table']):
        candidates = frame[(frame.table == number) & (frame.category == category)
                           & (frame.method == 'CORSAIR + ICP')].sort_values('rre_deg')
        row = candidates.iloc[-1 if selection == 'worst' else len(candidates) // 2]
        with np.load(row.archive) as arrays:
            source, target = arrays['source'][::5], arrays['target'][::5]
            for column, method in enumerate(['input', 'corsair', 'refined']):
                ax = fig.add_subplot(2, 3, row_index * 3 + column + 1, projection='3d')
                if method == 'input':
                    points = target if number == 1 else source
                    title = f'{category.title()} input'
                else:
                    transform = arrays[method]
                    points = source @ transform[:3, :3].T + transform[:3, 3]
                    ax.scatter(target[:, 0], target[:, 2], target[:, 1], s=1, c='#aaaaaa', alpha=.3)
                    metric_method = 'CORSAIR' if method == 'corsair' else 'CORSAIR + ICP'
                    metric = frame[(frame.table == number) & (frame.category == category)
                                   & (frame.case_index == row.case_index) & (frame.method == metric_method)].iloc[0]
                    title = f'{metric_method}\nRRE {metric.rre_deg:.3g}° · RTE {metric.rte:.3g}'
                finite = np.isfinite(points).all(axis=1)
                points = points[finite]
                ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=2,
                           c='#36878e' if method == 'refined' else '#bd7654', alpha=.7)
                ax.set_title(title, fontsize=11)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=20, azim=45)
                ax.set_axis_off()
    protocol = 'full noiseless ShapeNet pairs' if number == 1 else 'real partial Scan2CAD scans'
    fig.suptitle(f'Table {number}: {protocol}\n{selection.title()} final rotation-error case per category; gray = target', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .92))
    path = output / f'table{number}-{selection}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def collect(runs, output, expected=100):
    output.mkdir(parents=True, exist_ok=True)
    frames, protocols = [], []
    for run in runs:
        run = Path(run).resolve()
        protocol = json.loads((run / 'protocol.json').read_text())
        if 'completed_pairs' not in protocol:
            raise ValueError(f'Run has not finished: {run}')
        protocols.append(dict(source=os.path.relpath(run, output), protocol=protocol))
        frame = pd.read_csv(run / 'per_case.csv')
        frame['archive'] = [str(run / f'table{row.table}' / row.category / f'{row.case_index:04d}.npz')
                            for row in frame.itertuples()]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['table', 'category', 'case_index', 'method']).any():
        raise ValueError('Duplicate cases across runs')
    groups = frame.groupby(['table', 'category', 'method']).size()
    numbers = sorted(frame.table.unique())
    expected_groups = 6 * len(numbers)
    if len(groups) != expected_groups or not (groups == expected).all():
        raise ValueError(f'Expected {expected} cases in each of {expected_groups} groups; got {groups.to_dict()}')
    verified = 0
    for (_, _, _), group in frame.groupby(['table', 'category', 'case_index']):
        with np.load(group.iloc[0].archive) as arrays:
            selected, details = guard_pose_update(arrays['source'], arrays['target'], arrays['corsair'], arrays['proposal'])
            np.testing.assert_allclose(selected, arrays['refined'], rtol=0, atol=0, equal_nan=True)
            for row in group.itertuples():
                key = {'RANSAC': 'ransac', 'CORSAIR': 'corsair', 'CORSAIR + ICP': 'refined'}[row.method]
                transform = arrays[key]
                if np.isfinite(transform).all():
                    errors = min((pose_errors(transform, truth) for truth in arrays['truth_variants']), key=lambda pair: pair[0])
                    np.testing.assert_allclose(errors, [row.rre_deg, row.rte], rtol=1e-8, atol=1e-10)
                else:
                    assert np.isinf(row.rre_deg) and np.isinf(row.rte)
            guarded_row = group[group.method == 'CORSAIR + ICP'].iloc[0]
            assert bool(guarded_row.guard_accepted) == details['accepted']
        verified += 1
    exported = frame.copy()
    exported['archive'] = [os.path.relpath(path, output) for path in frame.archive]
    write_tables(exported.to_dict('records'), output)
    (output / 'audit.json').write_text(json.dumps(dict(verified_pairs=verified,
        protocol_sources=protocols, per_group_counts={str(key): int(value) for key, value in groups.items()},
        registration_failures=int(frame[frame.method == 'CORSAIR'].error.notna().sum()),
        accepted_icp_updates=int(frame[frame.method == 'CORSAIR + ICP'].guard_accepted.sum())), indent=2))
    pictures = [plot_examples(frame, output, table, selection) for table in numbers for selection in ['median', 'worst']]
    with (output / 'tables.html').open('a') as stream:
        stream.write('<h2>Examples</h2>' + ''.join(f'<p><img style="max-width:100%" src="{name}"></p>' for name in pictures))
    print('Joint achievement = RRE≤5° AND RTE≤0.02 on the same case; target 50%.', flush=True)
    for number in numbers:
        print(f'\nTable {number} (percentages)\n' + pd.read_csv(output / f'table{number}.csv').to_string(index=False), flush=True)
    print(f'Verified {verified} pairs. Tables and examples: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pose-tables' /
                        datetime.now(timezone.utc).strftime('report-%Y%m%dT%H%M%SZ'))
    parser.add_argument('--expected', type=int, default=100)
    args = parser.parse_args()
    collect(args.runs, args.output, args.expected)
