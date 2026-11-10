"""Compare overlap-guarded and RMSE-only selection on identical saved ICP proposals."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from utils.pose_metrics import pose_errors
from utils.pose_refinement import guard_pose_update


def compare(source, target, initial, proposal, saved, truths, metadata):
    # Selection has no access to truths; they are used afterward for scoring.
    old, old_details = guard_pose_update(source, target, initial, proposal, require_overlap=True)
    new, new_details = guard_pose_update(source, target, initial, proposal, require_overlap=False)
    np.testing.assert_allclose(old, saved, rtol=0, atol=0, equal_nan=True)
    assert not old_details['accepted'] or new_details['accepted']
    rows = []
    for policy, selected, details in [('CORSAIR', initial, None),
                                      ('RMSE + overlap', old, old_details),
                                      ('RMSE only', new, new_details)]:
        rre, rte = min((pose_errors(selected, truth) for truth in truths), key=lambda pair: pair[0])
        record = dict(**metadata, policy=policy, rre_deg=rre, rte=rte,
                      accepted=False if details is None else details['accepted'])
        if policy == 'RMSE only':
            record.update(new_details)
        rows.append(record)
    return rows, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-run', type=Path, required=True)
    parser.add_argument('--partial-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'icp-acceptance' /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, transforms = [], {}
    protocol = json.loads((args.pose_run / 'protocol.json').read_text())
    if 'completed_pairs' not in protocol:
        raise ValueError('The pose run must be complete')
    frame = pd.read_csv(args.pose_run / 'per_case.csv')
    for row in frame[frame.method == 'CORSAIR + ICP'].itertuples():
        key = f'table{row.table}/{row.category}/{row.case_index:04d}'
        with np.load(args.pose_run / f'{key}.npz') as arrays:
            result, selected = compare(arrays['source'], arrays['target'], arrays['corsair'],
                arrays['proposal'], arrays['refined'], arrays['truth_variants'],
                dict(dataset=f'Table {row.table}', category=row.category, case_id=key))
        rows.extend(result)
        transforms[key] = selected
    print(f'Checked {len(transforms)} complete-pair cases', flush=True)
    labels = {Path(line.split()[0]).stem: int(line.split()[1]) for line in
              (ROOT / 'configs/03001627_scan2cad_rot_sym_label.txt').read_text().splitlines()}
    frame = pd.read_csv(args.partial_run / 'per_chair.csv')
    with np.load(args.partial_run / 'poses_and_clouds.npz') as arrays:
        for row in frame.itertuples():
            idx, mode = int(row.query_index), row.mode
            prefix = f'{idx}_{mode}'
            truths = []
            for k in range(labels[row.gt_id]):
                angle = 2 * np.pi * k / labels[row.gt_id]
                c, s = np.cos(angle), np.sin(angle)
                symmetry = np.eye(4)
                symmetry[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                truths.append(np.linalg.inv(symmetry) @ np.linalg.inv(arrays[f'{idx}_transform']))
            result, selected = compare(arrays[f'{idx}_observed'], arrays[f'{prefix}_cad'],
                arrays[f'{prefix}_before'], arrays[f'{prefix}_proposal'], arrays[f'{prefix}_after'],
                truths, dict(dataset=f'Partial {mode}', category='chair', case_id=prefix))
            np.testing.assert_allclose([result[1]['rre_deg'], result[1]['rte']],
                                       [row.after_rre_deg, row.after_rte], rtol=1e-8, atol=1e-10)
            rows.extend(result)
            transforms[f'partial/{prefix}'] = selected
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'per_case.csv', index=False)
    np.savez_compressed(args.output / 'selected_poses.npz', **transforms)
    summaries = []
    for (dataset, category, policy), group in frame.groupby(['dataset', 'category', 'policy'], sort=False):
        summaries.append(dict(dataset=dataset, category=category, policy=policy, N=len(group),
            accepted=int(group.accepted.sum()),
            rotation_pass_pct=float(100 * (group.rre_deg <= 5).mean()),
            translation_pass_pct=float(100 * (group.rte <= .02).mean()),
            joint_pass_pct=float(100 * ((group.rre_deg <= 5) & (group.rte <= .02)).mean()),
            median_rre_deg=float(group.rre_deg.median()), median_rte=float(group.rte.median())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / 'comparison.csv', index=False)
    (args.output / 'comparison.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>ICP acceptance comparison</title><style>body{font:16px system-ui;margin:30px}'
        'td,th{padding:8px}table{border-collapse:collapse}</style>'
        '<h1>ICP acceptance comparison</h1><p>Identical saved CORSAIR poses and 10-step ICP proposals. '
        'RMSE only removes the overlap condition. All original selections were verified. '
        'Ground truth is used only for scoring. Success: RRE ≤5°, RTE ≤0.02.</p>'
        + summary.to_html(index=False, float_format=lambda value: f'{value:.4g}'))
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        experiment='Saved-proposal ablation; no inference or ICP rerun',
        pose_run=str(args.pose_run.resolve()), partial_run=str(args.partial_run.resolve()),
        verified_cases=len(transforms), icp_iterations=protocol['icp_iterations'],
        selection='Accept iff full-source nearest-neighbor RMSE decreases by more than 1e-12; no overlap condition',
        ground_truth_use='Scoring only'), indent=2))
    print(summary.to_string(index=False), flush=True)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
