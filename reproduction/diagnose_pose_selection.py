"""Ground-truth oracle diagnostic only; never alters estimated or selected poses."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd


def diagnose(frame, output):
    required = ['dataset', 'category', 'case_id', 'method', 'rre_deg', 'rte']
    columns = required + [name for name in ['proposal_rre_deg', 'proposal_rte'] if name in frame]
    frame = frame[columns]
    if frame.duplicated(['dataset', 'category', 'case_id', 'method']).any():
        raise ValueError('Duplicate case/method rows')
    initial = frame[frame.method == 'CORSAIR'][required].drop(columns='method')
    refined = frame[frame.method == 'CORSAIR + ICP'].drop(columns='method')
    refined = refined.copy()
    refined['selected_rre_deg'], refined['selected_rte'] = refined.rre_deg, refined.rte
    for metric, proposal in [('rre_deg', 'proposal_rre_deg'), ('rte', 'proposal_rte')]:
        if proposal in refined:
            refined[metric] = refined[proposal].fillna(refined[metric])
    paired = initial.merge(refined, on=['dataset', 'category', 'case_id'], suffixes=('_corsair', '_icp'), validate='one_to_one')
    if len(paired) != len(initial) or len(paired) != len(refined):
        raise ValueError('Missing paired poses')
    summaries = []
    for metric in ['rre_deg', 'rte']:
        paired[f'{metric}_oracle_choice'] = 'tie'
        paired.loc[paired[f'{metric}_icp'] < paired[f'{metric}_corsair'], f'{metric}_oracle_choice'] = 'CORSAIR + ICP'
        paired.loc[paired[f'{metric}_icp'] > paired[f'{metric}_corsair'], f'{metric}_oracle_choice'] = 'CORSAIR'
    for (dataset, category), group in paired.groupby(['dataset', 'category'], sort=False):
        for metric in ['rre_deg', 'rte']:
            choices = group[f'{metric}_oracle_choice']
            # Oracle uses one metric at a time; it need not optimize the other.
            choose_icp = choices == 'CORSAIR + ICP'
            rre = group.rre_deg_icp.where(choose_icp, group.rre_deg_corsair)
            rte = group.rte_icp.where(choose_icp, group.rte_corsair)
            summaries.append(dict(dataset=dataset, category=category, oracle_metric=metric, N=len(group),
                icp_better=int(choose_icp.sum()), corsair_better=int((choices == 'CORSAIR').sum()),
                ties=int((choices == 'tie').sum()),
                rmse_selection_joint_pct=float(100 * ((group.selected_rre_deg <= 5) & (group.selected_rte <= .02)).mean()),
                oracle_joint_pct=float(100 * ((rre <= 5) & (rte <= .02)).mean())))
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / 'gt-diagnostic-per_case.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'gt-diagnostic.csv', index=False)
    (output / 'gt-diagnostic.html').write_text('<!doctype html><meta charset="utf-8"><title>GT selection diagnostic</title>'
        '<h1>Ground-truth oracle diagnostic</h1><p>Analysis only. No poses or acceptance decisions were changed. '
        'Compare CORSAIR against raw ICP proposals when proposal errors are available; '
        'otherwise compare against the RMSE-selected CORSAIR + ICP output. '
        'RRE and RTE oracles select independently; ties retain CORSAIR.</p>' + summary.to_html(index=False))
    print(summary.to_string(index=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--complete-run', type=Path)
    sources.add_argument('--acceptance-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frames = []
    if args.complete_run:
        for number in [1, 2, 3]:
            frame = pd.read_csv(args.complete_run / f'table{number}-per_case.csv')
            frame['dataset'] = f'Table {number}'
            frame['case_id'] = frame['case_index' if number == 1 else 'query_index'].astype(str)
            frames.append(frame)
        default = args.complete_run
    else:
        frame = pd.read_csv(args.acceptance_run / 'per_case.csv')
        frame = frame[frame.policy.isin(['CORSAIR', 'RMSE only'])].copy()
        frame['method'] = frame.policy.map({'CORSAIR': 'CORSAIR', 'RMSE only': 'CORSAIR + ICP'})
        frames.append(frame)
        default = args.acceptance_run
    diagnose(pd.concat(frames, ignore_index=True), args.output or default)


if __name__ == '__main__':
    main()
