from pathlib import Path
import hashlib
import json
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
START = time.perf_counter()
ROOT = Path('/content/adversarially-hardened-dynamic-sensing')
STATE = AHDS_M6_CLOSED_LOOP_STATE
CHECKPOINT = 'a9258ba4a307d6a6ad4ff23788a392c16260ed48'
BASE = ROOT / 'results/module6/closed_loop_perception'
OUTPUT = ROOT / 'results/module6/recovery_analysis'
PROTECTED = dict(STATE['protected_files'])
SYSTEMS = ('A', 'B', 'C')
SCENES = (300000, 300040, 300080, 300120)
RECOVERY_FRACTION = 0.95
CONSECUTIVE_FRAMES = 3
CENSORING_THRESHOLD = 0.005
BASELINE_WINDOW = 5
ROI_COMPARISON_WINDOW = 3
COLORS = {'A': '#64748b', 'B': '#d97706', 'C': '#2563eb'}
print('=' * 70)
print('AHDS — M6-DEV-01D-03')
print('Closed-loop Perception Findings and Recovery Analysis')
print('=' * 70)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_protected():
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == CHECKPOINT
    for name, digest in PROTECTED.items():
        assert sha256(ROOT / name) == digest, name
assert STATE['task_passed']
assert len(PROTECTED) == 150
assert not OUTPUT.exists(), f'Output already exists: {OUTPUT}'
verify_protected()
assert sha256(BASE / 'experiment_manifest.json') == '8596189a6cfb737118a0f75319a53e022490bef066c5c211f8ebd0f7973f88c8'
print('\n[PREREQUISITES]')
print('Research checkpoint: VERIFIED')
print('Protected inventory: 150/150')
print('Experiment manifest: VERIFIED')
print('Output overwrite protection: PASS')
DF = pd.read_csv(BASE / 'frame_records.csv')
PROTOCOL = json.loads((BASE / 'experiment_protocol.json').read_text())
SCENARIOS = PROTOCOL['scenarios']
KEYS = ['scenario', 'scene_seed', 'system', 'frame']
assert len(DF) == 4320
assert not DF.duplicated(KEYS).any()
assert set(DF['system']) == set(SYSTEMS)
assert set(DF['scene_seed']) == set(SCENES)
for column in ['foreground_miou', 'censored_ratio', 'sampling_ratio', 'exposure_ms', 'gain']:
    assert np.isfinite(DF[column]).all(), column
for column in ['foreground_miou', 'censored_ratio', 'sampling_ratio']:
    assert DF[column].between(0, 1).all(), column
DF = DF.sort_values(KEYS).reset_index(drop=True)
for scenario in SCENARIOS:
    sid = scenario['id']
    for seed in SCENES:
        for system in SYSTEMS:
            group = DF[(DF.scenario == sid) & (DF.scene_seed == seed) & (DF.system == system)]
            assert group.frame.tolist() == list(range(scenario['frames']))
            expected_events = [any((start <= frame < end for start, end in scenario['windows'])) for frame in group.frame]
            assert group.event_active.tolist() == expected_events
print('\n[INPUT EVIDENCE]')
print('Frame records:', len(DF))
print('Unique scene seeds:', len(SCENES))
print('Scenarios:', len(SCENARIOS))
print('Systems:', len(SYSTEMS))
print('Temporal sequence integrity: PASS')
ANALYSIS_PROTOCOL = {'task_id': 'M6-DEV-01D-03', 'research_checkpoint': CHECKPOINT, 'source_experiment_manifest_sha256': sha256(BASE / 'experiment_manifest.json'), 'analysis_type': 'post_experiment_descriptive', 'semantic_recovery_fraction': RECOVERY_FRACTION, 'consecutive_recovery_frames': CONSECUTIVE_FRAMES, 'physical_censoring_threshold': CENSORING_THRESHOLD, 'clean_baseline_window_frames': BASELINE_WINDOW, 'baseline_definition': 'Mean frame foreground mIoU over the five frames preceding the first event, separately for each scenario, scene and system.', 'repeated_event_baseline': 'The original pre-first-event baseline is retained for every episode.', 'recovery_search_window': 'Event end inclusive to next event onset exclusive, or sequence end.', 'recovery_delay_definition': 'First qualifying frame minus event end.', 'confirmation_definition': 'Third consecutive qualifying frame.', 'sustained_recovery_definition': 'All frames from first qualifying frame through the recovery window end qualify.', 'roi_comparison_window': ROI_COMPARISON_WINDOW, 'roi_transition_definition': 'FULL_FRAME to ROI_PRIORITY transition.', 'aggregation': 'Scene-level and episode-level descriptive statistics, not pooled confusion matrices.', 'model_inference_performed': False, 'sensor_simulation_performed': False, 'controller_tuning_performed': False, 'statistical_significance_testing': False}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'analysis_protocol.json').write_text(json.dumps(ANALYSIS_PROTOCOL, indent=2) + '\n')
print('\n[ANALYSIS CRITERIA]')
print('Semantic recovery: >=95% of own clean baseline')
print('Physical recovery: censoring <0.5%')
print('Confirmation: 3 consecutive frames')
print('Baseline: 5 pre-first-event frames')
print('Criterion status: POST-EXPERIMENT DESCRIPTIVE')

def detect_recovery(frames, conditions, event_end):
    frames = np.asarray(frames, dtype=int)
    conditions = np.asarray(conditions, dtype=bool)
    assert len(frames) == len(conditions)
    result = {'recovered': False, 'restoration_frame': None, 'confirmation_frame': None, 'delay_frames': None, 'sustained': False}
    for i in range(len(conditions) - CONSECUTIVE_FRAMES + 1):
        if conditions[i:i + CONSECUTIVE_FRAMES].all():
            restoration = int(frames[i])
            confirmation = int(frames[i + CONSECUTIVE_FRAMES - 1])
            result.update({'recovered': True, 'restoration_frame': restoration, 'confirmation_frame': confirmation, 'delay_frames': restoration - event_end, 'sustained': bool(conditions[i:].all())})
            break
    return result
assert detect_recovery([30, 31, 32, 33], [True, True, True, False], 30)['delay_frames'] == 0
assert not detect_recovery([30, 31, 32, 33], [True, True, True, False], 30)['sustained']
assert not detect_recovery([30, 31], [True, True], 30)['recovered']
assert detect_recovery([30, 31, 32, 33], [False, True, True, True], 30)['delay_frames'] == 1
print('\nRecovery utility checks: 4/4 PASS')
RECOVERY_ROWS = []
for scenario in SCENARIOS:
    sid = scenario['id']
    windows = scenario['windows']
    if not windows:
        continue
    first_onset = windows[0][0]
    for seed in SCENES:
        for system in SYSTEMS:
            group = DF[(DF.scenario == sid) & (DF.scene_seed == seed) & (DF.system == system)].copy()
            baseline = group[(group.frame >= first_onset - BASELINE_WINDOW) & (group.frame < first_onset)]
            assert len(baseline) == BASELINE_WINDOW
            baseline_miou = float(baseline.foreground_miou.mean())
            semantic_threshold = RECOVERY_FRACTION * baseline_miou
            for episode, (onset, end) in enumerate(windows, start=1):
                recovery_end = windows[episode][0] if episode < len(windows) else scenario['frames']
                event = group[(group.frame >= onset) & (group.frame < end)]
                recovery = group[(group.frame >= end) & (group.frame < recovery_end)]
                assert len(event) == end - onset
                assert len(recovery) == recovery_end - end
                semantic = detect_recovery(recovery.frame, recovery.foreground_miou >= semantic_threshold, end)
                physical = detect_recovery(recovery.frame, recovery.censored_ratio < CENSORING_THRESHOLD, end)
                joint = detect_recovery(recovery.frame, (recovery.foreground_miou >= semantic_threshold) & (recovery.censored_ratio < CENSORING_THRESHOLD), end)
                row = {'scenario': sid, 'kind': scenario['kind'], 'scene_seed': seed, 'system': system, 'episode': episode, 'event_onset': onset, 'event_end': end, 'recovery_window_end': recovery_end, 'recovery_window_frames': len(recovery), 'baseline_miou': baseline_miou, 'semantic_threshold': semantic_threshold, 'event_mean_miou': float(event.foreground_miou.mean()), 'event_min_miou': float(event.foreground_miou.min()), 'event_mean_censoring': float(event.censored_ratio.mean()), 'recovery_mean_miou': float(recovery.foreground_miou.mean()), 'final_recovery_miou': float(recovery.foreground_miou.iloc[-1]), 'event_miou_loss_pp': 100 * (baseline_miou - event.foreground_miou.mean())}
                for prefix, result in [('semantic', semantic), ('physical', physical), ('joint', joint)]:
                    for name, value in result.items():
                        row[f'{prefix}_{name}'] = value
                if semantic['recovered'] and physical['recovered']:
                    row['semantic_minus_physical_delay'] = semantic['delay_frames'] - physical['delay_frames']
                else:
                    row['semantic_minus_physical_delay'] = None
                RECOVERY_ROWS.append(row)
RECOVERY = pd.DataFrame(RECOVERY_ROWS)
assert len(RECOVERY) == 72
assert len(RECOVERY[RECOVERY.kind == 'laser']) == 36
RECOVERY.to_csv(OUTPUT / 'episode_recovery.csv', index=False)
print('\n[EPISODE RECOVERY]')
print('Total system-episode observations:', len(RECOVERY))
print('Optical disturbance observations: 36')
print('Benign illumination observations: 36')
SUMMARY_ROWS = []
GROUPS = {'optical_disturbance': RECOVERY[RECOVERY.kind == 'laser'], 'benign_illumination': RECOVERY[RECOVERY.kind != 'laser']}
for category, subset in GROUPS.items():
    for system in SYSTEMS:
        group = subset[subset.system == system]
        row = {'category': category, 'system': system, 'episodes': len(group)}
        for metric in ['semantic', 'physical', 'joint']:
            recovered = group[group[f'{metric}_recovered']]
            row[f'{metric}_recovered_count'] = len(recovered)
            row[f'{metric}_sustained_count'] = int(group[f'{metric}_sustained'].sum())
            delays = recovered[f'{metric}_delay_frames']
            row[f'{metric}_mean_delay_frames'] = float(delays.mean()) if len(delays) else None
            row[f'{metric}_median_delay_frames'] = float(delays.median()) if len(delays) else None
        SUMMARY_ROWS.append(row)
RECOVERY_SUMMARY = pd.DataFrame(SUMMARY_ROWS)
RECOVERY_SUMMARY.to_csv(OUTPUT / 'recovery_summary.csv', index=False)
for category in GROUPS:
    print(f'\n{category.upper()}')
    for _, row in RECOVERY_SUMMARY[RECOVERY_SUMMARY.category == category].iterrows():
        print(f'  System {row.system} | Semantic: {row.semantic_recovered_count}/{row.episodes} | Sustained: {row.semantic_sustained_count}/{row.episodes} | Physical: {row.physical_recovered_count}/{row.episodes} | Joint: {row.joint_recovered_count}/{row.episodes}')
        print(f'    Mean recovered-only delays: semantic={row.semantic_mean_delay_frames:.2f}, physical={row.physical_mean_delay_frames:.2f}, joint={row.joint_mean_delay_frames:.2f} frames')
SCENE_ROWS = []
for scenario in SCENARIOS:
    sid = scenario['id']
    phase = 'clean' if sid == 'P01' else 'event'
    for seed in SCENES:
        group = DF[(DF.scenario == sid) & (DF.scene_seed == seed) & (DF.phase == phase)]
        means = group.groupby('system')[['foreground_miou', 'censored_ratio', 'ideal_sample_reduction']].mean()
        assert set(means.index) == set(SYSTEMS)
        row = {'scenario': sid, 'scene_seed': seed, 'phase': phase}
        for system in SYSTEMS:
            row[f'{system}_miou'] = float(means.loc[system, 'foreground_miou'])
            row[f'{system}_censoring'] = float(means.loc[system, 'censored_ratio'])
            row[f'{system}_sample_reduction'] = float(means.loc[system, 'ideal_sample_reduction'])
        row['C_minus_B_miou_pp'] = 100 * (row['C_miou'] - row['B_miou'])
        row['C_minus_A_miou_pp'] = 100 * (row['C_miou'] - row['A_miou'])
        row['C_minus_B_censoring_pp'] = 100 * (row['C_censoring'] - row['B_censoring'])
        SCENE_ROWS.append(row)
PAIRED = pd.DataFrame(SCENE_ROWS)
assert len(PAIRED) == 24
PAIRED.to_csv(OUTPUT / 'paired_scene_comparison.csv', index=False)
TOLERANCE = 1e-06
print('\n[SCENE-LEVEL COMPARISONS]')
for scenario in SCENARIOS:
    sid = scenario['id']
    group = PAIRED[PAIRED.scenario == sid]
    delta = group.C_minus_B_miou_pp
    better = int((delta > TOLERANCE).sum())
    equal = int((delta.abs() <= TOLERANCE).sum())
    worse = int((delta < -TOLERANCE).sum())
    print(f'{sid} | C vs B better/equal/worse: {better}/{equal}/{worse} | Mean: {delta.mean():+.4f} pp | Range: [{delta.min():+.4f}, {delta.max():+.4f}]')
ROI_ROWS = []
for scenario in SCENARIOS:
    sid = scenario['id']
    for seed in SCENES:
        group = DF[(DF.scenario == sid) & (DF.scene_seed == seed) & (DF.system == 'C')].reset_index(drop=True)
        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]
            if not (previous.readout_mode == 'FULL_FRAME' and current.readout_mode == 'ROI_PRIORITY'):
                continue
            frame = int(current.frame)
            before = group.iloc[max(0, i - ROI_COMPARISON_WINDOW):i]
            after = group.iloc[i:i + ROI_COMPARISON_WINDOW]
            transition_type = 'startup' if frame == 1 else 'roi_resumption'
            complete_window = len(before) == ROI_COMPARISON_WINDOW and len(after) == ROI_COMPARISON_WINDOW
            stable_readout_windows = before.readout_mode.eq('FULL_FRAME').all() and after.readout_mode.eq('ROI_PRIORITY').all()
            same_phase = pd.concat([before, after]).phase.nunique() == 1
            matched_configuration = pd.concat([before, after]).exposure_ms.nunique() == 1 and pd.concat([before, after]).gain.nunique() == 1
            ROI_ROWS.append({'scenario': sid, 'scene_seed': seed, 'transition_frame': frame, 'transition_type': transition_type, 'phase': current.phase, 'complete_window': complete_window, 'stable_readout_windows': bool(stable_readout_windows), 'same_phase': bool(same_phase), 'matched_configuration': bool(matched_configuration), 'before_miou': float(before.foreground_miou.mean()), 'after_miou': float(after.foreground_miou.mean()), 'miou_change_pp': 100 * float(after.foreground_miou.mean() - before.foreground_miou.mean()), 'before_sampling_ratio': float(before.sampling_ratio.mean()), 'after_sampling_ratio': float(after.sampling_ratio.mean()), 'before_exposure_ms': float(before.exposure_ms.mean()), 'after_exposure_ms': float(after.exposure_ms.mean()), 'before_gain': float(before.gain.mean()), 'after_gain': float(after.gain.mean())})
ROI = pd.DataFrame(ROI_ROWS)
assert not ROI.empty
ROI.to_csv(OUTPUT / 'roi_transition_analysis.csv', index=False)
RESUMPTIONS = ROI[(ROI.transition_type == 'roi_resumption') & ROI.complete_window & ROI.stable_readout_windows & ROI.same_phase].copy()
MATCHED = RESUMPTIONS[RESUMPTIONS.matched_configuration]
print('\n[ROI RESUMPTION]')
print('All full-to-ROI transitions:', len(ROI))
print('Eligible resumption transitions:', len(RESUMPTIONS))
print('Configuration-matched transitions:', len(MATCHED))
if len(RESUMPTIONS):
    print('Mean resumption mIoU change:', f'{RESUMPTIONS.miou_change_pp.mean():+.4f} pp')
    print('Resumptions with reduced mIoU:', int((RESUMPTIONS.miou_change_pp < -TOLERANCE).sum()), '/', len(RESUMPTIONS))
if len(MATCHED):
    print('Configuration-matched mean change:', f'{MATCHED.miou_change_pp.mean():+.4f} pp')
DISAGREEMENT_ROWS = []
for scenario in SCENARIOS:
    if scenario['kind'] != 'laser':
        continue
    sid = scenario['id']
    for seed in SCENES:
        for system in SYSTEMS:
            group = DF[(DF.scenario == sid) & (DF.scene_seed == seed) & (DF.system == system)]
            onset = scenario['windows'][0][0]
            baseline = group[(group.frame >= onset - BASELINE_WINDOW) & (group.frame < onset)].foreground_miou.mean()
            event = group[group.phase == 'event']
            low_censoring = event.censored_ratio < CENSORING_THRESHOLD
            degraded_semantics = event.foreground_miou < RECOVERY_FRACTION * baseline
            disagreement = low_censoring & degraded_semantics
            DISAGREEMENT_ROWS.append({'scenario': sid, 'scene_seed': seed, 'system': system, 'event_frames': len(event), 'low_censoring_frames': int(low_censoring.sum()), 'degraded_semantic_frames': int(degraded_semantics.sum()), 'low_censoring_but_degraded_frames': int(disagreement.sum()), 'disagreement_fraction': float(disagreement.mean())})
DISAGREEMENT = pd.DataFrame(DISAGREEMENT_ROWS)
assert len(DISAGREEMENT) == 24
DISAGREEMENT.to_csv(OUTPUT / 'physical_semantic_disagreement.csv', index=False)
print('\n[PHYSICAL-SEMANTIC DISAGREEMENT]')
for system in SYSTEMS:
    group = DISAGREEMENT[DISAGREEMENT.system == system]
    count = int(group.low_censoring_but_degraded_frames.sum())
    total = int(group.event_frames.sum())
    print(f'System {system}: {count}/{total} disturbance frames ({100 * count / total:.2f}%)')
FIGURES = OUTPUT / 'figures'
FIGURES.mkdir(exist_ok=False)

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, category in zip(axes, GROUPS):
    group = RECOVERY_SUMMARY[RECOVERY_SUMMARY.category == category].set_index('system')
    x = np.arange(4)
    metrics = ['semantic_recovered_count', 'semantic_sustained_count', 'physical_recovered_count', 'joint_recovered_count']
    for i, system in enumerate(SYSTEMS):
        values = [100 * group.loc[system, metric] / group.loc[system, 'episodes'] for metric in metrics]
        ax.bar(x + (i - 1) * 0.25, values, width=0.25, color=COLORS[system], label=system)
    ax.set_xticks(x, ['Semantic', 'Sustained\nsemantic', 'Physical', 'Joint'])
    ax.set_title(category.replace('_', ' ').title())
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.2)
axes[0].set_ylabel('Recovered episodes (%)')
axes[0].legend()
fig.suptitle('M6-F14 — Physical and semantic recovery outcomes')
save_figure(fig, 'M6_F14_recovery_outcomes')
matrix = PAIRED.pivot(index='scenario', columns='scene_seed', values='C_minus_B_miou_pp')
values = matrix.to_numpy()
limit = max(float(np.abs(values).max()), 0.1)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
ax.set_xticks(range(len(matrix.columns)), matrix.columns)
ax.set_yticks(range(len(matrix.index)), matrix.index)
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(j, i, f'{values[i, j]:+.2f}', ha='center', va='center', color='white' if abs(values[i, j]) > 0.65 * limit else 'black')
ax.set_xlabel('Held-out scene seed')
ax.set_title('M6-F15 — AHDS minus unprotected dynamic\nMean frame foreground mIoU difference')
fig.colorbar(im, ax=ax, label='Percentage points')
save_figure(fig, 'M6_F15_paired_scene_improvements')
fig, ax = plt.subplots(figsize=(10, 5))
if len(RESUMPTIONS):
    plot_data = RESUMPTIONS.sort_values(['scenario', 'scene_seed', 'transition_frame'])
    labels = [f'{r.scenario}-{r.scene_seed}\nf{r.transition_frame}' for r in plot_data.itertuples()]
    ax.bar(range(len(plot_data)), plot_data.miou_change_pp, color=['#2563eb' if matched else '#94a3b8' for matched in plot_data.matched_configuration])
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
else:
    ax.text(0.5, 0.5, 'No eligible ROI resumption windows', transform=ax.transAxes, ha='center')
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('After minus before mIoU (pp)')
ax.set_title('M6-F16 — Perception change at ROI resumption\nBlue: matched exposure/gain; grey: configuration changed')
ax.grid(axis='y', alpha=0.2)
save_figure(fig, 'M6_F16_roi_resumption')
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
for i, system in enumerate(SYSTEMS):
    values = []
    for sid in ['P02', 'P03']:
        group = DISAGREEMENT[(DISAGREEMENT.scenario == sid) & (DISAGREEMENT.system == system)]
        values.append(100 * group.low_censoring_but_degraded_frames.sum() / group.event_frames.sum())
    ax.bar(x + (i - 1) * 0.25, values, width=0.25, color=COLORS[system], label=system)
ax.set_xticks(x, ['Continuous disturbance', 'Repeated disturbance'])
ax.set_ylabel('Disturbance frames (%)')
ax.set_ylim(0, 105)
ax.set_title('M6-F17 — Low censoring does not guarantee\nsemantic information preservation')
ax.legend()
ax.grid(axis='y', alpha=0.2)
save_figure(fig, 'M6_F17_physical_semantic_disagreement')

def json_records(frame):
    return json.loads(frame.to_json(orient='records'))
REPORT = {'task_id': 'M6-DEV-01D-03', 'task_passed': True, 'research_checkpoint': CHECKPOINT, 'source_frame_records': len(DF), 'unique_scenes': len(SCENES), 'system_episode_observations': len(RECOVERY), 'optical_disturbance_observations': 36, 'benign_illumination_observations': 36, 'paired_scene_comparisons': len(PAIRED), 'recovery_summary': json_records(RECOVERY_SUMMARY), 'roi_transitions': len(ROI), 'eligible_roi_resumptions': len(RESUMPTIONS), 'configuration_matched_resumptions': len(MATCHED), 'mean_roi_resumption_change_pp': float(RESUMPTIONS.miou_change_pp.mean()) if len(RESUMPTIONS) else None, 'matched_roi_resumption_change_pp': float(MATCHED.miou_change_pp.mean()) if len(MATCHED) else None, 'physical_semantic_disagreement': json_records(DISAGREEMENT), 'model_training_performed': False, 'model_inference_performed': False, 'sensor_simulation_performed': False, 'controller_modified': False, 'dataset_modified': False, 'historical_modifications': False, 'limitations': ['Post-experiment descriptive recovery criteria.', 'Four selected procedural held-out scenes.', 'Noise-free static scene sequences.', 'Temporal observations are not independent.', 'Recovery thresholds are system-specific.', 'Recovery to own baseline does not imply equal absolute perception quality.', 'Recovered-only delay excludes failed episodes.', 'Mean frame mIoU differs from pooled mIoU.', 'ROI transition comparisons are observational.', 'Exposure and gain may change near transitions.', 'Physical censoring recovery is not full optical information recovery.', 'Real-world validation remains pending.'], 'execution_seconds': time.perf_counter() - START}
REPORT_PATH = OUTPUT / 'analysis_report.json'
REPORT_PATH.write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')
verify_protected()
INVENTORY = {str(path.relative_to(ROOT)): {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in sorted(OUTPUT.rglob('*')) if path.is_file()}
MANIFEST_PATH = OUTPUT / 'analysis_manifest.json'
MANIFEST_PATH.write_text(json.dumps({'task_id': 'M6-DEV-01D-03', 'research_checkpoint': CHECKPOINT, 'files': INVENTORY}, indent=2) + '\n')
NEW_PROTECTED = {}
for name, metadata in INVENTORY.items():
    path = ROOT / name
    assert sha256(path) == metadata['sha256']
    assert path.stat().st_size == metadata['bytes']
    NEW_PROTECTED[name] = metadata['sha256']
NEW_PROTECTED[str(MANIFEST_PATH.relative_to(ROOT))] = sha256(MANIFEST_PATH)
assert len(NEW_PROTECTED) == 16
verify_protected()
AHDS_M6_RECOVERY_STATE = {**REPORT, 'report_path': str(REPORT_PATH.relative_to(ROOT)), 'manifest_path': str(MANIFEST_PATH.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST_PATH), 'protected_files': {**PROTECTED, **NEW_PROTECTED}, 'next_task': 'M6-DEV-01D-04 — Evidence-Guided Refinement Decision'}
print('\n' + '=' * 70)
print('CLOSED-LOOP RECOVERY ANALYSIS SUMMARY')
print('=' * 70)
print('Source frame records:', len(DF))
print('System-episode observations:', len(RECOVERY))
print('Paired scene comparisons:', len(PAIRED))
print('ROI transitions:', len(ROI))
print('Eligible ROI resumptions:', len(RESUMPTIONS))
print('Configuration-matched resumptions:', len(MATCHED))
print('\n[OPTICAL DISTURBANCE RECOVERY]')
for _, row in RECOVERY_SUMMARY[RECOVERY_SUMMARY.category == 'optical_disturbance'].iterrows():
    print(f'System {row.system} | Semantic: {row.semantic_recovered_count}/{row.episodes} | Sustained: {row.semantic_sustained_count}/{row.episodes} | Physical: {row.physical_recovered_count}/{row.episodes} | Joint: {row.joint_recovered_count}/{row.episodes}')
print('\n[ARTIFACT INTEGRITY]')
print('Historical files: 150/150')
print('New artifacts:', len(NEW_PROTECTED))
print('Artifact verification: PASS')
print('New model inference: NONE')
print('New sensor simulation: NONE')
print('Model modifications: NONE')
print('Controller modifications: NONE')
print('Dataset modifications: NONE')
print('Git staging: NONE')
print('Protected inventory:', len(AHDS_M6_RECOVERY_STATE['protected_files']))
print('\n[ARTIFACT IDENTITIES]')
for name, digest in NEW_PROTECTED.items():
    print(name)
    print('SHA256:', digest)
print('\nExecution time:', f'{time.perf_counter() - START:.2f}s')
print('\nM6-DEV-01D-03: RECOVERY ANALYSIS PASSED')
print('Next: M6-DEV-01D-04 — Evidence-Guided Refinement Decision')