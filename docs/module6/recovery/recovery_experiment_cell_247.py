from pathlib import Path
from collections import defaultdict
import hashlib
import subprocess
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
START = time.perf_counter()
ROOT = Path('/content/adversarially-hardened-dynamic-sensing')
STATE = AHDS_M6_RECOVERY_STATE
PROTECTED = dict(STATE['protected_files'])
CHECKPOINT = 'a9258ba4a307d6a6ad4ff23788a392c16260ed48'
OUTPUT = ROOT / 'results/module6/acquisition_ablation'
MODEL_PATH = ROOT / 'results/module6/model_training/best_model.pt'
MODEL_HASH = '5a4d3736aef265bce70dc540cad07a83e4c4555244c2185806478507cdce257f'
CLASS_NAMES = ('background', 'lane_markings', 'vehicle', 'pedestrian')
SCENE_INDICES = tuple(range(0, 160, 5))
CONFIGURATIONS = {'K1': (10.0, 1.0), 'K2': (30.0, 1.0), 'K3': (30.0, 3.0), 'K4': (2.0, 1.0), 'K5': (2.0, 3.0)}
CONDITIONS = ('clean', 'disturbed')
MODES = ('full', 'roi')
EXPECTED = len(SCENE_INDICES) * len(CONFIGURATIONS) * len(CONDITIONS) * len(MODES)
assert EXPECTED == 640
print('=' * 70)
print('AHDS — M6-DEV-01D-04A')
print('Controlled Acquisition Ablation')
print('=' * 70)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_protected():
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == CHECKPOINT
    for name, digest in PROTECTED.items():
        assert sha256(ROOT / name) == digest, name
assert STATE['task_passed']
assert len(PROTECTED) == 166
assert not OUTPUT.exists(), f'Output already exists: {OUTPUT}'
verify_protected()
assert sha256(MODEL_PATH) == MODEL_HASH
print('\n[PREREQUISITES]')
print('Protected inventory: 166/166')
print('Research checkpoint: VERIFIED')
print('Frozen model checkpoint: VERIFIED')
print('Output overwrite protection: PASS')
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from ahds.module1_environment.optical_environment import AttackConfig, apply_optical_attack
from ahds.module2_sensor.sensor_simulator import SensorConfig, simulate_sensor
from ahds.module3_dynamic_sensing.roi_allocator import allocate_readout
from ahds.integration.roi_closed_loop import AdaptiveROIConfig
from ahds.module6_perception.procedural_dataset import ProceduralPerceptionDataset
from ahds.module6_perception.lightweight_segmentation import LightweightSegmentationNet
from ahds.module6_perception.segmentation_metrics import SegmentationMetrics
DEVICE = torch.device('cpu')
MODEL = LightweightSegmentationNet(num_classes=4).to(DEVICE)
checkpoint_data = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
MODEL.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
MODEL.eval()
MODEL.requires_grad_(False)
assert checkpoint_data['epoch'] == 38
assert sum((p.numel() for p in MODEL.parameters())) == 10700

def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()

def model_hashes():
    return {name: tensor_hash(value) for name, value in MODEL.state_dict().items()}
INITIAL_MODEL_HASHES = model_hashes()
DATASET = ProceduralPerceptionDataset('validation')
REGISTRY = pd.read_csv(ROOT / 'datasets/ahds_procedural_v1/validation_index.csv')
assert len(DATASET) == 160
assert len(REGISTRY) == 160
SCENES = {}
for index in SCENE_INDICES:
    scene, target, metadata = DATASET.optical_sample(index)
    row = REGISTRY.iloc[index]
    seed = DATASET.scene_seed(index)
    assert int(row.scene_seed) == seed
    assert tensor_hash(scene) == row.optical_scene_sha256
    assert tensor_hash(target) == row.semantic_target_sha256
    SCENES[index] = {'scene': scene, 'target': target, 'seed': seed}
assert len(SCENES) == 32
print('\n[MODEL AND DATA]')
print('Frozen model: Epoch 38')
print('Parameters: 10700')
print('Validation scenes:', len(SCENES))
print('Registry tensor checks: 64/64')
print('Held-out test scenes used: 0')
ROI_CONFIG = AdaptiveROIConfig()
ROI_CONFIG.validate()
ROI = ROI_CONFIG.resolve(256, 256)
ATTACK = AttackConfig(attack_type='laser', center_x=0.5, center_y=0.5, radius=0.12, intensity=15.0)
PROTOCOL = {'task_id': 'M6-DEV-01D-04A', 'research_checkpoint': CHECKPOINT, 'model_checkpoint_sha256': MODEL_HASH, 'dataset_split': 'validation', 'scene_indices': list(SCENE_INDICES), 'scene_seeds': [SCENES[i]['seed'] for i in SCENE_INDICES], 'configurations': {name: {'exposure_ms': exposure, 'gain': gain} for name, (exposure, gain) in CONFIGURATIONS.items()}, 'conditions': list(CONDITIONS), 'readout_modes': list(MODES), 'sensor_noise_enabled': False, 'optical_disturbance': {'type': 'laser', 'intensity': 15.0, 'radius': 0.12, 'center': [0.5, 0.5], 'time_step': 0}, 'background_stride': ROI_CONFIG.background_stride, 'expected_evaluations': EXPECTED, 'primary_metric': 'foreground_miou', 'experiment_type': 'matched_acquisition_diagnostic', 'controller_updates': False, 'model_training': False, 'test_split_used': False, 'device': str(DEVICE), 'cpu_threads': torch.get_num_threads()}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'ablation_protocol.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')
print('\n[EXPERIMENT PROTOCOL]')
print('Configurations:', len(CONFIGURATIONS))
print('Optical conditions:', len(CONDITIONS))
print('Readout modes:', len(MODES))
print('Sensor acquisitions: 320')
print('Perception evaluations:', EXPECTED)
print('Sensor noise: DISABLED')
print('Controller tuning: NONE')

def confusion_matrix(prediction, target):
    encoded = target.reshape(-1).long() * 4 + prediction.reshape(-1).long()
    return torch.bincount(encoded, minlength=16).reshape(4, 4).cpu().numpy()

def summarize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = actual + predicted - tp
    iou = [float(tp[i] / union[i]) if union[i] > 0 else None for i in range(4)]
    foreground = [value for value in iou[1:] if value is not None]
    return {'foreground_miou': float(np.mean(foreground)), 'pixel_accuracy': float(tp.sum() / matrix.sum()), 'class_iou': dict(zip(CLASS_NAMES, iou))}

def model_input(rgb, mask):
    assert rgb.shape == (3, 256, 256)
    assert mask.shape == (256, 256)
    assert (rgb[:, ~mask.bool()] == 0).all()
    return torch.cat([rgb, mask.float().unsqueeze(0)], dim=0)
RECORDS = []
POOLED = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
ACQUISITIONS = 0
PAIR_CHECKS = 0
METRIC_CHECKS = 0
print('\n[ABLATION EXECUTION]')
with torch.inference_mode():
    for scene_number, index in enumerate(SCENE_INDICES, start=1):
        clean = SCENES[index]['scene']
        target = SCENES[index]['target']
        seed = SCENES[index]['seed']
        disturbed, attack_map = apply_optical_attack(clean, ATTACK, time_step=0)
        optical_conditions = {'clean': clean, 'disturbed': disturbed}
        for condition, optical_scene in optical_conditions.items():
            for config_id, (exposure, gain) in CONFIGURATIONS.items():
                config = SensorConfig(exposure_ms=exposure, gain=gain)
                sensor = simulate_sensor(optical_scene, config=config, add_noise=False, seed=42)
                ACQUISITIONS += 1
                rgb = sensor['normalized_output']
                full_mask = torch.ones((256, 256), dtype=torch.bool, device=rgb.device)
                roi_readout = allocate_readout(rgb, ROI, background_stride=ROI_CONFIG.background_stride)
                roi_mask = roi_readout['sampling_mask'].bool()
                roi_rgb = roi_readout['readout']
                assert torch.equal(roi_rgb[:, roi_mask], rgb[:, roi_mask])
                assert (roi_rgb[:, ~roi_mask] == 0).all()
                assert abs(float(roi_mask.float().mean()) - float(roi_readout['sampling_ratio'])) < 1e-07
                PAIR_CHECKS += 3
                inputs = torch.stack([model_input(rgb, full_mask), model_input(roi_rgb, roi_mask)]).to(DEVICE)
                logits = MODEL(inputs)
                assert logits.shape == (2, 4, 256, 256)
                assert torch.isfinite(logits).all()
                predictions = logits.argmax(dim=1).cpu()
                censoring = sensor['saturated'].bool() | sensor['adc_clipped'].bool()
                for mode, prediction, mask in zip(MODES, predictions, [full_mask, roi_mask]):
                    matrix = confusion_matrix(prediction, target)
                    metrics = summarize(matrix)
                    if METRIC_CHECKS < 2:
                        reference = SegmentationMetrics(num_classes=4)
                        reference.update(prediction, target)
                        result = reference.compute()
                        assert abs(result['foreground_miou'] - metrics['foreground_miou']) < 1e-07
                        METRIC_CHECKS += 1
                    POOLED[condition, config_id, mode] += matrix
                    row = {'scene_index': index, 'scene_seed': seed, 'condition': condition, 'configuration': config_id, 'exposure_ms': exposure, 'gain': gain, 'readout_mode': mode, 'sampling_ratio': float(mask.float().mean()), 'censored_ratio': float(censoring.float().mean()), 'selected_censored_ratio': float(censoring[:, mask].float().mean()), 'mean_sensor_output': float(rgb.mean()), 'foreground_miou': metrics['foreground_miou'], 'pixel_accuracy': metrics['pixel_accuracy']}
                    for name in CLASS_NAMES:
                        row[f'iou_{name}'] = metrics['class_iou'][name]
                    RECORDS.append(row)
        if scene_number % 4 == 0:
            print(f'  Scenes: {scene_number}/32 | Evaluations: {len(RECORDS)}/{EXPECTED}', flush=True)
assert len(RECORDS) == EXPECTED
assert ACQUISITIONS == 320
assert PAIR_CHECKS == 960
assert METRIC_CHECKS == 2
assert model_hashes() == INITIAL_MODEL_HASHES
assert sha256(MODEL_PATH) == MODEL_HASH
assert not MODEL.training
verify_protected()
DF = pd.DataFrame(RECORDS)
assert not DF.duplicated(['scene_seed', 'condition', 'configuration', 'readout_mode']).any()
DF.to_csv(OUTPUT / 'ablation_records.csv', index=False)
print('\n[EXECUTION INTEGRITY]')
print('Evaluations:', len(DF), '/640')
print('Sensor acquisitions:', ACQUISITIONS)
print('Matched acquisition checks:', PAIR_CHECKS)
print('Metric equivalence checks:', METRIC_CHECKS)
print('Frozen model state: UNCHANGED')
SUMMARY_ROWS = []
for key, matrix in sorted(POOLED.items()):
    condition, config_id, mode = key
    group = DF[(DF.condition == condition) & (DF.configuration == config_id) & (DF.readout_mode == mode)]
    assert len(group) == 32
    assert matrix.sum() == 32 * 256 * 256
    metrics = summarize(matrix)
    row = {'condition': condition, 'configuration': config_id, 'readout_mode': mode, 'scenes': len(group), 'foreground_miou': metrics['foreground_miou'], 'pixel_accuracy': metrics['pixel_accuracy'], 'mean_scene_miou': float(group.foreground_miou.mean()), 'mean_censoring': float(group.censored_ratio.mean()), 'sampling_ratio': float(group.sampling_ratio.mean())}
    for name in CLASS_NAMES:
        row[f'iou_{name}'] = metrics['class_iou'][name]
    SUMMARY_ROWS.append(row)
SUMMARY = pd.DataFrame(SUMMARY_ROWS)
assert len(SUMMARY) == 20
SUMMARY.to_csv(OUTPUT / 'configuration_summary.csv', index=False)
print('\n[CONFIGURATION PERFORMANCE]')
for condition in CONDITIONS:
    print(f'\n{condition.upper()}')
    for config_id in CONFIGURATIONS:
        group = SUMMARY[(SUMMARY.condition == condition) & (SUMMARY.configuration == config_id)].set_index('readout_mode')
        full = group.loc['full']
        roi = group.loc['roi']
        print(f'{config_id} | Full: {100 * full.foreground_miou:.4f}% | ROI: {100 * roi.foreground_miou:.4f}% | ROI delta: {100 * (roi.foreground_miou - full.foreground_miou):+.4f} pp | Censoring: {100 * full.mean_censoring:.4f}%')
PAIRED = DF.pivot(index=['scene_seed', 'condition', 'configuration'], columns='readout_mode', values='foreground_miou').reset_index()
PAIRED.columns.name = None
PAIRED['roi_minus_full_pp'] = 100 * (PAIRED.roi - PAIRED.full)
assert len(PAIRED) == 320
PAIRED.to_csv(OUTPUT / 'paired_roi_effect.csv', index=False)
print('\n[MATCHED ROI EFFECT]')
for condition in CONDITIONS:
    for config_id in CONFIGURATIONS:
        group = PAIRED[(PAIRED.condition == condition) & (PAIRED.configuration == config_id)]
        delta = group.roi_minus_full_pp
        tolerance = 1e-06
        print(f'{condition:10s} {config_id} | Mean: {delta.mean():+.4f} pp | Better/equal/worse: {int((delta > tolerance).sum())}/{int((delta.abs() <= tolerance).sum())}/{int((delta < -tolerance).sum())}')
CONTRASTS = [('increased_exposure', 'K1', 'K2'), ('gain_at_high_exposure', 'K2', 'K3'), ('reduced_exposure', 'K1', 'K4'), ('gain_at_low_exposure', 'K4', 'K5')]
FACTOR_ROWS = []
for condition in CONDITIONS:
    for mode in MODES:
        group = SUMMARY[(SUMMARY.condition == condition) & (SUMMARY.readout_mode == mode)].set_index('configuration')
        for name, reference, changed in CONTRASTS:
            before = group.loc[reference]
            after = group.loc[changed]
            FACTOR_ROWS.append({'condition': condition, 'readout_mode': mode, 'factor': name, 'reference': reference, 'changed': changed, 'miou_change_pp': 100 * (after.foreground_miou - before.foreground_miou), 'censoring_change_pp': 100 * (after.mean_censoring - before.mean_censoring)})
FACTORS = pd.DataFrame(FACTOR_ROWS)
assert len(FACTORS) == 16
FACTORS.to_csv(OUTPUT / 'acquisition_factor_contrasts.csv', index=False)
print('\n[ACQUISITION FACTOR CONTRASTS]')
print(FACTORS[['condition', 'readout_mode', 'factor', 'miou_change_pp', 'censoring_change_pp']].to_string(index=False, float_format=lambda value: f'{value:+.4f}'))
RESIDUAL_ROWS = []
for config_id in CONFIGURATIONS:
    for mode in MODES:
        group = SUMMARY[(SUMMARY.configuration == config_id) & (SUMMARY.readout_mode == mode)].set_index('condition')
        clean = group.loc['clean']
        disturbed = group.loc['disturbed']
        RESIDUAL_ROWS.append({'configuration': config_id, 'readout_mode': mode, 'clean_miou': float(clean.foreground_miou), 'disturbed_miou': float(disturbed.foreground_miou), 'disturbance_miou_loss_pp': 100 * (clean.foreground_miou - disturbed.foreground_miou), 'disturbed_censoring': float(disturbed.mean_censoring)})
RESIDUAL = pd.DataFrame(RESIDUAL_ROWS)
RESIDUAL.to_csv(OUTPUT / 'residual_interference.csv', index=False)
print('\n[RESIDUAL OPTICAL INTERFERENCE]')
for row in RESIDUAL.itertuples():
    print(f'{row.configuration} {row.readout_mode:4s} | Disturbance mIoU loss: {row.disturbance_miou_loss_pp:.4f} pp | Censoring: {100 * row.disturbed_censoring:.4f}%')
FIGURES = OUTPUT / 'figures'
FIGURES.mkdir(exist_ok=False)

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
x = np.arange(len(CONFIGURATIONS))
for ax, condition in zip(axes, CONDITIONS):
    group = SUMMARY[SUMMARY.condition == condition]
    for i, mode in enumerate(MODES):
        values = [100 * group[(group.configuration == config_id) & (group.readout_mode == mode)].foreground_miou.iloc[0] for config_id in CONFIGURATIONS]
        ax.bar(x + (i - 0.5) * 0.35, values, width=0.35, label=mode)
    ax.set_xticks(x, list(CONFIGURATIONS))
    ax.set_title(condition.title())
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.2)
axes[0].set_ylabel('Foreground mIoU (%)')
axes[0].legend()
fig.suptitle('M6-F18 — Matched acquisition perception ablation')
save_figure(fig, 'M6_F18_acquisition_ablation')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, condition in zip(axes, CONDITIONS):
    values = [PAIRED[(PAIRED.condition == condition) & (PAIRED.configuration == config_id)].roi_minus_full_pp.to_numpy() for config_id in CONFIGURATIONS]
    ax.boxplot(values, tick_labels=list(CONFIGURATIONS), showmeans=True)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title(condition.title())
    ax.grid(axis='y', alpha=0.2)
axes[0].set_ylabel('ROI minus full-frame mIoU (pp)')
fig.suptitle('M6-F19 — ROI effect under identical acquisition')
save_figure(fig, 'M6_F19_matched_roi_effect')
fig, ax = plt.subplots(figsize=(9, 6))
group = SUMMARY[SUMMARY.condition == 'disturbed']
for mode, marker in [('full', 'o'), ('roi', 's')]:
    subset = group[group.readout_mode == mode]
    ax.scatter(100 * subset.mean_censoring, 100 * subset.foreground_miou, marker=marker, s=60, label=mode)
    for row in subset.itertuples():
        ax.annotate(row.configuration, (100 * row.mean_censoring, 100 * row.foreground_miou), xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Mean full-sensor censoring (%)')
ax.set_ylabel('Foreground mIoU (%)')
ax.set_title('M6-F20 — Physical protection versus perception')
ax.legend()
ax.grid(alpha=0.2)
save_figure(fig, 'M6_F20_physical_semantic_tradeoff')

def json_records(frame):
    return json.loads(frame.to_json(orient='records'))
REPORT = {'task_id': 'M6-DEV-01D-04A', 'task_passed': True, 'research_checkpoint': CHECKPOINT, 'validation_scenes': len(SCENES), 'test_scenes_used': 0, 'sensor_acquisitions': ACQUISITIONS, 'perception_evaluations': len(DF), 'matched_acquisition_checks': PAIR_CHECKS, 'configuration_summary': json_records(SUMMARY), 'acquisition_factor_contrasts': json_records(FACTORS), 'residual_interference': json_records(RESIDUAL), 'model_state_unchanged': True, 'model_training_performed': False, 'controller_modified': False, 'dataset_modified': False, 'historical_modifications': False, 'limitations': ['32 selected synthetic validation scenes.', 'Diagnostic configurations, not optimized settings.', 'Static optical scenes and noise-free acquisitions.', 'Fixed disturbance geometry and intensity.', 'No temporal controller execution.', 'Post-ADC sample selection.', 'No physical hardware bandwidth measurement.', 'No real-world validation.', 'No independent confirmatory test of refinements.'], 'execution_seconds': time.perf_counter() - START}
REPORT_PATH = OUTPUT / 'ablation_report.json'
REPORT_PATH.write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')
verify_protected()
assert model_hashes() == INITIAL_MODEL_HASHES
assert sha256(MODEL_PATH) == MODEL_HASH
INVENTORY = {str(path.relative_to(ROOT)): {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in sorted(OUTPUT.rglob('*')) if path.is_file()}
MANIFEST_PATH = OUTPUT / 'ablation_manifest.json'
MANIFEST_PATH.write_text(json.dumps({'task_id': 'M6-DEV-01D-04A', 'research_checkpoint': CHECKPOINT, 'files': INVENTORY}, indent=2) + '\n')
NEW_PROTECTED = {}
for name, metadata in INVENTORY.items():
    path = ROOT / name
    assert sha256(path) == metadata['sha256']
    assert path.stat().st_size == metadata['bytes']
    NEW_PROTECTED[name] = metadata['sha256']
NEW_PROTECTED[str(MANIFEST_PATH.relative_to(ROOT))] = sha256(MANIFEST_PATH)
assert len(NEW_PROTECTED) == 14
verify_protected()
AHDS_M6_ABLATION_STATE = {**REPORT, 'report_path': str(REPORT_PATH.relative_to(ROOT)), 'manifest_path': str(MANIFEST_PATH.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST_PATH), 'protected_files': {**PROTECTED, **NEW_PROTECTED}, 'next_task': 'M6-DEV-01D-04B — Targeted Perception-Aware Refinement'}
print('\n' + '=' * 70)
print('CONTROLLED ACQUISITION ABLATION SUMMARY')
print('=' * 70)
print('Validation scenes: 32')
print('Test scenes used: 0')
print('Sensor acquisitions:', ACQUISITIONS)
print('Perception evaluations:', len(DF))
print('Matched acquisition checks:', PAIR_CHECKS)
print('\n[ARTIFACT INTEGRITY]')
print('Historical files: 166/166')
print('New artifacts:', len(NEW_PROTECTED))
print('Frozen model state: VERIFIED')
print('Model training: NONE')
print('Controller modifications: NONE')
print('Dataset modifications: NONE')
print('Artifact verification: PASS')
print('Git staging: NONE')
print('Protected inventory:', len(AHDS_M6_ABLATION_STATE['protected_files']))
print('\n[ARTIFACT IDENTITIES]')
for name, digest in NEW_PROTECTED.items():
    print(name)
    print('SHA256:', digest)
print('\nExecution time:', f'{(time.perf_counter() - START) / 60:.2f} minutes')
print('\nM6-DEV-01D-04A: ACQUISITION ABLATION PASSED')
print('Next: M6-DEV-01D-04B — Targeted Perception-Aware Refinement')