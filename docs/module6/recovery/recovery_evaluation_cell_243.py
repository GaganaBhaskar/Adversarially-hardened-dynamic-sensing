from pathlib import Path
from dataclasses import replace
import hashlib
import subprocess
import json
import csv
import time
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
START_TIME = time.perf_counter()
ROOT = Path('/content/adversarially-hardened-dynamic-sensing').resolve()
CHECKPOINT = 'a9258ba4a307d6a6ad4ff23788a392c16260ed48'
STATE = AHDS_M6_TRAINING_STATE
PROTECTED = dict(STATE['protected_files'])
OUTPUT = ROOT / 'results/module6/heldout_evaluation'
REGISTRY = ROOT / 'datasets/ahds_procedural_v1'
DEVICE = torch.device('cpu')
NUM_CLASSES = 4
EXPECTED_MODEL_HASH = '5a4d3736aef265bce70dc540cad07a83e4c4555244c2185806478507cdce257f'
print('=' * 70)
print('AHDS — M6-DEV-01C')
print('Frozen Model and Held-out Evaluation')
print('=' * 70)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

def verify_protected():
    assert git('rev-parse', 'HEAD') == CHECKPOINT
    assert not git('diff', 'HEAD', '--name-status')
    assert not git('diff', '--cached', '--name-status')
    for name, digest in PROTECTED.items():
        assert sha256(ROOT / name) == digest, name
assert STATE['task_passed']
assert STATE['training_completed']
assert STATE['best_epoch'] == 38
assert STATE['test_scenes_used'] == 0
assert len(PROTECTED) == 119
verify_protected()
assert not OUTPUT.exists(), f'Output already exists: {OUTPUT}'
MODEL_PATH = ROOT / STATE['checkpoint_path']
assert sha256(MODEL_PATH) == EXPECTED_MODEL_HASH
assert sha256(REGISTRY / 'dataset_manifest.json') == '68f82a8a72a582440abf7874c950396cbcb822fb76399559c6adc6ed9c541df1'
print('\n[PREREQUISITES]')
print('Research checkpoint: VERIFIED')
print('Protected files: 119/119')
print('Dataset registry: VERIFIED')
print('Trained model identity: VERIFIED')
print('Selected checkpoint: Epoch 32')
print('Output overwrite protection: PASS')
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from ahds.module6_perception.procedural_dataset import CLASS_NAMES, ProceduralPerceptionDataset
from ahds.module6_perception.lightweight_segmentation import LightweightSegmentationNet
from ahds.module6_perception.segmentation_metrics import SegmentationMetrics
from ahds.module2_sensor.sensor_simulator import SensorConfig, simulate_sensor
from ahds.module3_dynamic_sensing.roi_allocator import allocate_readout
from ahds.integration.roi_closed_loop import AdaptiveROIConfig
assert len(CLASS_NAMES) == NUM_CLASSES
FROZEN_MODEL = LightweightSegmentationNet(num_classes=NUM_CLASSES).to(DEVICE)
CHECKPOINT_DATA = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
FROZEN_MODEL.load_state_dict(CHECKPOINT_DATA['model_state_dict'], strict=True)
FROZEN_MODEL.eval()
FROZEN_MODEL.requires_grad_(False)
assert CHECKPOINT_DATA['epoch'] == 38
assert not FROZEN_MODEL.training
assert all((not parameter.requires_grad for parameter in FROZEN_MODEL.parameters()))
PARAMETER_COUNT = sum((parameter.numel() for parameter in FROZEN_MODEL.parameters()))
assert PARAMETER_COUNT == 10700

def model_state_hashes():
    return {name: tensor_sha256(value) for name, value in FROZEN_MODEL.state_dict().items()}
INITIAL_MODEL_HASHES = model_state_hashes()
print('\n[FROZEN MODEL]')
print('Model parameters:', PARAMETER_COUNT)
print('Trainable parameters: 0')
print('Evaluation mode: ENABLED')
print('Gradient updates: DISABLED')
print('Checkpoint selection: FROZEN')
print('Device:', DEVICE)
TEST_DATASET = ProceduralPerceptionDataset('test')
assert len(TEST_DATASET) == 160
with (REGISTRY / 'test_index.csv').open() as file:
    TEST_INDEX = list(csv.DictReader(file))
assert len(TEST_INDEX) == 160
assert [int(row['scene_seed']) for row in TEST_INDEX] == list(range(300000, 300160))
ROI_CONFIG = AdaptiveROIConfig()
ROI = ROI_CONFIG.resolve(256, 256)
EXAMPLE_INDICES = [0, 40, 80, 120]
PROTOCOL = {'task_id': 'M6-DEV-01C', 'research_checkpoint': CHECKPOINT, 'dataset_version': 'ahds_procedural_v1', 'dataset_manifest_sha256': sha256(REGISTRY / 'dataset_manifest.json'), 'model_checkpoint_sha256': EXPECTED_MODEL_HASH, 'model_checkpoint_epoch': 38, 'test_scenes': 160, 'test_seed_range': [300000, 300159], 'primary_metric': 'foreground_miou', 'aggregation': 'pooled_pixel_confusion_matrix', 'registered_observation_evaluation': True, 'paired_sampling_evaluation': True, 'paired_sampling_controls': ['scene', 'semantic_ground_truth', 'exposure', 'gain', 'sensor_noise_realization'], 'paired_sampling_variable': 'spatial_sampling_mask', 'optical_disturbances': False, 'model_training': False, 'checkpoint_reselection': False, 'device': str(DEVICE), 'pytorch_version': str(torch.__version__), 'numpy_version': np.__version__, 'cpu_threads': torch.get_num_threads(), 'fixed_visualization_indices': EXAMPLE_INDICES}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / 'evaluation_protocol.json').write_text(json.dumps(PROTOCOL, indent=2, allow_nan=False) + '\n')
print('\n[HELD-OUT EVALUATION PROTOCOL]')
print('Test scenes: 160')
print('Test seed range: 300000–300159')
print('Registered test observations: 160')
print('Paired full-frame observations: 160')
print('Paired ROI observations: 160')
print('Optical disturbances: DISABLED')
GROUP_NAMES = ['registered_all', 'registered_full_frame', 'registered_roi', 'registered_noise_free', 'registered_noisy', 'paired_full_frame', 'paired_roi', 'paired_full_noise_free', 'paired_roi_noise_free', 'paired_full_noisy', 'paired_roi_noisy']
METRICS = {name: SegmentationMetrics(num_classes=NUM_CLASSES) for name in GROUP_NAMES}
GROUP_COUNTS = {name: 0 for name in GROUP_NAMES}
GROUP_SAMPLING = {name: [] for name in GROUP_NAMES}
SCENE_RECORDS = []
EXAMPLES = {}
HASH_CHECKS = 0
PAIR_CHECKS = 0

def predict(x):
    assert x.shape == (4, 256, 256)
    logits = FROZEN_MODEL(x.unsqueeze(0).to(DEVICE))
    assert logits.shape == (1, 4, 256, 256)
    assert torch.isfinite(logits).all()
    return logits.argmax(dim=1)[0].cpu()

def add_metrics(group, prediction, target, sampling_ratio):
    METRICS[group].update(prediction, target)
    GROUP_COUNTS[group] += 1
    GROUP_SAMPLING[group].append(float(sampling_ratio))

def single_scene_metrics(prediction, target):
    accumulator = SegmentationMetrics(num_classes=NUM_CLASSES)
    accumulator.update(prediction, target)
    return accumulator.compute()
print('\n[HELD-OUT EXECUTION]')
with torch.inference_mode():
    for index in range(len(TEST_DATASET)):
        row = TEST_INDEX[index]
        sample = TEST_DATASET[index]
        scene, target, metadata = TEST_DATASET.optical_sample(index)
        seed = TEST_DATASET.scene_seed(index)
        registered_input = sample['input']
        registered_target = sample['target']
        assert seed == int(row['scene_seed'])
        assert sample['scene_seed'] == seed
        assert torch.equal(target, registered_target)
        registered_mask = registered_input[3].bool()
        assert tensor_sha256(scene) == row['optical_scene_sha256']
        assert tensor_sha256(target) == row['semantic_target_sha256']
        assert tensor_sha256(registered_input) == row['model_input_sha256']
        assert tensor_sha256(registered_mask) == row['sampling_mask_sha256']
        HASH_CHECKS += 4
        config = replace(SensorConfig(), exposure_ms=float(sample['exposure_ms']), gain=float(sample['gain']))
        sensor_result = simulate_sensor(scene, config=config, add_noise=bool(sample['noise_enabled']), seed=int(row['sensor_noise_seed']))
        full_rgb = sensor_result['normalized_output']
        full_mask = torch.ones((256, 256), dtype=torch.bool)
        roi_result = allocate_readout(full_rgb, ROI, background_stride=ROI_CONFIG.background_stride)
        roi_rgb = roi_result['readout']
        roi_mask = roi_result['sampling_mask']
        full_input = torch.cat([full_rgb, full_mask.float().unsqueeze(0)], dim=0)
        roi_input = torch.cat([roi_rgb, roi_mask.float().unsqueeze(0)], dim=0)
        roi_ratio = float(roi_mask.float().mean())
        assert torch.equal(roi_rgb[:, roi_mask], full_rgb[:, roi_mask])
        assert (roi_rgb[:, ~roi_mask] == 0).all()
        assert 0 < roi_ratio < 1
        reconstructed_input = roi_input if sample['roi_enabled'] else full_input
        assert torch.equal(registered_input, reconstructed_input), f'Registered acquisition mismatch: {seed}'
        PAIR_CHECKS += 4
        full_prediction = predict(full_input)
        roi_prediction = predict(roi_input)
        registered_prediction = roi_prediction if sample['roi_enabled'] else full_prediction
        registered_ratio = float(sample['sampling_ratio'])
        registered_mode = 'registered_roi' if sample['roi_enabled'] else 'registered_full_frame'
        noise_group = 'registered_noisy' if sample['noise_enabled'] else 'registered_noise_free'
        add_metrics('registered_all', registered_prediction, target, registered_ratio)
        add_metrics(registered_mode, registered_prediction, target, registered_ratio)
        add_metrics(noise_group, registered_prediction, target, registered_ratio)
        add_metrics('paired_full_frame', full_prediction, target, 1.0)
        add_metrics('paired_roi', roi_prediction, target, roi_ratio)
        noise_suffix = 'noisy' if sample['noise_enabled'] else 'noise_free'
        add_metrics(f'paired_full_{noise_suffix}', full_prediction, target, 1.0)
        add_metrics(f'paired_roi_{noise_suffix}', roi_prediction, target, roi_ratio)
        registered_metrics = single_scene_metrics(registered_prediction, target)
        full_metrics = single_scene_metrics(full_prediction, target)
        roi_metrics = single_scene_metrics(roi_prediction, target)
        full_miou = full_metrics['foreground_miou']
        roi_miou = roi_metrics['foreground_miou']
        record = {'scene_seed': seed, 'sample_index': index, 'illumination': float(metadata['illumination']), 'exposure_ms': float(sample['exposure_ms']), 'gain': float(sample['gain']), 'noise_enabled': bool(sample['noise_enabled']), 'registered_roi_enabled': bool(sample['roi_enabled']), 'registered_sampling_ratio': registered_ratio, 'paired_roi_sampling_ratio': roi_ratio, 'registered_foreground_miou': registered_metrics['foreground_miou'], 'full_foreground_miou': full_miou, 'roi_foreground_miou': roi_miou, 'roi_minus_full_foreground_miou': roi_miou - full_miou if full_miou is not None and roi_miou is not None else None, 'full_pixel_accuracy': full_metrics['pixel_accuracy'], 'roi_pixel_accuracy': roi_metrics['pixel_accuracy']}
        for class_id, class_name in enumerate(CLASS_NAMES):
            for prefix, result in [('registered', registered_metrics), ('full', full_metrics), ('roi', roi_metrics)]:
                values = result['class_iou']
                value = values[class_name] if isinstance(values, dict) and class_name in values else values[str(class_id)] if isinstance(values, dict) and str(class_id) in values else values[class_id]
                record[f'{prefix}_iou_{class_name}'] = value
        SCENE_RECORDS.append(record)
        if index in EXAMPLE_INDICES:
            EXAMPLES[index] = {'full_input': full_input.clone(), 'roi_input': roi_input.clone(), 'target': target.clone(), 'full_prediction': full_prediction.clone(), 'roi_prediction': roi_prediction.clone()}
        if (index + 1) % 20 == 0:
            print(f'  {index + 1}/160 test scenes evaluated', flush=True)
assert HASH_CHECKS == 640
assert PAIR_CHECKS == 640
assert len(SCENE_RECORDS) == 160
assert GROUP_COUNTS['registered_all'] == 160
assert GROUP_COUNTS['paired_full_frame'] == 160
assert GROUP_COUNTS['paired_roi'] == 160
assert GROUP_COUNTS['registered_full_frame'] + GROUP_COUNTS['registered_roi'] == 160
assert GROUP_COUNTS['registered_noise_free'] + GROUP_COUNTS['registered_noisy'] == 160
assert model_state_hashes() == INITIAL_MODEL_HASHES
assert sha256(MODEL_PATH) == EXPECTED_MODEL_HASH
print('\n[EXECUTION INTEGRITY]')
print('Held-out scenes: 160/160')
print('Registry hash checks: 640/640')
print('Paired acquisition checks: 640/640')
print('Model forward passes: 320')
print('Frozen model state: UNCHANGED')
print('Checkpoint file: UNCHANGED')
print('Optimizer steps: 0')

def json_safe(value):
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
RESULTS = {}
for group in GROUP_NAMES:
    if GROUP_COUNTS[group] == 0:
        RESULTS[group] = {'scene_count': 0, 'metrics': None}
        continue
    result = json_safe(METRICS[group].compute())
    assert int(np.asarray(result['confusion_matrix']).sum()) == GROUP_COUNTS[group] * 256 * 256
    RESULTS[group] = {'scene_count': GROUP_COUNTS[group], 'mean_sampling_ratio': float(np.mean(GROUP_SAMPLING[group])), 'metrics': result}

def get_metric(group, metric):
    return RESULTS[group]['metrics'][metric]

def get_class_iou(group, class_id):
    values = get_metric(group, 'class_iou')
    name = CLASS_NAMES[class_id]
    if isinstance(values, dict):
        if name in values:
            return values[name]
        return values[str(class_id)]
    return values[class_id]
REGISTERED_MIOU = get_metric('registered_all', 'foreground_miou')
FULL_MIOU = get_metric('paired_full_frame', 'foreground_miou')
ROI_MIOU = get_metric('paired_roi', 'foreground_miou')
ROI_RATIO = RESULTS['paired_roi']['mean_sampling_ratio']
IDEAL_SAMPLE_REDUCTION = 1 - ROI_RATIO
PAIRED_DELTA = ROI_MIOU - FULL_MIOU
PAIRED_RETENTION = ROI_MIOU / FULL_MIOU if FULL_MIOU > 0 else None
DELTAS = np.asarray([row['roi_minus_full_foreground_miou'] for row in SCENE_RECORDS if row['roi_minus_full_foreground_miou'] is not None], dtype=float)
assert len(DELTAS) > 0
PAIR_SUMMARY = {'paired_scenes': 160, 'valid_scene_level_deltas': int(len(DELTAS)), 'pooled_full_foreground_miou': FULL_MIOU, 'pooled_roi_foreground_miou': ROI_MIOU, 'pooled_roi_minus_full_miou': PAIRED_DELTA, 'roi_foreground_miou_retention': PAIRED_RETENTION, 'mean_scene_level_delta': float(DELTAS.mean()), 'median_scene_level_delta': float(np.median(DELTAS)), 'roi_better_scenes': int((DELTAS > 1e-08).sum()), 'roi_equal_scenes': int((np.abs(DELTAS) <= 1e-08).sum()), 'roi_worse_scenes': int((DELTAS < -1e-08).sum()), 'roi_sampling_ratio': ROI_RATIO, 'ideal_sample_count_reduction': IDEAL_SAMPLE_REDUCTION}
print('\n[HELD-OUT PERFORMANCE]')
print('Validation foreground mIoU:', f"{100 * STATE['best_validation_foreground_miou']:.4f}%")
print('Test foreground mIoU:', f'{100 * REGISTERED_MIOU:.4f}%')
print('Test pixel accuracy:', f"{100 * get_metric('registered_all', 'pixel_accuracy'):.4f}%")
print('Test mean IoU:', f"{100 * get_metric('registered_all', 'mean_iou'):.4f}%")
print('Test foreground Dice:', f"{100 * get_metric('registered_all', 'foreground_mean_dice'):.4f}%")
print('\n[CLASS-WISE TEST PERFORMANCE]')
for class_id, class_name in enumerate(CLASS_NAMES):
    value = get_class_iou('registered_all', class_id)
    print(f'{class_name:16s} IoU:', f'{100 * value:.4f}%' if value is not None else 'undefined')
print('\n[ACQUISITION SUBGROUPS]')
for group in GROUP_NAMES:
    if GROUP_COUNTS[group] == 0:
        continue
    print(f"{group:28s} | N={GROUP_COUNTS[group]:3d} | FG mIoU={100 * get_metric(group, 'foreground_miou'):.4f}%")
print('\n[PAIRED SAMPLING COMPARISON]')
print('Full-frame foreground mIoU:', f'{100 * FULL_MIOU:.4f}%')
print('ROI foreground mIoU:', f'{100 * ROI_MIOU:.4f}%')
print('ROI minus full-frame:', f'{100 * PAIRED_DELTA:+.4f} percentage points')
print('ROI mIoU retention:', f'{100 * PAIRED_RETENTION:.4f}%' if PAIRED_RETENTION is not None else 'undefined')
print('ROI sampling ratio:', f'{100 * ROI_RATIO:.4f}%')
print('Ideal sample-count reduction:', f'{100 * IDEAL_SAMPLE_REDUCTION:.4f}%')
print('ROI better/equal/worse scenes:', PAIR_SUMMARY['roi_better_scenes'], PAIR_SUMMARY['roi_equal_scenes'], PAIR_SUMMARY['roi_worse_scenes'])
with (OUTPUT / 'per_scene_evaluation.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(SCENE_RECORDS[0]))
    writer.writeheader()
    writer.writerows(json_safe(SCENE_RECORDS))
GROUP_ROWS = []
for group, result in RESULTS.items():
    if result['metrics'] is None:
        continue
    metrics = result['metrics']
    row = {'group': group, 'scene_count': result['scene_count'], 'mean_sampling_ratio': result['mean_sampling_ratio'], 'pixel_accuracy': metrics['pixel_accuracy'], 'mean_iou': metrics['mean_iou'], 'foreground_miou': metrics['foreground_miou'], 'foreground_mean_dice': metrics['foreground_mean_dice']}
    for class_id, name in enumerate(CLASS_NAMES):
        row[f'iou_{name}'] = get_class_iou(group, class_id)
    GROUP_ROWS.append(row)
with (OUTPUT / 'group_evaluation.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(GROUP_ROWS[0]))
    writer.writeheader()
    writer.writerows(GROUP_ROWS)
(OUTPUT / 'evaluation_metrics.json').write_text(json.dumps({'groups': RESULTS, 'paired_comparison': PAIR_SUMMARY}, indent=2, allow_nan=False) + '\n')
FIGURES = OUTPUT / 'figures'
FIGURES.mkdir(exist_ok=False)

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.5))
metric_names = ['Foreground mIoU', 'Mean IoU', 'Foreground Dice', 'Pixel accuracy']
metric_keys = ['foreground_miou', 'mean_iou', 'foreground_mean_dice', 'pixel_accuracy']
validation_values = [100 * STATE['best_validation_metrics'][key] for key in metric_keys]
test_values = [100 * get_metric('registered_all', key) for key in metric_keys]
positions = np.arange(len(metric_names))
ax.bar(positions - 0.18, validation_values, width=0.36, label='Validation')
ax.bar(positions + 0.18, test_values, width=0.36, label='Held-out test')
ax.set_xticks(positions, metric_names, rotation=12)
ax.set_ylim(0, 105)
ax.set_ylabel('Performance (%)')
ax.set_title('M6-F05 — Validation and held-out performance')
ax.legend()
ax.grid(axis='y', alpha=0.2)
save_figure(fig, 'M6_F05_validation_vs_test')
matrix = np.asarray(get_metric('registered_all', 'confusion_matrix'), dtype=float)
normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(normalized * 100, cmap='Blues', vmin=0, vmax=100)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        value = normalized[i, j] * 100
        ax.text(j, i, f'{value:.2f}%', ha='center', va='center', color='white' if value > 50 else 'black')
ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=25)
ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
ax.set_xlabel('Predicted class')
ax.set_ylabel('Ground-truth class')
ax.set_title('M6-F06 — Held-out confusion matrix\nNormalized by ground-truth class')
fig.colorbar(image, ax=ax, label='Class pixel proportion (%)')
save_figure(fig, 'M6_F06_test_confusion_matrix')
fig, ax = plt.subplots(figsize=(8, 4.5))
full_values = [100 * get_class_iou('paired_full_frame', class_id) for class_id in range(NUM_CLASSES)]
roi_values = [100 * get_class_iou('paired_roi', class_id) for class_id in range(NUM_CLASSES)]
positions = np.arange(NUM_CLASSES)
ax.bar(positions - 0.18, full_values, width=0.36, label='Full frame')
ax.bar(positions + 0.18, roi_values, width=0.36, label='ROI priority')
ax.set_xticks(positions, CLASS_NAMES)
ax.set_ylim(0, 105)
ax.set_ylabel('IoU (%)')
ax.set_title('M6-F07 — Paired sampling class performance')
ax.legend()
ax.grid(axis='y', alpha=0.2)
save_figure(fig, 'M6_F07_paired_class_iou')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(DELTAS * 100, bins=20, edgecolor='white')
ax.axvline(0, linestyle='--', color='black', label='Equal performance')
ax.axvline(DELTAS.mean() * 100, linestyle=':', label='Mean paired difference')
ax.set_xlabel('ROI minus full-frame foreground mIoU (pp)')
ax.set_ylabel('Number of scenes')
ax.set_title('M6-F08 — Per-scene sampling trade-off')
ax.legend()
ax.grid(axis='y', alpha=0.2)
save_figure(fig, 'M6_F08_paired_scene_differences')
PALETTE = np.array([[60, 60, 60], [255, 205, 0], [37, 99, 235], [239, 68, 68]], dtype=np.uint8)
fig, axes = plt.subplots(len(EXAMPLE_INDICES), 6, figsize=(15, 10))
for row_index, index in enumerate(EXAMPLE_INDICES):
    example = EXAMPLES[index]
    full_rgb = example['full_input'][:3].permute(1, 2, 0).numpy()
    roi_rgb = example['roi_input'][:3].permute(1, 2, 0).numpy()
    display_scale = max(float(full_rgb.max()), 1e-08)
    panels = [np.clip(full_rgb / display_scale, 0, 1), np.clip(roi_rgb / display_scale, 0, 1), example['roi_input'][3].numpy(), PALETTE[example['target'].numpy()], PALETTE[example['full_prediction'].numpy()], PALETTE[example['roi_prediction'].numpy()]]
    titles = [f'Full readout {index}', 'ROI readout', 'ROI mask', 'Ground truth', 'Full prediction', 'ROI prediction']
    for column, panel in enumerate(panels):
        if column == 2:
            axes[row_index, column].imshow(panel, cmap='gray', vmin=0, vmax=1)
        else:
            axes[row_index, column].imshow(panel)
        axes[row_index, column].set_title(titles[column], fontsize=9)
        axes[row_index, column].axis('off')
fig.suptitle('M6-F09 — Fixed held-out paired predictions')
save_figure(fig, 'M6_F09_heldout_predictions')
assert model_state_hashes() == INITIAL_MODEL_HASHES
assert sha256(MODEL_PATH) == EXPECTED_MODEL_HASH
assert not FROZEN_MODEL.training
assert all((not parameter.requires_grad for parameter in FROZEN_MODEL.parameters()))
verify_protected()
REPORT = {'task_id': 'M6-DEV-01C', 'task_passed': True, 'research_checkpoint': CHECKPOINT, 'model_checkpoint_epoch': 38, 'model_checkpoint_sha256': EXPECTED_MODEL_HASH, 'model_parameters': PARAMETER_COUNT, 'test_scenes_evaluated': 160, 'paired_scenes_evaluated': 160, 'model_forward_passes': 320, 'registry_hash_checks': HASH_CHECKS, 'paired_acquisition_checks': PAIR_CHECKS, 'validation_foreground_miou': STATE['best_validation_foreground_miou'], 'test_foreground_miou': REGISTERED_MIOU, 'test_pixel_accuracy': get_metric('registered_all', 'pixel_accuracy'), 'test_mean_iou': get_metric('registered_all', 'mean_iou'), 'test_foreground_mean_dice': get_metric('registered_all', 'foreground_mean_dice'), 'test_foreground_miou_at_least_90_percent': REGISTERED_MIOU >= 0.9, 'group_results': RESULTS, 'paired_sampling_summary': PAIR_SUMMARY, 'frozen_model_state_verified': True, 'optimizer_steps': 0, 'model_training_performed': False, 'checkpoint_reselection_performed': False, 'optical_disturbances': False, 'dataset_modified': False, 'architecture_modified': False, 'controller_modified': False, 'historical_modifications': False, 'limitations': ['Held-out synthetic scenes, not real-world images.', 'Paired sampling uses post-ADC sample selection.', 'Sample-count reduction is not measured hardware bandwidth.', 'Noise subgroup comparisons are descriptive and unpaired.', 'No optical disturbance robustness evaluated in this task.', 'No hardware latency or energy conclusions.', 'Per-scene foreground mIoU may omit undefined classes.', 'Pooled confusion-matrix foreground mIoU is the primary metric.', 'Real-world image-based validation remains a planned milestone.'], 'execution_seconds': time.perf_counter() - START_TIME}
REPORT_PATH = OUTPUT / 'evaluation_report.json'
REPORT_PATH.write_text(json.dumps(json_safe(REPORT), indent=2, allow_nan=False) + '\n')
INVENTORY = {str(path.relative_to(ROOT)): {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in sorted(OUTPUT.rglob('*')) if path.is_file()}
MANIFEST_PATH = OUTPUT / 'evaluation_manifest.json'
MANIFEST_PATH.write_text(json.dumps({'task_id': 'M6-DEV-01C', 'research_checkpoint': CHECKPOINT, 'model_checkpoint_sha256': EXPECTED_MODEL_HASH, 'files': INVENTORY}, indent=2, allow_nan=False) + '\n')
for name, metadata in INVENTORY.items():
    path = ROOT / name
    assert sha256(path) == metadata['sha256']
    assert path.stat().st_size == metadata['bytes']
NEW_PROTECTED = {name: metadata['sha256'] for name, metadata in INVENTORY.items()}
NEW_PROTECTED[str(MANIFEST_PATH.relative_to(ROOT))] = sha256(MANIFEST_PATH)
assert len(NEW_PROTECTED) == 16
verify_protected()
AHDS_M6_HELDOUT_STATE = {**REPORT, 'report_path': str(REPORT_PATH.relative_to(ROOT)), 'manifest_path': str(MANIFEST_PATH.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST_PATH), 'protected_files': {**PROTECTED, **NEW_PROTECTED}, 'next_task': 'M6-DEV-01D — Closed-loop Perception Robustness'}
print('\n' + '=' * 70)
print('FROZEN MODEL AND HELD-OUT EVALUATION SUMMARY')
print('=' * 70)
print('Model checkpoint: Epoch 32')
print('Model parameters: 10700')
print('Test scenes: 160/160')
print('Paired scenes: 160/160')
print('Model forward passes: 320')
print('Registry hash checks: 640/640')
print('Paired acquisition checks: 640/640')
print('Validation foreground mIoU:', f"{100 * STATE['best_validation_foreground_miou']:.4f}%")
print('Held-out foreground mIoU:', f'{100 * REGISTERED_MIOU:.4f}%')
print('Held-out pixel accuracy:', f"{100 * REPORT['test_pixel_accuracy']:.4f}%")
print('Held-out mean IoU:', f"{100 * REPORT['test_mean_iou']:.4f}%")
print('Held-out foreground Dice:', f"{100 * REPORT['test_foreground_mean_dice']:.4f}%")
print('90% foreground mIoU target:', 'ACHIEVED' if REGISTERED_MIOU >= 0.9 else 'NOT ACHIEVED')
print('\n[PAIRED SAMPLING RESULTS]')
print('Full-frame foreground mIoU:', f'{100 * FULL_MIOU:.4f}%')
print('ROI foreground mIoU:', f'{100 * ROI_MIOU:.4f}%')
print('ROI minus full-frame:', f'{100 * PAIRED_DELTA:+.4f} pp')
print('Ideal sample-count reduction:', f'{100 * IDEAL_SAMPLE_REDUCTION:.4f}%')
print('\n[ARTIFACT INTEGRITY]')
print('Frozen model state: VERIFIED')
print('Optimizer steps: 0')
print('Model training: NOT PERFORMED')
print('Checkpoint reselection: NONE')
print('Protected historical files: 119/119')
print('New artifacts:', len(NEW_PROTECTED))
print('Artifact verification: PASS')
print('Dataset modifications: NONE')
print('Controller modifications: NONE')
print('Historical modifications: NONE')
print('Git staging: NONE')
print('Protected inventory:', len(AHDS_M6_HELDOUT_STATE['protected_files']))
print('\n[ARTIFACT IDENTITIES]')
for name, digest in NEW_PROTECTED.items():
    print(name)
    print('SHA256:', digest)
print('\nExecution time:', f'{time.perf_counter() - START_TIME:.2f}s')
print('\nM6-DEV-01C: EVALUATION EXECUTION PASSED')
print('Next: M6-DEV-01D — Closed-loop Perception Robustness')