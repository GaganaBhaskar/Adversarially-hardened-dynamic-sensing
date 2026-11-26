
# ============================================================
# AHDS — M5-DEF-02A-03A
# Archive Physical Observation Validation
# ============================================================

from pathlib import Path
from dataclasses import asdict
import json
import hashlib

ROOT = Path(
    "/content/adversarially-hardened-dynamic-sensing"
)

OUT = ROOT / "results/module5/exposure_aware_observation"
OUT.mkdir(parents=True, exist_ok=True)

assert len(checks) == 13
assert all(checks.values())

results.to_csv(
    OUT / "physical_reconstruction.csv",
    index=False
)

comparison_df.to_csv(
    OUT / "exposure_gain_invariance.csv",
    index=False
)

report = {
    "task": "M5-DEF-02A-02B",
    "validation_checks": len(checks),
    "passed_checks": sum(
        bool(v) for v in checks.values()
    ),
    "checks": {
        k: bool(v) for k, v in checks.items()
    },
    "recovered_rgb_samples": int(
        newly_observable_count
    ),
    "censored_samples_10ms": int(
        high_exposure["censored_mask"].sum().item()
    ),
    "censored_samples_5ms": int(
        low_exposure["censored_mask"].sum().item()
    ),
    "noise_enabled": False,
    "random_seed": 42,
    "image_size": [256, 256],
    "sensor_config": asdict(base_config),
    "attack_config": asdict(attack_config),
    "estimator_config": asdict(estimator.config),
    "numerical_tolerance": 1e-5,
    "limitations": [
        "Deterministic noise-free validation.",
        "Simplified linear RGB sensor model.",
        "No hardware calibration.",
        "No disturbance detection validation yet."
    ]
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2)
)

source = (
    ROOT / "src/ahds/module5_defense/"
    "exposure_aware_observation.py"
)

assert source.is_file()

manifest = {}

for path in [
    source,
    OUT / "physical_reconstruction.csv",
    OUT / "exposure_gain_invariance.csv",
    OUT / "validation_report.json"
]:
    content = path.read_bytes()

    manifest[str(path.relative_to(ROOT))] = {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest()
    }

(OUT / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print("=" * 70)
print("PHYSICAL OBSERVATION RESULTS ARCHIVAL")
print("=" * 70)

for path in sorted(OUT.iterdir()):
    print("[SAVED]", path.name)

print("\nSource implementation: VERIFIED")
print("Validation checks: 13/13")
print("Artifact manifest entries:", len(manifest))
print("Results directory:", OUT)

print("\nM5-DEF-02A-03A: PASSED")