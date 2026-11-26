"""AHDS controlled five-strategy closed-loop experiment.

Implements the frozen M5-VAL-01B fairness contract.

Importing this module does not execute experiments or
write result files.
"""

from pathlib import Path
from dataclasses import asdict, replace
from itertools import combinations
from copy import deepcopy

import hashlib
import json
import math
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    apply_optical_attack,
)

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor,
)

from src.ahds.module4_anomaly_detection.disturbance_assessor import (
    DisturbanceConfig,
    PhysicalDisturbanceAssessor,
)

from src.ahds.module4_anomaly_detection.disturbance_state_tracker import (
    StateTrackerConfig,
    DisturbanceStateTracker,
)

from src.ahds.module5_defense.exposure_protection import (
    ExposureProtectionConfig,
    ExposureProtectionController,
)

from src.ahds.module5_defense.probed_exposure import (
    ProbedExposureConfig,
    ProbedExposureController,
)

from src.ahds.module5_defense.state_aware_exposure import (
    StateAwareExposureConfig,
    StateAwareExposureController,
)

from src.ahds.module5_defense.disturbance_aware_defense import (
    DisturbanceAwareDefenseConfig,
    DisturbanceAwareDefenseController,
)

from src.ahds.module7_evaluation.information_preservation import (
    InformationMetricConfig,
    InformationPreservationMetrics,
)


CONTRACT_PATH = (
    ROOT / "docs/module5/"
    "five_strategy_fairness_contract.json"
)

RESULT_DIR = (
    ROOT / "results/module5/five_strategy_comparison"
)

CONTRACT_HASH = (
    "fea57bb9bad661b7e849be32f277241bca"
    "17329b9b390c91b8116e329ca98636"
)

RUNTIME_HASH = (
    "a4cb2057780ef9f1fd87989467d2d5ee33"
    "cc8239ce1e7f0db619b1a865521291"
)

REGION_HASH = (
    "3bd84b2477493ee3cf29cf64ba1ce1e81"
    "e645b4b2e69c06daefa7fc345eae3b1"
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor):

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def read_verified_json(path, expected_hash):

    if file_hash(path) != expected_hash:
        raise RuntimeError(f"Artifact identity changed: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value):

    if isinstance(value, torch.Tensor):

        if value.numel() != 1:
            raise ValueError("Expected scalar tensor")

        value = value.item()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite scalar")

    return value


class FiveStrategyHarness:

    """Common deterministic defensive-controller benchmark."""

    def __init__(self):

        self.contract = read_verified_json(
            CONTRACT_PATH,
            CONTRACT_HASH
        )

        self.runtime = read_verified_json(
            RESULT_DIR / "runtime_configuration_snapshot.json",
            RUNTIME_HASH
        )

        self.region_report = read_verified_json(
            RESULT_DIR / "experimental_region_statistics.json",
            REGION_HASH
        )

        self.environment = self.contract["environment"]

        self.defaults = (
            self.contract["source_configuration_defaults"]
        )

        self.strategies = self.contract["strategies"]

        self.metric_names = set(
            self.contract["evaluation"]["metric_fields"]
        )

        self.permitted_fields = set(
            self.contract["controller_information_boundary"]
            ["permitted_sensor_fields"]
        )

        self.seed = self.environment["seed"]

        self.initial_config = SensorConfig(
            **self.environment["initial_sensor_config"]
        )

        self.attack_config = AttackConfig(
            **self.environment["attack_config"]
        )

        self.initial_config.validate()
        self.attack_config.validate()

        assert not self.environment["sensor_noise_enabled"]

        assert not self.environment["module3_adaptation_enabled"]

        assert self.environment["command_delay_frames"] == 1

        self._verify_sources()

        self._build_optical_inputs()

        self.reference = self.capture(
            self.clean_scene,
            self.initial_config
        )

        self.evaluator = InformationPreservationMetrics(
            InformationMetricConfig(**self.defaults["metrics"])
        )


    def _verify_sources(self):

        for item in self.contract["source_inventory"].values():

            assert file_hash(ROOT / item["path"]) == (
                item["sha256"]
            )

        for item in self.runtime["dadrc_internal_sources"].values():

            assert file_hash(ROOT / item["path"]) == (
                item["sha256"]
            )


    def _build_optical_inputs(self):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        h = self.environment["height"]
        w = self.environment["width"]

        self.clean_scene = generate_clean_scene(
            height=h,
            width=w,
            device=self.environment["device"]
        )

        self.disturbed_scene, self.attack_map = apply_optical_attack(
            self.clean_scene,
            self.attack_config,
            time_step=self.environment[
                "canonical_attack_map_time_step"
            ]
        )

        assert tensor_hash(self.clean_scene) == (
            self.region_report["clean_scene_sha256"]
        )

        assert tensor_hash(self.disturbed_scene) == (
            self.region_report["disturbed_scene_sha256"]
        )

        assert tensor_hash(self.attack_map) == (
            self.region_report["canonical_attack_map_sha256"]
        )

        peak = float(self.attack_map.max().item())

        assert math.isfinite(peak) and peak > 0

        specifications = (
            self.contract["spatial_evaluation"]["regions"]
        )

        high = peak * specifications["core"][
            "minimum_relative_peak"
        ]

        low = peak * specifications["transition"][
            "minimum_relative_peak"
        ]

        attack = self.attack_map.squeeze(0)

        self.regions = {

            "full": torch.ones_like(
                attack, dtype=torch.bool
            ),

            "core": attack >= high,

            "transition": (
                (attack >= low) & (attack < high)
            ),

            "background": attack < low
        }

        for name, mask in self.regions.items():

            assert tensor_hash(mask) == (
                self.region_report["regions"][name]
                ["mask_sha256"]
            )

        subregions = [
            self.regions[name]
            for name in ("core", "transition", "background")
        ]

        for first, second in combinations(subregions, 2):

            assert not (first & second).any().item()

        assert torch.equal(
            subregions[0] | subregions[1] | subregions[2],
            self.regions["full"]
        )


    def capture(self, scene, config):

        return simulate_sensor(
            scene,
            config=config,
            add_noise=False,
            seed=self.seed
        )


    def controller_view(self, result):

        view = {
            name: result[name].detach().clone()
            for name in sorted(self.permitted_fields)
        }

        assert set(view) == self.permitted_fields

        return view


    def phase(self, frame):

        for name, limits in self.environment["phases"].items():

            if limits["start"] <= frame < limits["stop_exclusive"]:
                return name

        raise ValueError(f"Invalid frame: {frame}")


    def build_strategy(self, strategy):

        assessor = PhysicalDisturbanceAssessor(
            DisturbanceConfig(**self.defaults["assessor"])
        )

        tracker = DisturbanceStateTracker(
            StateTrackerConfig(**self.defaults["tracker"])
        )

        spec = self.strategies[strategy]

        if strategy == "S0":

            controller = None

        elif strategy == "S1":

            controller = ExposureProtectionController(
                ExposureProtectionConfig(**spec["config"])
            )

        elif strategy == "S2":

            controller = ProbedExposureController(
                ProbedExposureConfig(**spec["config"])
            )

        elif strategy == "S3":

            controller = StateAwareExposureController(
                StateAwareExposureConfig(**spec["config"])
            )

        elif strategy == "S4":

            controller = DisturbanceAwareDefenseController(
                DisturbanceAwareDefenseConfig(**spec["config"])
            )

            expected = self.runtime[
                "dadrc_internal_configurations"
            ]

            assert asdict(controller.estimator.config) == (
                expected["observation_estimator"]["configuration"]
            )

            assert asdict(controller.memory.config) == (
                expected["disturbance_memory"]["configuration"]
            )

            startup = self.capture(
                self.clean_scene,
                self.initial_config
            )

            assert startup is not self.reference

            assert startup["normalized_output"].data_ptr() != (
                self.reference["normalized_output"].data_ptr()
            )

            startup_view = self.controller_view(startup)

            assert not (
                startup_view["saturated"]
                | startup_view["adc_clipped"]
            ).any().item()

            controller.initialize_reference(
                startup_view,
                replace(self.initial_config)
            )

        else:

            raise ValueError(strategy)

        return {
            "assessor": assessor,
            "tracker": tracker,
            "controller": controller
        }


    def command(
        self,
        strategy,
        components,
        view,
        config,
        assessment,
        tracker_state
    ):

        controller = components["controller"]

        if strategy == "S0":

            result = {
                "exposure_ms":
                    self.strategies["S0"]["fixed_exposure_ms"],
                "action": "FIXED"
            }

        elif strategy in ("S1", "S2"):

            result = controller.update(
                config.exposure_ms,
                assessment
            )

        elif strategy == "S3":

            result = controller.update(
                config.exposure_ms,
                assessment,
                tracker_state
            )

        elif strategy == "S4":

            result = controller.update(
                view,
                replace(config),
                assessment,
                tracker_state
            )

        else:

            raise ValueError(strategy)

        exposure = float(result["exposure_ms"])

        if strategy == "S0":

            assert exposure == self.strategies["S0"][
                "fixed_exposure_ms"
            ]

        else:

            limits = self.strategies[strategy]["config"]

            assert math.isfinite(exposure)

            assert (
                limits["min_exposure_ms"]
                <= exposure
                <= limits["max_exposure_ms"]
            )

        assert isinstance(result["action"], str)

        return result


    def evaluate_regions(
        self,
        strategy,
        frame,
        phase,
        observed,
        counterfactual,
        config
    ):

        records = []

        for name, mask in self.regions.items():

            metrics = self.evaluator.evaluate(
                reference_result=self.reference,
                counterfactual_result=counterfactual,
                observed_result=observed,
                reference_config=self.initial_config,
                applied_config=config,
                clean_optical_scene=self.clean_scene,
                region_mask=mask
            )

            assert set(metrics) == self.metric_names

            assert metrics["region_sample_count"] == (
                self.region_report["regions"][name]
                ["rgb_sample_count"]
            )

            json.dumps(metrics, allow_nan=False)

            records.append({
                "strategy": strategy,
                "frame": frame,
                "phase": phase,
                "region": name,
                **metrics
            })

        return records


    def run_strategy(self, strategy, frame_count=None):

        total = self.environment["total_frames"]

        if frame_count is None:
            frame_count = total

        if not 1 <= frame_count <= total:
            raise ValueError("Invalid frame count")

        components = self.build_strategy(strategy)

        applied_exposure = self.initial_config.exposure_ms

        frame_records = []
        regional_records = []
        observations = []

        for frame in range(frame_count):

            phase = self.phase(frame)

            attack_active = phase == "disturbance"

            config = replace(
                self.initial_config,
                exposure_ms=applied_exposure
            )

            if frame > 0:

                assert applied_exposure == (
                    frame_records[-1]["next_exposure_ms"]
                )

            actual_scene = (
                self.disturbed_scene
                if attack_active
                else self.clean_scene
            )

            observed = self.capture(actual_scene, config)

            view = self.controller_view(observed)

            assessment = components["assessor"].detect(view)

            tracker_result = components["tracker"].update(
                assessment
            )

            tracker_state = tracker_result["state"]

            assert tracker_state in (
                "NORMAL", "ACTIVE", "RECOVERY", "TRANSIENT"
            )

            command = self.command(
                strategy,
                components,
                view,
                config,
                assessment,
                tracker_state
            )

            next_exposure = float(command["exposure_ms"])

            # Evaluation-only counterfactual.
            # It is created after the controller command.
            counterfactual = self.capture(
                self.clean_scene,
                config
            )

            regional_records.extend(
                self.evaluate_regions(
                    strategy,
                    frame,
                    phase,
                    observed,
                    counterfactual,
                    config
                )
            )

            valid = ~(
                observed["saturated"]
                | observed["adc_clipped"]
            )

            adc_max = 2 ** config.adc_bits - 1

            optical_scale = (
                config.exposure_ms
                * config.gain
                * config.electrons_per_unit_ms
                / config.full_well_electrons
            )

            reconstructed = (
                observed["adc_output"].double()
                / adc_max
                / optical_scale
            )

            absolute_error = (
                reconstructed
                - self.clean_scene.double()
            ).abs()

            observations.append({
                "frame": frame,
                "valid": valid.detach().clone(),
                "absolute_error": absolute_error
            })

            diagnostic_fields = (
                "controller_state",
                "protection_active",
                "reference_exposure",
                "safe_exposure",
                "probe_pending",
                "clear_frames",
                "persistent_frames",
                "cooldown_remaining",
                "evidence_status",
                "evidence_reason",
                "optical_changed_fraction",
                "global_scale",
                "common_valid_fraction",
                "physical_clipping_fraction"
            )

            diagnostics = {
                name: scalar(command.get(name))
                for name in diagnostic_fields
            }

            frame_record = {

                "strategy": strategy,
                "frame": frame,
                "phase": phase,
                "attack_active": attack_active,

                "applied_exposure_ms": applied_exposure,
                "next_exposure_ms": next_exposure,

                "gain": config.gain,

                "action": command["action"],

                "tracker_state": tracker_state,

                "disturbance_detected":
                    bool(assessment["disturbance_detected"]),

                "spatial_saturation_fraction":
                    float(assessment["saturation_fraction"]),

                "saturation_ratio":
                    float(observed["saturation_ratio"]),

                "adc_clipping_ratio":
                    float(observed["adc_clip_ratio"]),

                "censored_ratio":
                    (~valid).double().mean().item(),

                **diagnostics
            }

            json.dumps(frame_record, allow_nan=False)

            frame_records.append(frame_record)

            # The command affects the next frame only.
            applied_exposure = next_exposure

        assert len(frame_records) == frame_count

        assert len(regional_records) == (
            frame_count * len(self.regions)
        )

        return {
            "frames": frame_records,
            "regional_metrics": regional_records,
            "observations": observations
        }


    def matched_record(
        self,
        strategy_ids,
        frame,
        region_name,
        runs
    ):

        mask = self.regions[region_name]

        expanded = mask.unsqueeze(0).expand_as(
            runs[strategy_ids[0]]["observations"][frame]["valid"]
        )

        common = expanded.clone()

        for strategy in strategy_ids:

            observation = runs[strategy]["observations"][frame]

            assert observation["frame"] == frame

            common &= observation["valid"]

        count = int(common.sum().item())

        total = int(expanded.sum().item())

        errors = {}

        for strategy in strategy_ids:

            error = runs[strategy]["observations"][frame][
                "absolute_error"
            ]

            errors[strategy] = (
                error[common].mean().item()
                if count else None
            )

        record = {

            "frame": frame,
            "phase": self.phase(frame),
            "region": region_name,

            "strategies": list(strategy_ids),

            "region_sample_count": total,

            "common_valid_sample_count": count,

            "common_valid_coverage": (
                count / total if total else None
            ),

            "radiance_mae": errors
        }

        json.dumps(record, allow_nan=False)

        return record


    def run_all(self, frame_count=None):

        if frame_count is None:

            frame_count = self.environment["total_frames"]

        runs = {
            strategy: self.run_strategy(strategy, frame_count)
            for strategy in self.strategies
        }

        frame_records = []
        regional_records = []
        pairwise_records = []
        all_strategy_records = []

        for result in runs.values():

            frame_records.extend(result["frames"])

            regional_records.extend(
                result["regional_metrics"]
            )

        pairs = list(combinations(self.strategies, 2))

        all_ids = tuple(self.strategies)

        for frame in range(frame_count):

            for region in self.regions:

                for pair in pairs:

                    pairwise_records.append(
                        self.matched_record(
                            pair, frame, region, runs
                        )
                    )

                all_strategy_records.append(
                    self.matched_record(
                        all_ids, frame, region, runs
                    )
                )

        expected = {
            "frames": 5 * frame_count,
            "regional_metrics": 20 * frame_count,
            "pairwise_matched": 40 * frame_count,
            "all_strategy_matched": 4 * frame_count
        }

        result = {
            "frames": frame_records,
            "regional_metrics": regional_records,
            "pairwise_matched": pairwise_records,
            "all_strategy_matched": all_strategy_records
        }

        for name, count in expected.items():

            assert len(result[name]) == count, name

        assert tensor_hash(self.clean_scene) == (
            self.region_report["clean_scene_sha256"]
        )

        assert tensor_hash(self.attack_map) == (
            self.region_report["canonical_attack_map_sha256"]
        )

        for name, mask in self.regions.items():

            assert tensor_hash(mask) == (
                self.region_report["regions"][name]["mask_sha256"]
            )

        json.dumps(result, allow_nan=False)

        return result
