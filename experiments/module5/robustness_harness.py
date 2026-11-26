
"""Multi-scenario AHDS comparative evaluation."""

from dataclasses import replace
from itertools import combinations
import json
import math

import torch

from five_strategy_closed_loop import (
    FiveStrategyHarness,
    scalar,
)

from robustness_scenarios import RobustnessScenario

from ahds.module2_sensor.sensor_simulator import (
    simulate_sensor,
)


class RobustnessHarness(FiveStrategyHarness):

    def __init__(self, spec, trial_seed):

        # Preserve all archived controller configurations.
        super().__init__()

        self.spec = spec
        self.trial_seed = int(trial_seed)

        self.scenario = RobustnessScenario(spec)

        self.noise_enabled = bool(
            spec["noise_enabled"]
        )

        self.clean_scene = (
            self.scenario.base_scene.clone()
        )

        self.regions = {
            name: mask
            for name, mask in
            self.scenario.regions.items()
            if mask.any().item()
        }

        self.omitted_regions = [
            name
            for name, mask in
            self.scenario.regions.items()
            if not mask.any().item()
        ]

        self.reference = self.capture_role(
            self.clean_scene,
            self.initial_config,
            frame=0,
            role="reference"
        )


    def capture_role(
        self, scene, config, frame, role
    ):

        role_ids = {
            "observed": 1,
            "counterfactual": 2,
            "reference": 3,
            "startup": 4,
        }

        seed = (
            self.trial_seed * 100000
            + int(frame) * 10
            + role_ids[role]
        ) % (2**32)

        return simulate_sensor(
            scene,
            config,
            add_noise=self.noise_enabled,
            seed=seed
        )


    def capture(self, scene, config):

        # Used by the inherited startup initialization.
        # The parent constructor also calls capture.
        if not hasattr(self, "trial_seed"):

            return super().capture(scene, config)

        return self.capture_role(
            scene,
            config,
            frame=0,
            role="startup"
        )


    def regional_metrics(
        self,
        strategy,
        frame_data,
        observed,
        counterfactual,
        reference,
        config
    ):

        records = []

        for name, mask in self.regions.items():

            metrics = self.evaluator.evaluate(
                reference_result=reference,
                counterfactual_result=counterfactual,
                observed_result=observed,
                reference_config=self.initial_config,
                applied_config=config,
                clean_optical_scene=(
                    frame_data.benign_scene
                ),
                region_mask=mask
            )

            assert set(metrics) == self.metric_names

            assert metrics["region_sample_count"] == (
                int(mask.sum()) * 3
            )

            record = {
                "scenario_id": self.spec["scenario_id"],
                "seed": self.trial_seed,
                "strategy": strategy,
                "frame": frame_data.frame,
                "phase": frame_data.phase,
                "region": name,
                **metrics
            }

            json.dumps(record, allow_nan=False)

            records.append(record)

        return records


    def run_robust_strategy(
        self, strategy, frame_count=None
    ):

        total = self.scenario.total_frames

        if frame_count is None:
            frame_count = total

        assert 1 <= frame_count <= total

        components = self.build_strategy(strategy)

        applied_exposure = float(
            self.initial_config.exposure_ms
        )

        frames = []
        regional = []
        observations = []

        for frame in range(frame_count):

            data = self.scenario.frame(frame)

            config = replace(
                self.initial_config,
                exposure_ms=applied_exposure
            )

            if frame > 0:

                assert applied_exposure == (
                    frames[-1]["next_exposure_ms"]
                )

            observed = self.capture_role(
                data.observed_scene,
                config,
                frame,
                "observed"
            )

            view = self.controller_view(observed)

            assessment = (
                components["assessor"].detect(view)
            )

            tracker_result = (
                components["tracker"].update(
                    assessment
                )
            )

            tracker_state = tracker_result["state"]

            command = self.command(
                strategy,
                components,
                view,
                config,
                assessment,
                tracker_state
            )

            next_exposure = float(
                command["exposure_ms"]
            )

            assert math.isfinite(next_exposure)
            assert next_exposure > 0

            # Evaluation-only captures occur after the
            # controller has generated its command.

            reference = self.capture_role(
                data.benign_scene,
                self.initial_config,
                frame,
                "reference"
            )

            counterfactual = self.capture_role(
                data.benign_scene,
                config,
                frame,
                "counterfactual"
            )

            regional.extend(
                self.regional_metrics(
                    strategy,
                    data,
                    observed,
                    counterfactual,
                    reference,
                    config
                )
            )

            valid = ~(
                observed["saturated"]
                | observed["adc_clipped"]
            )

            adc_max = 2**config.adc_bits - 1

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
                - data.benign_scene.double()
            ).abs()

            observations.append({
                "valid": valid.detach().clone(),
                "absolute_error": absolute_error,
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
                "physical_clipping_fraction",
            )

            diagnostics = {
                name: scalar(command.get(name))
                for name in diagnostic_fields
            }

            record = {

                "scenario_id": self.spec["scenario_id"],
                "seed": self.trial_seed,
                "strategy": strategy,
                "frame": frame,
                "phase": data.phase,

                "event_active": data.event_active,
                "attack_active": data.disturbance_active,

                "disturbance_peak": float(
                    data.disturbance_map.max()
                ),

                "applied_exposure_ms": applied_exposure,
                "next_exposure_ms": next_exposure,
                "gain": config.gain,

                "action": command["action"],
                "tracker_state": tracker_state,

                "disturbance_detected": bool(
                    assessment["disturbance_detected"]
                ),

                "spatial_saturation_fraction": float(
                    assessment["saturation_fraction"]
                ),

                "saturation_ratio": float(
                    observed["saturation_ratio"]
                ),

                "adc_clipping_ratio": float(
                    observed["adc_clip_ratio"]
                ),

                "censored_ratio": (
                    (~valid).double().mean().item()
                ),

                **diagnostics
            }

            json.dumps(record, allow_nan=False)

            frames.append(record)

            applied_exposure = next_exposure

        assert len(frames) == frame_count

        assert len(regional) == (
            frame_count * len(self.regions)
        )

        return {
            "frames": frames,
            "regional_metrics": regional,
            "observations": observations
        }


    def matched_metrics(
        self, ids, frame, region, runs
    ):

        mask = self.regions[region]

        mask = mask.expand(
            3, -1, -1
        )

        common = mask.clone()

        for strategy in ids:

            common &= (
                runs[strategy]["observations"]
                [frame]["valid"]
            )

        sample_count = int(common.sum())

        region_count = int(mask.sum())

        errors = {}

        for strategy in ids:

            error = (
                runs[strategy]["observations"]
                [frame]["absolute_error"]
            )

            errors[strategy] = (
                float(error[common].mean())
                if sample_count > 0
                else None
            )

        return {

            "scenario_id": self.spec["scenario_id"],
            "seed": self.trial_seed,
            "frame": frame,
            "phase": self.scenario.phase(frame),
            "region": region,
            "strategies": list(ids),

            "region_sample_count": region_count,
            "matched_sample_count": sample_count,

            "matched_valid_coverage": (
                sample_count / region_count
            ),

            "matched_radiance_mae": errors,
        }


    def run_robust_all(self, frame_count=None):

        if frame_count is None:

            frame_count = self.scenario.total_frames

        runs = {}

        for strategy in self.strategies:

            runs[strategy] = (
                self.run_robust_strategy(
                    strategy,
                    frame_count
                )
            )

        frames = []
        regional = []
        pairwise = []
        all_matched = []

        ids = tuple(self.strategies)

        for result in runs.values():

            frames.extend(result["frames"])
            regional.extend(
                result["regional_metrics"]
            )

        for frame in range(frame_count):

            for region in self.regions:

                for pair in combinations(ids, 2):

                    pairwise.append(
                        self.matched_metrics(
                            pair,
                            frame,
                            region,
                            runs
                        )
                    )

                all_matched.append(
                    self.matched_metrics(
                        ids,
                        frame,
                        region,
                        runs
                    )
                )

        result = {
            "scenario_id": self.spec["scenario_id"],
            "seed": self.trial_seed,
            "frame_count": frame_count,
            "omitted_regions": self.omitted_regions,
            "frames": frames,
            "regional_metrics": regional,
            "pairwise_matched": pairwise,
            "all_strategy_matched": all_matched,
        }

        regions = len(self.regions)

        assert len(frames) == 5 * frame_count

        assert len(regional) == (
            5 * frame_count * regions
        )

        assert len(pairwise) == (
            10 * frame_count * regions
        )

        assert len(all_matched) == (
            frame_count * regions
        )

        json.dumps(result, allow_nan=False)

        return result
