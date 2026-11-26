
"""AHDS multi-scenario optical input generation."""

from dataclasses import dataclass
import torch

from ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    generate_attack_map,
)

from ahds.module1_environment.optical_scenarios import (
    ILLUMINATION_PRESETS,
    apply_illumination_condition,
    generate_benign_light,
)


@dataclass
class ScenarioFrame:

    frame: int
    phase: str
    event_active: bool
    disturbance_active: bool

    benign_scene: torch.Tensor
    observed_scene: torch.Tensor

    disturbance_map: torch.Tensor
    benign_light_map: torch.Tensor


class RobustnessScenario:

    def __init__(self, spec):

        self.spec = spec

        self.height = int(spec["height"])
        self.width = int(spec["width"])

        self.total_frames = int(spec["total_frames"])
        self.fps = float(spec["fps"])

        self.windows = [
            tuple(window)
            for window in spec["event_windows"]
        ]

        self.event_type = spec["event_type"]

        assert self.fps > 0

        previous_end = 0

        for start, end in self.windows:

            assert previous_end <= start < end
            assert end <= self.total_frames

            previous_end = end

        self.base_scene = generate_clean_scene(
            height=self.height,
            width=self.width,
            device="cpu"
        )

        illumination = spec["illumination"]

        if illumination != "canonical":

            self.base_scene = apply_illumination_condition(
                self.base_scene,
                ILLUMINATION_PRESETS[illumination]
            )

        self.sources = [
            AttackConfig(**source)
            for source in spec["sources"]
        ]

        for source in self.sources:
            source.validate()

        self.zero_map = torch.zeros(
            (1, self.height, self.width),
            dtype=self.base_scene.dtype
        )

        self.regions = self._build_regions()


    def active_window(self, frame):

        for start, end in self.windows:

            if start <= frame < end:
                return start, end

        return None


    def phase(self, frame):

        if self.active_window(frame) is not None:

            if self.event_type == "disturbance":
                return "disturbance"

            return "benign_event"

        if any(end <= frame for _, end in self.windows):
            return "recovery"

        return "clean"


    def _disturbance_map(self, elapsed_frames):

        result = self.zero_map.clone()

        time_seconds = elapsed_frames / self.fps

        for source in self.sources:

            result += generate_attack_map(
                self.height,
                self.width,
                source,
                device="cpu",
                time_step=time_seconds
            )

        return result


    def _build_regions(self):

        full = torch.ones(
            (1, self.height, self.width),
            dtype=torch.bool
        )

        if self.event_type != "disturbance":
            return {"full": full}

        # Fixed peak envelope for all disturbance episodes.
        envelope = self.zero_map.clone()

        max_duration = max(
            end - start
            for start, end in self.windows
        )

        for elapsed in range(max_duration):

            envelope = torch.maximum(
                envelope,
                self._disturbance_map(elapsed)
            )

        peak = float(envelope.max())

        assert peak > 0

        relative = envelope / peak

        return {
            "full": full,
            "core": relative >= 0.10,
            "transition": (
                (relative >= 0.01)
                & (relative < 0.10)
            ),
            "background": relative < 0.01,
        }


    def frame(self, frame):

        if not 0 <= frame < self.total_frames:
            raise IndexError(frame)

        window = self.active_window(frame)

        event_active = (
            window is not None
            and self.event_type != "none"
        )

        benign_scene = self.base_scene.clone()

        disturbance_map = self.zero_map.clone()
        benign_light_map = self.zero_map.clone()

        if event_active:

            elapsed = frame - window[0]

            if self.event_type == "disturbance":

                disturbance_map = (
                    self._disturbance_map(elapsed)
                )

            elif self.event_type == "benign_global":

                benign_scene = (
                    benign_scene
                    * self.spec["benign_scale"]
                )

            elif self.event_type == "benign_local":

                source = self.sources[0]

                benign_light_map = generate_benign_light(
                    self.height,
                    self.width,
                    center_x=source.center_x,
                    center_y=source.center_y,
                    intensity=source.intensity,
                    radius=source.radius,
                    device="cpu"
                )

                benign_scene = (
                    benign_scene + benign_light_map
                )

        observed_scene = (
            benign_scene + disturbance_map
        )

        # Ground truth refers to actual incident interference,
        # not merely the scheduled event window.
        disturbance_active = bool(
            disturbance_map.max().item() > 1e-7
        )

        return ScenarioFrame(
            frame=frame,
            phase=self.phase(frame),
            event_active=event_active,
            disturbance_active=disturbance_active,
            benign_scene=benign_scene,
            observed_scene=observed_scene,
            disturbance_map=disturbance_map,
            benign_light_map=benign_light_map,
        )
