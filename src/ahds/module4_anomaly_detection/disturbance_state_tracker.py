
from dataclasses import dataclass
import math


@dataclass
class StateTrackerConfig:
    active_saturation_threshold: float = 0.001
    recovery_frames: int = 2

    def validate(self):
        if (
            not math.isfinite(self.active_saturation_threshold)
            or not 0 <= self.active_saturation_threshold <= 1
        ):
            raise ValueError("Invalid saturation threshold.")

        if (
            not isinstance(self.recovery_frames, int)
            or isinstance(self.recovery_frames, bool)
            or self.recovery_frames < 1
        ):
            raise ValueError("Invalid recovery frame count.")


class DisturbanceStateTracker:

    def __init__(self, config=None):
        self.config = config or StateTrackerConfig()
        self.config.validate()
        self.reset()

    def reset(self):
        self.state = "NORMAL"
        self.clear_count = 0

    def update(self, assessment):

        saturation = assessment["saturation_fraction"]
        disturbance = bool(
            assessment["disturbance_detected"]
        )

        # Persistent saturation is strong evidence
        # of ongoing sensor impairment.
        persistent_impairment = (
            saturation >=
            self.config.active_saturation_threshold
        )

        if persistent_impairment:
            self.state = "ACTIVE"
            self.clear_count = 0

        elif self.state == "ACTIVE":
            self.state = "RECOVERY"
            self.clear_count = 1

        elif self.state == "RECOVERY":
            self.clear_count += 1

            if self.clear_count >= self.config.recovery_frames:
                self.state = "NORMAL"
                self.clear_count = 0

        elif disturbance:
            # Transient disturbance without persistent
            # saturation: record as an active event.
            self.state = "TRANSIENT"

        else:
            self.state = "NORMAL"

        # TRANSIENT lasts only while a disturbance
        # is observed without persistent impairment.
        return {
            "state": self.state,
            "persistent_impairment": bool(persistent_impairment),
            "disturbance_observed": disturbance,
            "recovery_count": self.clear_count
        }
