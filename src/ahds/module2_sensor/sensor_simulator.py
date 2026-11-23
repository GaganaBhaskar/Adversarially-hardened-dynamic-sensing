
from dataclasses import dataclass
import math
import torch


@dataclass
class SensorConfig:
    exposure_ms: float = 10.0
    gain: float = 1.0
    electrons_per_unit_ms: float = 100.0
    full_well_electrons: float = 10000.0
    read_noise_electrons: float = 3.0
    adc_bits: int = 12

    def validate(self):
        if not math.isfinite(self.exposure_ms) or self.exposure_ms <= 0:
            raise ValueError("Exposure must be positive and finite.")

        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("Gain must be positive and finite.")

        if (
            not math.isfinite(self.electrons_per_unit_ms)
            or self.electrons_per_unit_ms <= 0
        ):
            raise ValueError("Conversion factor must be positive.")

        if (
            not math.isfinite(self.full_well_electrons)
            or self.full_well_electrons <= 0
        ):
            raise ValueError("Full-well capacity must be positive.")

        if (
            not math.isfinite(self.read_noise_electrons)
            or self.read_noise_electrons < 0
        ):
            raise ValueError("Read noise cannot be negative.")

        if not isinstance(self.adc_bits, int) or not 1 <= self.adc_bits <= 16:
            raise ValueError("ADC bits must be between 1 and 16.")


def simulate_sensor(
    optical_scene,
    config=None,
    add_noise=True,
    seed=None
):
    """
    Simulate a simplified linear RGB sensor.

    Input:
        optical_scene: (3, H, W), nonnegative
        normalized optical radiance-like values.

    Output:
        dictionary of sensor measurements and diagnostics.

    Assumptions:
        - No Bayer color filter array
        - No spectral response model
        - No rolling shutter
        - No optics/PSF model
        - Uniform exposure across the frame
        - Gain applied after charge saturation
    """

    if config is None:
        config = SensorConfig()

    config.validate()

    if optical_scene.ndim != 3 or optical_scene.shape[0] != 3:
        raise ValueError("Expected RGB tensor of shape (3, H, W).")

    if not torch.is_floating_point(optical_scene):
        raise ValueError("Input must be a floating-point tensor.")

    if not torch.isfinite(optical_scene).all():
        raise ValueError("Optical scene must contain finite values.")

    if (optical_scene < 0).any():
        raise ValueError("Optical scene must be nonnegative.")

    # Expected photoelectrons
    expected_electrons = (
        optical_scene
        * config.exposure_ms
        * config.electrons_per_unit_ms
    )

    if add_noise:
        # Use a local generator for reproducibility.
        generator = torch.Generator(device=optical_scene.device)

        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Photon shot noise
        electrons = torch.poisson(
            expected_electrons,
            generator=generator
        )

        # Electronic read noise
        read_noise = torch.randn(
            electrons.shape,
            device=electrons.device,
            dtype=electrons.dtype,
            generator=generator
        ) * config.read_noise_electrons

        electrons = electrons + read_noise

    else:
        electrons = expected_electrons.clone()

    # Full-well saturation occurs before gain.
    saturated = electrons >= config.full_well_electrons

    clipped_electrons = electrons.clamp(
        min=0.0,
        max=config.full_well_electrons
    )

    # Simplified analog gain.
    amplified = clipped_electrons * config.gain

    # ADC range
    adc_max = (2 ** config.adc_bits) - 1

    # Fixed conversion scale based on full-well capacity.
    normalized = (
        amplified / config.full_well_electrons
    ).clamp(0.0, 1.0)

    adc_output = torch.round(
        normalized * adc_max
    ).to(torch.int32)

    adc_clipped = amplified >= config.full_well_electrons

    return {
        "expected_electrons": expected_electrons,
        "electrons": electrons,
        "saturated": saturated,
        "adc_clipped": adc_clipped,
        "adc_output": adc_output,
        "normalized_output": adc_output.float() / adc_max,
        "saturation_ratio": saturated.float().mean().item(),
        "adc_clip_ratio": adc_clipped.float().mean().item()
    }
