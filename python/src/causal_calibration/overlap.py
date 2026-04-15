"""Overlap diagnostics for treatment propensity behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._utils import (
    as_optional_vector,
    as_vector,
    clip_propensity,
    validate_binary,
    validate_nonnegative_weights,
    validate_same_length,
    weighted_effective_sample_size,
)


@dataclass
class OverlapDiagnostics:
    """Summary of overlap quality for binary-treatment data."""

    min_propensity: float
    max_propensity: float
    fraction_below_005: float
    fraction_above_095: float
    fraction_below_010: float
    fraction_above_090: float
    clipped_fraction: float
    ipw_effective_sample_size: float
    overlap_effective_sample_size: float
    severity: str
    recommended_loss: str
    clip: float
    n_obs: int

    def summary(self) -> dict[str, Any]:
        return {
            "min_propensity": self.min_propensity,
            "max_propensity": self.max_propensity,
            "fraction_below_005": self.fraction_below_005,
            "fraction_above_095": self.fraction_above_095,
            "fraction_below_010": self.fraction_below_010,
            "fraction_above_090": self.fraction_above_090,
            "clipped_fraction": self.clipped_fraction,
            "ipw_effective_sample_size": self.ipw_effective_sample_size,
            "overlap_effective_sample_size": self.overlap_effective_sample_size,
            "severity": self.severity,
            "recommended_loss": self.recommended_loss,
            "clip": self.clip,
            "n_obs": self.n_obs,
        }

    def recommendation_messages(self) -> list[str]:
        messages: list[str] = []
        if self.severity == "adequate":
            return messages
        messages.append(
            "The package's default overlap screen flagged weak overlap; consider `loss=\"r\"` for overlap-weighted calibration."
        )
        if self.clipped_fraction > 0.02:
            messages.append(
                "A nontrivial fraction of propensities were clipped; DR-targeted calibration may be unstable."
            )
        if self.ipw_effective_sample_size / max(self.n_obs, 1) < 0.25:
            messages.append(
                "The IPW effective sample size is small relative to n, which suggests unstable original-population weighting."
            )
        return messages

    def plot_data(self) -> dict[str, Any]:
        return {
            "bins": [0.0, 0.05, 0.1, 0.9, 0.95, 1.0],
            "summary": self.summary(),
        }

    def plot(self, ax: Any | None = None) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Plotting overlap diagnostics requires matplotlib.") from exc

        if ax is None:
            _, ax = plt.subplots()
        labels = ["<0.05", "<0.10", ">0.90", ">0.95", "clipped"]
        values = [
            self.fraction_below_005,
            self.fraction_below_010,
            self.fraction_above_090,
            self.fraction_above_095,
            self.clipped_fraction,
        ]
        ax.bar(labels, values, color="#d95f02")
        ax.set_ylim(0.0, max(values + [0.05]) * 1.15)
        ax.set_ylabel("Fraction")
        ax.set_title(f"Overlap Diagnostics ({self.severity})")
        return ax


def assess_overlap(
    *,
    treatment: list[float] | tuple[float, ...],
    propensity: list[float] | tuple[float, ...],
    sample_weight: list[float] | tuple[float, ...] | None = None,
    clip: float = 1e-6,
) -> OverlapDiagnostics:
    treatment_vector = as_vector(treatment, "treatment")
    propensity_raw = as_vector(propensity, "propensity")
    validate_same_length(len(treatment_vector), propensity=propensity_raw)
    validate_binary(treatment_vector, "treatment")
    weights = as_optional_vector(sample_weight, "sample_weight", len(treatment_vector))
    validate_nonnegative_weights(weights)
    propensity_clipped = clip_propensity(propensity_raw, clip)

    ipw_weights = [
        weight * ((a_value / e_value) + ((1.0 - a_value) / (1.0 - e_value)))
        for weight, a_value, e_value in zip(weights, treatment_vector, propensity_clipped)
    ]
    overlap_weights = [
        weight * e_value * (1.0 - e_value)
        for weight, e_value in zip(weights, propensity_clipped)
    ]
    clipped_fraction = sum(
        1.0 for raw_value, clipped_value in zip(propensity_raw, propensity_clipped) if abs(raw_value - clipped_value) > 1e-12
    ) / float(len(propensity_raw))
    min_propensity = min(propensity_raw)
    max_propensity = max(propensity_raw)
    fraction_below_005 = sum(value < 0.05 for value in propensity_raw) / float(len(propensity_raw))
    fraction_above_095 = sum(value > 0.95 for value in propensity_raw) / float(len(propensity_raw))
    fraction_below_010 = sum(value < 0.10 for value in propensity_raw) / float(len(propensity_raw))
    fraction_above_090 = sum(value > 0.90 for value in propensity_raw) / float(len(propensity_raw))
    ipw_ess = weighted_effective_sample_size(ipw_weights)
    overlap_ess = weighted_effective_sample_size(overlap_weights)

    severe = clipped_fraction > 0.02 or ipw_ess / len(propensity_raw) < 0.25
    weak = min_propensity < 0.05 or max_propensity > 0.95 or severe
    severity = "severe" if severe else ("weak" if weak else "adequate")
    recommended_loss = "r" if severity != "adequate" else "dr"

    return OverlapDiagnostics(
        min_propensity=min_propensity,
        max_propensity=max_propensity,
        fraction_below_005=fraction_below_005,
        fraction_above_095=fraction_above_095,
        fraction_below_010=fraction_below_010,
        fraction_above_090=fraction_above_090,
        clipped_fraction=clipped_fraction,
        ipw_effective_sample_size=ipw_ess,
        overlap_effective_sample_size=overlap_ess,
        severity=severity,
        recommended_loss=recommended_loss,
        clip=clip,
        n_obs=len(propensity_raw),
    )
