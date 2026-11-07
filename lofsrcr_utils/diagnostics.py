"""Physical diagnostics for ACE compression-response curves."""

from __future__ import annotations

import numpy as np


def _ordered_curve(results, model_name):
    distance = np.asarray(results[model_name]["distance"], dtype=float)
    energy = np.asarray(results[model_name]["energy_raw"], dtype=float)
    if distance.ndim != 1 or energy.ndim != 1 or len(distance) != len(energy):
        raise ValueError(f"{model_name}: distance and energy must be equal-length 1D arrays.")
    if len(distance) < 2:
        raise ValueError(f"{model_name}: at least two scan points are required.")
    if not np.isfinite(distance).all() or not np.isfinite(energy).all():
        raise ValueError(f"{model_name}: curve contains non-finite values.")
    order = np.argsort(distance)[::-1]  # uncompressed -> compressed
    return distance[order], energy[order]


def diagnose_compression_curve(results, model_name, equilibrium_fraction=0.8):
    """Quantify energy decreases along the compressed branch."""
    distance, energy = _ordered_curve(results, model_name)
    relative = energy - energy[0]
    equilibrium_region = distance >= equilibrium_fraction * distance[0]
    if not equilibrium_region.any():
        raise ValueError("No points found in equilibrium region.")
    eq_indices = np.flatnonzero(equilibrium_region)
    eq_idx = int(eq_indices[np.argmin(relative[equilibrium_region])])
    eq_distance = distance[eq_idx]
    compressed = distance < eq_distance
    compressed_energy = relative[compressed]
    delta = np.diff(compressed_energy)
    decreases = np.maximum(-delta, 0.0)
    return {
        "model": model_name,
        "equilibrium_distance": float(eq_distance),
        "equilibrium_energy": float(relative[eq_idx]),
        "compressed_points": int(compressed.sum()),
        "monotonicity_violations": int(np.sum(delta < 0)),
        "monotonicity_violation_fraction": float(np.mean(delta < 0)) if len(delta) else 0.0,
        "total_decreasing_energy": float(decreases.sum()),
        "largest_energy_decrease": float(decreases.max()) if len(decreases) else 0.0,
        "largest_absolute_energy_jump": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "finite_energy": bool(np.isfinite(energy).all()),
    }


def inspect_largest_energy_changes(results, model_name, n=10):
    """Return the largest adjacent energy changes along compression."""
    distance, energy = _ordered_curve(results, model_name)
    delta = np.diff(energy)
    indices = np.argsort(np.abs(delta))[::-1][:n]
    return [
        {
            "index": int(i),
            "distance_before": float(distance[i]),
            "distance_after": float(distance[i + 1]),
            "energy_change": float(delta[i]),
        }
        for i in indices
    ]


def identify_violation_regions(results, model_name):
    """Find contiguous intervals where energy decreases during compression."""
    distance, energy = _ordered_curve(results, model_name)
    delta = np.diff(energy)
    violation = delta < 0
    padded = np.r_[False, violation, False]
    transitions = np.diff(padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    regions = []
    for start, end in zip(starts, ends):
        changes = delta[start:end]
        regions.append({
            "start_index": int(start),
            "end_index": int(end),
            "distance_start": float(distance[start]),
            "distance_end": float(distance[end]),
            "number_of_intervals": int(end - start),
            "total_energy_decrease": float(-changes.sum()),
            "largest_single_decrease": float(-changes.min()),
        })
    return sorted(regions, key=lambda item: item["total_energy_decrease"], reverse=True)


def build_diagnostic_payload(model_name, regions, diagnostics, equilibrium_cutoff=None):
    """Build a structured, model-grounded prompt payload for LLM diagnosis."""
    report = diagnostics[model_name]
    selected_regions = list(regions[model_name])
    if equilibrium_cutoff is not None:
        selected_regions = [
            region for region in selected_regions
            if region["distance_start"] < equilibrium_cutoff
        ]
    return {
        "task": "ACE compression-response diagnosis",
        "model": model_name,
        "physical_expectation": {
            "equilibrium": "A local energy minimum near equilibrium.",
            "post_equilibrium": "Energy should increase as interatomic distance decreases.",
            "potential_artifacts": [
                "secondary minima",
                "energy decreases during compression",
                "discontinuities",
                "runaway energy collapse",
            ],
        },
        "baseline": {
            "equilibrium_distance_angstrom": report["equilibrium_distance"],
            "equilibrium_energy_ev_per_atom": report["equilibrium_energy"],
            "monotonicity_violation_fraction": report["monotonicity_violation_fraction"],
            "total_decreasing_energy_ev_per_atom": report["total_decreasing_energy"],
            "largest_energy_decrease_ev_per_atom": report["largest_energy_decrease"],
            "largest_energy_jump_ev_per_atom": report["largest_absolute_energy_jump"],
        },
        "artifact_regions": selected_regions,
        "editing_constraints": {
            "preserve_baseline": True,
            "modify_candidate_only": True,
            "preserve_near_equilibrium_response": True,
            "avoid_unnecessary_coefficient_changes": True,
        },
    }
