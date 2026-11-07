"""Safe coefficient-space operations for ACE B-basis models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyace import ACEBBasisSet, BBasisConfiguration, PyACECalculator


def load_baseline_coefficients(model_path):
    """Load a fresh ACE B-basis and return an independent coefficient vector."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    basis = ACEBBasisSet()
    basis.load(str(path))
    coefficients = np.asarray(basis.all_coeffs, dtype=np.float64).copy()
    if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
        raise ValueError("Baseline coefficients must be a finite 1D vector.")
    return basis, coefficients


def load_coefficient_permutation(permutation_path, n_coefficients):
    """Load and validate the verified permutation mapping used by the notebook."""
    path = Path(permutation_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Permutation file not found: {path}. Supply the verified mapping; "
            "the package will not infer coefficient ordering."
        )
    with np.load(path) as data:
        permutation = np.asarray(data["permutation"], dtype=int)
        inverse = np.asarray(data["inverse_permutation"], dtype=int)
    expected = np.arange(n_coefficients)
    if permutation.shape != (n_coefficients,) or inverse.shape != (n_coefficients,):
        raise ValueError("Permutation length does not match coefficient vector.")
    if not np.array_equal(np.sort(permutation), expected):
        raise ValueError("permutation is not a bijection.")
    if not np.array_equal(inverse, np.argsort(permutation)):
        raise ValueError("inverse_permutation does not match permutation.")
    return permutation, inverse


def build_candidate_bbasis(
    coefficients,
    model_path,
    inverse_permutation,
    expected_shape=None,
    output_path=None,
):
    """Create a new candidate configuration without modifying the baseline file."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Candidate coefficients must be a finite 1D vector.")
    if expected_shape is not None and values.shape != tuple(expected_shape):
        raise ValueError("Candidate coefficient vector has incorrect shape.")
    inverse = np.asarray(inverse_permutation, dtype=int)
    if inverse.shape != values.shape:
        raise ValueError("inverse_permutation shape does not match coefficients.")

    config = BBasisConfiguration()
    config.load(str(Path(model_path).expanduser()))
    basis = ACEBBasisSet(config)
    basis.all_coeffs = values[inverse].copy()
    candidate_config = basis.to_BBasisConfiguration()
    if output_path is not None:
        candidate_config.save(str(output_path))
    return candidate_config


def build_candidate_calculator(coefficients, model_path, inverse_permutation):
    config = build_candidate_bbasis(coefficients, model_path, inverse_permutation)
    return PyACECalculator(basis_set=ACEBBasisSet(config))


def verify_candidate_roundtrip(coefficients, model_path, inverse_permutation, atol=1e-12, rtol=1e-12):
    config = build_candidate_bbasis(coefficients, model_path, inverse_permutation)
    recovered_internal = np.asarray(ACEBBasisSet(config).all_coeffs, dtype=np.float64)
    recovered_user_order = recovered_internal[np.argsort(inverse_permutation)]
    target = np.asarray(coefficients, dtype=np.float64)
    if not np.allclose(recovered_user_order, target, rtol=rtol, atol=atol):
        raise RuntimeError("Coefficient write-back verification failed.")
    return {
        "verified": True,
        "n_coefficients": int(len(target)),
        "max_abs_difference": float(np.max(np.abs(recovered_user_order - target))),
    }


def validate_patch(patch, baseline_coefficients, max_step=0.05):
    """Validate a list of absolute coefficient updates against the baseline."""
    baseline = np.asarray(baseline_coefficients, dtype=np.float64)
    if not isinstance(patch, list):
        raise TypeError("patch must be a list of dictionaries.")
    seen = set()
    for entry in patch:
        if not isinstance(entry, dict) or not {"index", "value"} <= entry.keys():
            raise ValueError("Each patch entry requires index and value.")
        index, value = entry["index"], entry["value"]
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"Invalid coefficient index: {index!r}")
        if index < 0 or index >= len(baseline):
            raise IndexError(f"Coefficient index out of range: {index}")
        if index in seen:
            raise ValueError(f"Duplicate coefficient index: {index}")
        seen.add(index)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"Invalid coefficient value at index {index}")
        if not np.isfinite(value):
            raise ValueError(f"Non-finite coefficient value at index {index}")
        change = abs(float(value) - baseline[index])
        if change > max_step:
            raise ValueError(
                f"Coefficient {index} exceeds maximum step: {change:.6g} > {max_step:.6g}"
            )
    return True


def apply_patch(candidate_coefficients, patch, baseline_coefficients, max_step=0.05):
    """Return a patched copy; neither baseline nor input candidate is mutated."""
    validate_patch(patch, baseline_coefficients, max_step=max_step)
    updated = np.asarray(candidate_coefficients, dtype=np.float64).copy()
    baseline = np.asarray(baseline_coefficients, dtype=np.float64)
    if updated.shape != baseline.shape:
        raise ValueError("Candidate and baseline shapes differ.")
    for entry in patch:
        updated[int(entry["index"])] = float(entry["value"])
    return updated


def reset_candidate(baseline_coefficients):
    """Return an independent candidate copy of the baseline."""
    return np.asarray(baseline_coefficients, dtype=np.float64).copy()
