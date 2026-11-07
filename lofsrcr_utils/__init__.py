"""Utilities for the LofSRCR ACE compression and weight-space test bed."""

from .compression import (
    build_reference_structure,
    load_ace_calculators,
    load_bbasis_models,
    build_target_distances,
    run_compression_scan,
    plot_compression_results,
    save_results,
    run_compression_test,
)
from .diagnostics import (
    diagnose_compression_curve,
    inspect_largest_energy_changes,
    identify_violation_regions,
    build_diagnostic_payload,
)
from .wsp import (
    load_baseline_coefficients,
    load_coefficient_permutation,
    build_candidate_bbasis,
    build_candidate_calculator,
    verify_candidate_roundtrip,
    validate_patch,
    apply_patch,
    reset_candidate,
)

__all__ = [name for name in globals() if not name.startswith("_")]
