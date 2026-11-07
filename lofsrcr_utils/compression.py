"""BCC compression test-bed routines for ASE + PyACE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyace
from ase.build import bulk
from pyace import PyACECalculator


DEFAULT_ELEMENT = "Fe"
DEFAULT_LATTICE_CONSTANT = 2.8665
DEFAULT_SUPERCELL_SIZE = (3, 3, 3)
DEFAULT_MAX_COMPRESSION = 0.80
DEFAULT_N_STEPS = 300


def build_reference_structure(
    element: str = DEFAULT_ELEMENT,
    lattice_constant: float = DEFAULT_LATTICE_CONSTANT,
    supercell_size: tuple[int, int, int] = DEFAULT_SUPERCELL_SIZE,
):
    """Build a periodic BCC reference structure and retain its initial geometry."""
    if lattice_constant <= 0:
        raise ValueError("lattice_constant must be positive.")
    if len(supercell_size) != 3 or any(int(n) < 1 for n in supercell_size):
        raise ValueError("supercell_size must contain three positive integers.")

    atoms = bulk(element, crystalstructure="bcc", a=lattice_constant, cubic=True)
    atoms = atoms.repeat(supercell_size)
    atoms.pbc = True

    r_initial = float(np.sqrt(3.0) * lattice_constant / 2.0)
    return {
        "atoms": atoms,
        "initial_cell": atoms.cell.copy(),
        "initial_positions": atoms.positions.copy(),
        "r_initial": r_initial,
    }


def load_ace_calculators(model_names: Sequence[str], model_paths: Mapping[str, Path]):
    """Load independent ASE calculators from ACE YAML files."""
    calculators = {}
    for name in model_names:
        if name not in model_paths:
            raise KeyError(f"Unknown model {name!r}; available: {list(model_paths)}")
        path = Path(model_paths[name]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name} YAML not found: {path.resolve()}")
        if path.stat().st_size == 0:
            raise ValueError(f"{name} YAML is empty: {path}")
        try:
            calculators[name] = PyACECalculator(str(path))
        except Exception as exc:
            raise RuntimeError(f"Could not initialize {name} from {path}") from exc
    return calculators


def load_bbasis_models(model_names: Sequence[str], model_paths: Mapping[str, Path]):
    """Load PyACE B-basis sets and their source configurations."""
    models, configs = {}, {}
    for name in model_names:
        if name not in model_paths:
            raise KeyError(f"Unknown model {name!r}; available: {list(model_paths)}")
        path = Path(model_paths[name]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        config = pyace.BBasisConfiguration()
        config.load(str(path))
        basis = pyace.ACEBBasisSet()
        basis.load(str(path))
        coeffs = np.asarray(basis.all_coeffs, dtype=np.float64)
        if not np.isfinite(coeffs).all():
            raise ValueError(f"{name} contains non-finite coefficients.")
        models[name], configs[name] = basis, config
    return models, configs


def build_target_distances(
    r_initial: float,
    max_compression: float = DEFAULT_MAX_COMPRESSION,
    n_steps: int = DEFAULT_N_STEPS,
):
    """Generate a linear sequence from the uncompressed to compressed distance."""
    if r_initial <= 0:
        raise ValueError("r_initial must be positive.")
    if not 0.0 <= max_compression < 1.0:
        raise ValueError("max_compression must be in [0, 1).")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    return np.linspace(r_initial, r_initial * (1.0 - max_compression), n_steps + 1)


def run_compression_scan(
    model_names,
    calculators,
    reference_atoms,
    initial_cell,
    initial_positions,
    r_initial,
    target_distances,
):
    """Run an isotropic compression scan; energies are returned per atom."""
    results = {}
    for model_name in model_names:
        if model_name not in calculators:
            raise KeyError(f"No calculator supplied for {model_name!r}.")
        atoms = reference_atoms.copy()
        atoms.calc = calculators[model_name]
        data = {"distance": [], "energy_raw": [], "energy_relative": []}

        for target_distance in target_distances:
            scale = float(target_distance) / float(r_initial)
            atoms.set_cell(initial_cell, scale_atoms=False)
            atoms.set_positions(initial_positions)
            atoms.set_cell(initial_cell * scale, scale_atoms=True)

            energy_raw = np.asarray(atoms.get_potential_energy())

            if energy_raw.size != 1:
                raise ValueError(
                    f"{model_name}: Expected one total energy value, "
                    f"received shape {energy_raw.shape}: {energy_raw}"
                )
            
            energy_total = float(energy_raw.item())


            
            pair_distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(pair_distances, np.inf)
            nearest_neighbor = float(pair_distances.min(axis=1).mean())

            data["distance"].append(nearest_neighbor)
            data["energy_raw"].append(energy_total / len(atoms))

        energy = np.asarray(data["energy_raw"], dtype=float)
        data["energy_relative"] = (energy - energy[0]).tolist()
        results[model_name] = data
    return results


def plot_compression_results(results, model_names):
    """Plot full-range and near-equilibrium relative compression curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in model_names:
        data = results[name]
        distance = np.asarray(data["distance"], dtype=float)
        energy = np.asarray(data["energy_raw"], dtype=float)
        order = np.argsort(distance)
        distance, energy = distance[order], energy[order]
        relative = energy - energy[np.argmax(distance)]

        axes[0].plot(distance, relative, label=name, linewidth=1.5)
        mask = distance >= 0.8 * np.max(distance)
        axes[1].plot(distance[mask], relative[mask], label=name, linewidth=1.5)

    axes[0].set(title="Full compression scan", xlabel="Nearest-neighbour distance (Å)",
                ylabel="Relative energy per atom (eV)", yscale="symlog")
    axes[1].set(title="Near-equilibrium response", xlabel="Nearest-neighbour distance (Å)",
                ylabel="Relative energy per atom (eV)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig, axes


def save_results(results, fig, run_name, output_dir="results/compression_scans", metadata=None):
    """Save the plot, numerical arrays, and metadata for a scan."""
    run_dir = Path(output_dir).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "compression_scan.png", dpi=300, bbox_inches="tight")

    arrays = {}
    for name, data in results.items():
        for key in ("distance", "energy_raw", "energy_relative"):
            arrays[f"{name}_{key}"] = np.asarray(data[key], dtype=float)
    np.savez_compressed(run_dir / "compression_scan.npz", **arrays)

    payload = {
        "run_name": run_name,
        "models": list(results),
        "metadata": metadata or {},
    }
    (run_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def run_compression_test(
    models,
    model_paths,
    run_name="baseline",
    output_dir="results/compression_scans",
    element=DEFAULT_ELEMENT,
    lattice_constant=DEFAULT_LATTICE_CONSTANT,
    supercell_size=DEFAULT_SUPERCELL_SIZE,
    max_compression=DEFAULT_MAX_COMPRESSION,
    n_steps=DEFAULT_N_STEPS,
):
    """Execute the complete reference -> load -> scan -> plot -> save workflow."""
    if not models:
        raise ValueError("models cannot be empty.")
    if len(models) != len(set(models)):
        raise ValueError("models contains duplicate names.")

    reference = build_reference_structure(element, lattice_constant, supercell_size)
    calculators = load_ace_calculators(models, model_paths)
    bbasis_models, bbasis_configs = load_bbasis_models(models, model_paths)
    targets = build_target_distances(reference["r_initial"], max_compression, n_steps)

    results = run_compression_scan(
        models, calculators, reference["atoms"], reference["initial_cell"],
        reference["initial_positions"], reference["r_initial"], targets,
    )
    fig, axes = plot_compression_results(results, models)
    run_dir = save_results(
        results, fig, run_name, output_dir,
        metadata={
            "element": element,
            "lattice_constant_angstrom": lattice_constant,
            "supercell_size": list(supercell_size),
            "max_compression": max_compression,
            "n_steps": n_steps,
        },
    )
    plt.show()
    return {
        "results": results,
        "figure": fig,
        "axes": axes,
        "run_dir": run_dir,
        "bbasis_models": bbasis_models,
        "bbasis_configs": bbasis_configs,
    }
