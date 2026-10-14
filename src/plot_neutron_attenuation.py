"""Plot one-dimensional neutron attenuation from an OpenMC output directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "reactorpath1"
TRANSPORT_COLUMNS = {
    "scenario",
    "distance_mm",
    "forward_neutron_transmission_fraction",
    "forward_energy_transmission_fraction",
    "mean_forward_energy_mev",
    "mean_forward_energy_std_dev_mev",
}
SPECTRUM_COLUMNS = {
    "scenario",
    "distance_mm",
    "energy_min_mev",
    "energy_max_mev",
    "forward_neutrons_per_source_neutron",
}


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required simulation output is missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: "
                + ", ".join(sorted(missing))
                + ". Rerun the simulation with the updated transport script."
            )
        return list(reader)


def _scenario_layers(
    configuration: dict[str, Any], scenario_name: str
) -> list[dict[str, Any]]:
    scenarios = configuration.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("configuration.json does not contain a scenarios list.")
    for scenario in scenarios:
        if (
            isinstance(scenario, dict)
            and scenario.get("name") == scenario_name
        ):
            layers = scenario.get("layers")
            if isinstance(layers, list):
                return layers
    raise ValueError(
        f"Scenario {scenario_name!r} is not defined in configuration.json."
    )


def _shade_layers(
    axes: list[Any], layers: list[dict[str, Any]], plt: Any
) -> float:
    colors = plt.get_cmap("Pastel1")
    position_mm = 0.0
    for index, layer in enumerate(layers):
        try:
            thickness_mm = float(layer["thickness_mm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each configured layer needs a material and numeric thickness_mm."
            ) from exc
        if (
            not isinstance(layer.get("material"), str)
            or not layer["material"].strip()
        ):
            raise ValueError(
                "Each configured layer needs a material and numeric thickness_mm."
            )
        if not math.isfinite(thickness_mm) or thickness_mm <= 0.0:
            raise ValueError("Layer thicknesses must be positive and finite.")
        end_mm = position_mm + thickness_mm
        color = colors(index % colors.N)
        for axis in axes:
            axis.axvspan(
                position_mm,
                end_mm,
                color=color,
                alpha=0.38,
                linewidth=0,
                zorder=0,
            )
            axis.axvline(end_mm, color="0.35", linewidth=0.6, alpha=0.55)
        position_mm = end_mm
    return position_mm


def _distance_edges(distances_mm: list[float]) -> list[float]:
    if len(distances_mm) < 2:
        raise ValueError("At least two scored distances are needed to plot results.")
    midpoints = [
        (left + right) / 2.0
        for left, right in zip(distances_mm, distances_mm[1:])
    ]
    first_edge = max(0.0, distances_mm[0] - (midpoints[0] - distances_mm[0]))
    last_edge = distances_mm[-1] + (distances_mm[-1] - midpoints[-1])
    return [first_edge, *midpoints, last_edge]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "scenario"


def plot_run(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    scenario_name: Optional[str] = None,
) -> list[Path]:
    input_dir = input_dir.resolve()
    output_dir = input_dir if output_dir is None else output_dir.resolve()
    transport_rows = _read_csv(
        input_dir / "neutron_transport_results.csv", TRANSPORT_COLUMNS
    )
    spectrum_rows = _read_csv(
        input_dir / "neutron_energy_spectrum.csv", SPECTRUM_COLUMNS
    )
    configuration_path = input_dir / "configuration.json"
    if not configuration_path.is_file():
        raise FileNotFoundError(
            f"Required simulation configuration is missing: {configuration_path}"
        )
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))

    available_scenarios = sorted(
        {row["scenario"] for row in transport_rows}
    )
    if scenario_name is not None and scenario_name not in available_scenarios:
        raise ValueError(
            f"Scenario {scenario_name!r} is not present in the results. "
            f"Available scenarios: {', '.join(available_scenarios)}"
        )
    scenarios = (
        [scenario_name] if scenario_name is not None else available_scenarios
    )
    if not scenarios:
        raise ValueError("The transport results contain no scenarios.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and Matplotlib are required; install environments/requirements.txt."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name in scenarios:
        points = sorted(
            (row for row in transport_rows if row["scenario"] == name),
            key=lambda row: float(row["distance_mm"]),
        )
        spectrum = [
            row
            for row in spectrum_rows
            if row["scenario"] == name
            and float(row["energy_min_mev"]) > 0.0
            and float(row["energy_max_mev"]) > 0.0
        ]
        if not points or not spectrum:
            raise ValueError(
                f"Scenario {name!r} needs both transport and spectrum rows."
            )
        distances_mm = [float(row["distance_mm"]) for row in points]
        if any(
            right <= left
            for left, right in zip(distances_mm, distances_mm[1:])
        ):
            raise ValueError(
                f"Scenario {name!r} has duplicate or unordered distance rows."
            )
        layers = _scenario_layers(configuration, name)

        profile_figure, profile_axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12},
        )
        _shade_layers(list(profile_axes), layers, plt)
        neutron_transmission = [
            float(row["forward_neutron_transmission_fraction"])
            for row in points
        ]
        energy_transmission = [
            float(row["forward_energy_transmission_fraction"])
            for row in points
        ]
        profile_axes[0].plot(
            distances_mm,
            neutron_transmission,
            color="tab:blue",
            linewidth=1.8,
            label="Forward neutron-current transmission",
        )
        profile_axes[0].plot(
            distances_mm,
            energy_transmission,
            color="tab:orange",
            linewidth=1.8,
            label="Forward energy-current transmission",
        )
        profile_axes[0].set_ylabel("Fraction of 0 mm entrance current")
        profile_axes[0].set_yscale("log")
        profile_axes[0].legend(loc="best")
        profile_axes[0].grid(True, which="both", alpha=0.22)

        mean_energy = [
            (
                math.nan
                if row["mean_forward_energy_mev"] == ""
                else float(row["mean_forward_energy_mev"])
            )
            for row in points
        ]
        mean_error = [
            (
                0.0
                if row["mean_forward_energy_std_dev_mev"] == ""
                else float(row["mean_forward_energy_std_dev_mev"])
            )
            for row in points
        ]
        profile_axes[1].plot(
            distances_mm,
            mean_energy,
            color="tab:purple",
            linewidth=1.8,
            label="Mean forward neutron energy",
        )
        profile_axes[1].fill_between(
            distances_mm,
            [
                max(0.0, value - error)
                for value, error in zip(mean_energy, mean_error)
            ],
            [
                value + error
                for value, error in zip(mean_energy, mean_error)
            ],
            color="tab:purple",
            alpha=0.2,
            linewidth=0,
            label="Approximate 1 sigma",
        )
        source_energy = float(configuration["source_energy_mev"])
        profile_axes[1].axhline(
            source_energy,
            color="black",
            linestyle=":",
            linewidth=1.1,
            label=f"Source energy: {source_energy:g} MeV",
        )
        profile_axes[1].set_ylabel("Mean forward energy (MeV)")
        profile_axes[1].set_xlabel("Distance through material stack (mm)")
        profile_axes[1].set_xlim(0.0, distances_mm[-1])
        profile_axes[1].grid(True, alpha=0.22)
        profile_axes[1].legend(loc="best")
        profile_figure.suptitle(
            f"1D neutron attenuation: {name}\n"
            "Shaded regions show material layers"
        )
        profile_path = output_dir / f"neutron_attenuation_{_safe_name(name)}.png"
        profile_figure.savefig(profile_path, dpi=180, bbox_inches="tight")
        plt.close(profile_figure)
        saved_paths.append(profile_path)

        distance_edges = _distance_edges(distances_mm)
        energy_edges_mev = sorted(
            {
                float(row["energy_min_mev"])
                for row in spectrum
            }
            | {
                float(row["energy_max_mev"])
                for row in spectrum
            }
        )
        if len(energy_edges_mev) < 2:
            raise ValueError(
                f"Scenario {name!r} has invalid energy-bin boundaries."
            )
        distance_index = {distance: index for index, distance in enumerate(distances_mm)}
        energy_index = {
            (lower, upper): index
            for index, (lower, upper) in enumerate(
                zip(energy_edges_mev, energy_edges_mev[1:])
            )
        }
        flux_grid = np.zeros(
            (len(energy_edges_mev) - 1, len(distances_mm)), dtype=float
        )
        for row in spectrum:
            distance = float(row["distance_mm"])
            energy_bin = (
                float(row["energy_min_mev"]),
                float(row["energy_max_mev"]),
            )
            if distance not in distance_index or energy_bin not in energy_index:
                raise ValueError(
                    f"Scenario {name!r} has inconsistent energy-spectrum rows."
                )
            flux_grid[
                energy_index[energy_bin], distance_index[distance]
            ] = float(row["forward_neutrons_per_source_neutron"])
        positive = flux_grid[flux_grid > 0.0]
        if positive.size == 0:
            raise ValueError(f"Scenario {name!r} has no positive spectrum bins.")
        masked_flux = np.ma.masked_less_equal(flux_grid, 0.0)
        minimum_flux = float(positive.min())
        maximum_flux = float(positive.max())
        if maximum_flux <= minimum_flux:
            maximum_flux = minimum_flux * (1.0 + 1.0e-6)

        spectrum_figure, spectrum_axis = plt.subplots(figsize=(12, 6))
        mesh = spectrum_axis.pcolormesh(
            distance_edges,
            energy_edges_mev,
            masked_flux,
            shading="flat",
            cmap="viridis",
            norm=LogNorm(vmin=minimum_flux, vmax=maximum_flux),
        )
        _shade_layers([spectrum_axis], layers, plt)
        spectrum_axis.set_yscale("log")
        spectrum_axis.set_xlabel("Distance through material stack (mm)")
        spectrum_axis.set_ylabel("Neutron energy (MeV)")
        spectrum_axis.set_title(
            f"Energy-resolved forward attenuation: {name}"
        )
        colorbar = spectrum_figure.colorbar(mesh, ax=spectrum_axis)
        colorbar.set_label("Forward crossings per source neutron per energy bin")
        spectrum_path = output_dir / f"neutron_spectrum_{_safe_name(name)}.png"
        spectrum_figure.savefig(spectrum_path, dpi=180, bbox_inches="tight")
        plt.close(spectrum_figure)
        saved_paths.append(spectrum_path)

    return saved_paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 1D forward-neutron attenuation and energy spectra from "
            "an OpenMC run directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"OpenMC output directory (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to save plots (defaults to --input-dir).",
    )
    parser.add_argument(
        "--scenario",
        help="Plot only this scenario (defaults to all scenarios in the run).",
    )
    args = parser.parse_args(argv)
    try:
        for path in plot_run(args.input_dir, args.output_dir, args.scenario):
            print(f"Saved {path}")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
