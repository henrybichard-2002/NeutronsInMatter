"""Run OpenMC fixed-source neutron transport through planar material layers.

Prepared JEFF-4.0 OpenMC HDF5 data are used directly when configured. Processed
ACE data can also be converted to OpenMC HDF5 format and cached on first use.
Raw ENDF evaluations are not used as transport cross sections.
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "settings" / "openmc_neutron_shielding.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "openmc_neutron_shielding"
DATA_TEMPERATURE_K = 293.6
MAX_NEUTRON_ENERGY_MEV = 20.0
SOURCE_Z_CM = -0.5
SOURCE_THICKNESS_CM = 1.0e-10
VACUUM_INLET_Z_CM = -1.0
DOWNSTREAM_VACUUM_CM = 1.0
ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ACE_FILENAME_RE = re.compile(r"^(\d+)-([A-Za-z]+)-(\d+)g-293\.6$")
HDF5_FILENAME_RE = re.compile(
    r"^(\d+)-([A-Za-z]+)-(\d+)g-293\.6\.h5$"
)


NATURAL_ISOTOPES: dict[str, dict[int, float]] = {
    "H": {1001: 0.999885, 1002: 0.000115},
    "C": {6012: 0.9893, 6013: 0.0107},
    "N": {7014: 0.99636, 7015: 0.00364},
    "O": {8016: 0.99757, 8017: 0.00038, 8018: 0.00205},
    "Ar": {18036: 0.003336, 18038: 0.000629, 18040: 0.996035},
    "Al": {13027: 1.0},
    "Pb": {82204: 0.014, 82206: 0.241, 82207: 0.221, 82208: 0.524},
}

BUILTIN_MATERIALS: dict[str, dict[str, Any]] = {
    "air": {
        "density_g_cm3": 0.00120479,
        "element_atom_fractions": {"N": 1.56168, "O": 0.41976, "Ar": 0.00934, "C": 0.00042},
    },
    "aluminium": {
        "density_g_cm3": 2.70,
        "element_atom_fractions": {"Al": 1.0},
    },
    "hdpe": {
        "density_g_cm3": 0.95,
        "element_atom_fractions": {"C": 1.0, "H": 2.0},
    },
    "lead": {
        "density_g_cm3": 11.34,
        "element_atom_fractions": {"Pb": 1.0},
    },
    "water": {
        "density_g_cm3": 0.9982,
        "element_atom_fractions": {"H": 2.0, "O": 1.0},
    },
}

MATERIAL_ALIASES = {
    "al": "aluminium",
    "aluminum": "aluminium",
    "airdry": "air",
    "dryair": "air",
    "h2o": "water",
    "polyethylene": "hdpe",
    "pb": "lead",
}


@dataclass(frozen=True)
class Layer:
    material: str
    thickness_cm: float
    density_g_cm3: float


@dataclass(frozen=True)
class Scenario:
    name: str
    layers: tuple[Layer, ...]


@dataclass(frozen=True)
class PointResult:
    scenario: str
    distance_cm: float
    forward_neutrons_per_source_neutron: float
    relative_error: float
    mean_energy_mev: Optional[float]


def _positive_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a positive finite number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{description} must be a positive finite number.")
    return number


def _safe_name(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not compact:
        raise ValueError(f"Name {value!r} cannot be used as a directory name.")
    return compact[:80]


def _canonical_material_name(name: str) -> str:
    compact = re.sub(r"[\s_()-]+", "", name.strip().casefold())
    return MATERIAL_ALIASES.get(compact, compact)


def _find_custom_material(
    custom_materials: dict[str, Any], material_name: str
) -> Optional[str]:
    return next(
        (
            key
            for key in custom_materials
            if _canonical_material_name(key) == material_name
        ),
        None,
    )


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a JSON object.")
    return config


def _validate_config(config: dict[str, Any]) -> None:
    library_name = config.get("nuclear_data_library")
    if not isinstance(library_name, str) or "jeff" not in library_name.casefold():
        raise ValueError(
            "Set 'nuclear_data_library' to the JEFF evaluation/library used to "
            "prepare the ACE data."
        )
    source_energy = _positive_number(config.get("source_energy_mev"), "Source energy")
    if source_energy > MAX_NEUTRON_ENERGY_MEV:
        raise ValueError(
            f"JEFF processed data are expected to cover energies through "
            f"{MAX_NEUTRON_ENERGY_MEV:g} MeV."
        )
    source_rate = config.get("source_rate_neutrons_per_s")
    if source_rate is not None:
        _positive_number(source_rate, "'source_rate_neutrons_per_s'")
    histories = config.get("histories")
    if isinstance(histories, bool) or not isinstance(histories, int) or histories < 2:
        raise ValueError("'histories' must be an integer of at least 2.")
    batches = config.get("batches", 100)
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 2:
        raise ValueError("'batches' must be an integer of at least 2.")
    if batches > histories or histories % batches:
        raise ValueError("'histories' must be divisible by 'batches'.")
    interval = _positive_number(config.get("sample_interval_cm"), "'sample_interval_cm'")
    beam_width = config.get("beam_width_cm", 0.0)
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, (int, float))
        or not math.isfinite(float(beam_width))
        or beam_width < 0.0
    ):
        raise ValueError("'beam_width_cm' must be a non-negative finite number.")
    lateral_half_width = _positive_number(
        config.get("lateral_half_width_cm", 50.0), "'lateral_half_width_cm'"
    )
    if beam_width > 2.0 * lateral_half_width:
        raise ValueError("'beam_width_cm' must fit inside the reflective lateral boundaries.")
    groups = config.get("energy_groups")
    if isinstance(groups, bool) or not isinstance(groups, int) or not 24 <= groups <= 997:
        raise ValueError("'energy_groups' must be an integer between 24 and 997.")
    if interval <= 0:
        raise ValueError("'sample_interval_cm' must be positive.")
    if not isinstance(config.get("thermal_scattering_tables", {}), dict):
        raise ValueError("'thermal_scattering_tables' must be a JSON object.")
    if not isinstance(config.get("custom_materials", {}), dict):
        raise ValueError("'custom_materials' must be a JSON object.")


def _make_scenarios(config: dict[str, Any]) -> tuple[Scenario, ...]:
    custom_materials = config.get("custom_materials", {})
    raw_scenarios = config.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("'scenarios' must be a non-empty JSON array.")

    scenarios: list[Scenario] = []
    names: set[str] = set()
    directories: set[str] = set()
    for raw_scenario in raw_scenarios:
        if not isinstance(raw_scenario, dict):
            raise ValueError("Each scenario must be a JSON object.")
        name = raw_scenario.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every scenario needs a non-empty 'name'.")
        safe_name = _safe_name(name)
        if name.strip().casefold() in names or safe_name.casefold() in directories:
            raise ValueError(f"Scenario names must be unique: {name!r}.")
        names.add(name.strip().casefold())
        directories.add(safe_name.casefold())
        raw_layers = raw_scenario.get("layers")
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ValueError(f"Scenario {name!r} must contain at least one layer.")

        layers: list[Layer] = []
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict):
                raise ValueError(f"Layers in scenario {name!r} must be JSON objects.")
            material = raw_layer.get("material")
            if not isinstance(material, str) or not material.strip():
                raise ValueError(f"Every layer in {name!r} needs a material name.")
            material_name = _canonical_material_name(material)
            if material_name in BUILTIN_MATERIALS:
                density_default = BUILTIN_MATERIALS[material_name]["density_g_cm3"]
            else:
                custom_name = _find_custom_material(custom_materials, material_name)
                if custom_name is None:
                    raise ValueError(
                        f"Unknown material {material!r}; define it in "
                        "'custom_materials'."
                    )
                custom = custom_materials[custom_name]
                if not isinstance(custom, dict):
                    raise ValueError(f"Custom material {custom_name!r} must be an object.")
                density_default = custom.get("density_g_cm3")
            layers.append(
                Layer(
                    material_name,
                    _positive_number(raw_layer.get("thickness_cm"), "Layer thickness"),
                    _positive_number(
                        raw_layer.get("density_g_cm3", density_default),
                        f"Density for {material!r}",
                    ),
                )
            )
        scenarios.append(Scenario(name.strip(), tuple(layers)))
    return tuple(scenarios)


def _material_nuclides(
    material_name: str, custom_materials: dict[str, Any]
) -> dict[int, float]:
    if material_name in BUILTIN_MATERIALS:
        amounts: dict[int, float] = {}
        for element, atom_fraction in BUILTIN_MATERIALS[material_name][
            "element_atom_fractions"
        ].items():
            for zaid, isotope_fraction in NATURAL_ISOTOPES[element].items():
                amounts[zaid] = amounts.get(zaid, 0.0) + atom_fraction * isotope_fraction
    else:
        custom_name = _find_custom_material(custom_materials, material_name)
        if custom_name is None:
            raise ValueError(f"Custom material {material_name!r} is not defined.")
        raw = custom_materials[custom_name].get("nuclide_atom_fractions")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                f"Custom material {custom_name!r} needs non-empty "
                "'nuclide_atom_fractions' keyed by ZAID."
            )
        amounts = {}
        for zaid_text, fraction in raw.items():
            try:
                zaid = int(zaid_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid nuclide ZAID {zaid_text!r}.") from exc
            if zaid < 1001 or zaid > 118999:
                raise ValueError(f"Invalid nuclide ZAID {zaid}.")
            amounts[zaid] = _positive_number(fraction, f"Atom fraction for ZAID {zaid}")
    total = sum(amounts.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"Material {material_name!r} has invalid atom fractions.")
    return {zaid: fraction / total for zaid, fraction in amounts.items()}


def _nuclide_name(zaid: int) -> str:
    atomic_number, mass_number = divmod(zaid, 1000)
    if not 1 <= atomic_number <= len(ELEMENT_SYMBOLS) or mass_number == 0:
        raise ValueError(f"Cannot convert ZAID {zaid} to an OpenMC nuclide name.")
    return f"{ELEMENT_SYMBOLS[atomic_number - 1]}{mass_number}"


def _index_ace_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed JEFF neutron ACE directory not found: {directory}")
    ace_files: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = ACE_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        atomic_number = int(match.group(1))
        symbol = match.group(2)
        mass_number = int(match.group(3))
        if (
            not 1 <= atomic_number <= len(ELEMENT_SYMBOLS)
            or ELEMENT_SYMBOLS[atomic_number - 1] != symbol
        ):
            continue
        zaid = atomic_number * 1000 + mass_number
        if zaid in ace_files:
            raise ValueError(f"More than one 293.6 K ACE file was found for ZAID {zaid}.")
        ace_files[zaid] = path
    return ace_files


def _index_hdf5_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Prepared JEFF OpenMC HDF5 directory not found: {directory}"
        )
    hdf5_files: dict[int, Path] = {}
    for path in directory.glob("*-293.6.h5"):
        if not path.is_file():
            continue
        match = HDF5_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        atomic_number = int(match.group(1))
        symbol = match.group(2)
        mass_number = int(match.group(3))
        if (
            not 1 <= atomic_number <= len(ELEMENT_SYMBOLS)
            or ELEMENT_SYMBOLS[atomic_number - 1] != symbol
        ):
            continue
        zaid = atomic_number * 1000 + mass_number
        if zaid in hdf5_files:
            raise ValueError(
                f"More than one 293.6 K HDF5 file was found for ZAID {zaid}."
            )
        hdf5_files[zaid] = path
    return hdf5_files


def _resolve_config_path(value: Any, description: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {description} in the OpenMC settings JSON.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=True)


def _load_openmc() -> Any:
    try:
        return import_module("openmc")
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is not installed in this Python environment. On Windows, "
            "run this project in WSL2 or Docker; OpenMC is not available as a "
            "native win-64 conda package. In WSL, create the environment with "
            "'conda env create -f environment-openmc.yml' and activate it."
        ) from exc


def _convert_ace_data(
    openmc: Any,
    config: dict[str, Any],
    config_path: Path,
    scenarios: tuple[Scenario, ...],
    data_output_dir: Path,
) -> tuple[Path, dict[str, str]]:
    neutron_hdf5_dir: Optional[Path] = None
    if config.get("neutron_hdf5_directory") is not None:
        neutron_hdf5_dir = _resolve_config_path(
            config.get("neutron_hdf5_directory"),
            "'neutron_hdf5_directory'",
            config_path,
        )
        neutron_files = _index_hdf5_files(neutron_hdf5_dir)
    else:
        neutron_dir = _resolve_config_path(
            config.get("neutron_ace_directory"),
            "'neutron_ace_directory'",
            config_path,
        )
        neutron_files = _index_ace_files(neutron_dir)
    custom_materials = config.get("custom_materials", {})
    required_zaids = {
        zaid
        for scenario in scenarios
        for layer in scenario.layers
        for zaid in _material_nuclides(layer.material, custom_materials)
    }
    missing = sorted(required_zaids - neutron_files.keys())
    if missing:
        raise FileNotFoundError(
            "JEFF neutron data files are missing for: "
            + ", ".join(_nuclide_name(zaid) for zaid in missing)
        )

    thermal_config = config.get("thermal_scattering_tables", {})
    used_materials = {
        layer.material for scenario in scenarios for layer in scenario.layers
    }
    thermal_sources: dict[str, tuple[Path, str]] = {}
    for material in sorted(used_materials):
        if material not in {"water", "hdpe"}:
            continue
        entry = thermal_config.get(material)
        if not isinstance(entry, dict):
            raise ValueError(
                f"Material {material!r} requires JEFF-derived bound-hydrogen "
                "thermal scattering. Configure its ACE file and OpenMC name "
                "under 'thermal_scattering_tables'."
            )
        thermal_dir = _resolve_config_path(
            config.get("thermal_ace_directory"),
            "'thermal_ace_directory'",
            config_path,
        )
        ace_file = entry.get("ace_file")
        openmc_name = entry.get("openmc_name")
        if not isinstance(ace_file, str) or not ace_file.strip():
            raise ValueError(
                f"Set 'thermal_scattering_tables.{material}.ace_file'."
            )
        if not isinstance(openmc_name, str) or not openmc_name.strip():
            raise ValueError(
                f"Set 'thermal_scattering_tables.{material}.openmc_name'."
            )
        source = (thermal_dir / ace_file).resolve(strict=True)
        thermal_sources[material] = (source, openmc_name)

    data_output_dir.mkdir(parents=True, exist_ok=True)
    library = openmc.data.DataLibrary()
    converted: dict[str, str] = {}
    for zaid in sorted(required_zaids):
        source = neutron_files[zaid]
        if neutron_hdf5_dir is not None:
            hdf5_path = source
        else:
            hdf5_path = data_output_dir / f"{_nuclide_name(zaid)}.h5"
        if neutron_hdf5_dir is None and (
            not hdf5_path.is_file()
            or source.stat().st_mtime_ns > hdf5_path.stat().st_mtime_ns
        ):
            try:
                neutron_data = openmc.data.IncidentNeutron.from_ace(
                    str(source), metastable_scheme="mcnp"
                )
                neutron_data.export_to_hdf5(str(hdf5_path), mode="w")
            except Exception as exc:
                raise RuntimeError(
                    f"OpenMC could not convert JEFF ACE neutron table {source}: {exc}"
                ) from exc
        library.register_file(str(hdf5_path))
        converted[_nuclide_name(zaid)] = str(hdf5_path)

    for material, (source, openmc_name) in thermal_sources.items():
        hdf5_path = data_output_dir / f"{_safe_name(openmc_name)}.h5"
        if not hdf5_path.is_file() or source.stat().st_mtime_ns > hdf5_path.stat().st_mtime_ns:
            try:
                thermal_data = openmc.data.ThermalScattering.from_ace(
                    str(source), name=openmc_name
                )
                thermal_data.export_to_hdf5(str(hdf5_path), mode="w")
            except Exception as exc:
                raise RuntimeError(
                    f"OpenMC could not convert JEFF thermal ACE table {source}: {exc}"
                ) from exc
        library.register_file(str(hdf5_path))
        converted[f"thermal:{material}"] = str(hdf5_path)

    cross_sections_path = data_output_dir / "cross_sections.xml"
    library.export_to_xml(str(cross_sections_path))
    return cross_sections_path, converted


def _monitor_distances(scenario: Scenario, interval_cm: float) -> list[float]:
    total = sum(layer.thickness_cm for layer in scenario.layers)
    distances = {0.0, round(total, 10)}
    layer_boundary = 0.0
    for layer in scenario.layers:
        layer_boundary += layer.thickness_cm
        distances.add(round(layer_boundary, 10))
    index = 1
    while index * interval_cm < total:
        distances.add(round(index * interval_cm, 10))
        index += 1
    return sorted(distances)


def _energy_edges_mev(source_energy_mev: float, groups: int) -> list[float]:
    lower_mev = 1.0e-11
    upper_mev = MAX_NEUTRON_ENERGY_MEV
    edges = [
        lower_mev * (upper_mev / lower_mev) ** (index / groups)
        for index in range(groups + 1)
    ]
    if lower_mev < source_energy_mev < upper_mev:
        edges.append(source_energy_mev)
    edges = sorted(set(edges))
    return [0.0] + edges


def _build_scenario_model(
    openmc: Any,
    scenario: Scenario,
    config: dict[str, Any],
) -> tuple[Any, dict[int, tuple[float, int, str, str]]]:
    materials: list[Any] = []
    layer_materials = []
    custom_materials = config.get("custom_materials", {})
    thermal_config = config.get("thermal_scattering_tables", {})
    for layer_index, layer in enumerate(scenario.layers, 1):
        material = openmc.Material(name=f"{scenario.name} {layer.material} {layer_index}")
        for zaid, fraction in _material_nuclides(
            layer.material, custom_materials
        ).items():
            material.add_nuclide(_nuclide_name(zaid), fraction, "ao")
        material.set_density("g/cm3", layer.density_g_cm3)
        material.temperature = DATA_TEMPERATURE_K
        thermal = thermal_config.get(layer.material)
        if isinstance(thermal, dict):
            material.add_s_alpha_beta(thermal["openmc_name"])
        materials.append(material)
        layer_materials.append(material)

    total_thickness = sum(layer.thickness_cm for layer in scenario.layers)
    distances = _monitor_distances(
        scenario, float(config["sample_interval_cm"])
    )
    z_nodes = [VACUUM_INLET_Z_CM, 0.0] + distances
    z_nodes.append(total_thickness + DOWNSTREAM_VACUUM_CM)
    z_nodes = sorted(set(round(value, 10) for value in z_nodes))
    z_surfaces = [
        openmc.ZPlane(
            z0=value,
            boundary_type=(
                "vacuum"
                if index in {0, len(z_nodes) - 1}
                else "transmission"
            ),
        )
        for index, value in enumerate(z_nodes)
    ]

    lateral_half_width = float(config.get("lateral_half_width_cm", 50.0))
    x_min = openmc.XPlane(x0=-lateral_half_width, boundary_type="reflective")
    x_max = openmc.XPlane(x0=lateral_half_width, boundary_type="reflective")
    y_min = openmc.YPlane(y0=-lateral_half_width, boundary_type="reflective")
    y_max = openmc.YPlane(y0=lateral_half_width, boundary_type="reflective")
    lateral_region = +x_min & -x_max & +y_min & -y_max

    cells = []
    upstream_cell_for_distance: dict[float, Any] = {}
    layer_edges = []
    current_edge = 0.0
    for layer, material in zip(scenario.layers, layer_materials):
        current_edge += layer.thickness_cm
        layer_edges.append((current_edge, material))

    for index, (lower, upper) in enumerate(zip(z_nodes, z_nodes[1:])):
        midpoint = (lower + upper) / 2.0
        fill = None
        if 0.0 <= midpoint < total_thickness:
            for edge, material in layer_edges:
                if midpoint < edge:
                    fill = material
                    break
        cell = openmc.Cell(
            name=f"slab segment {index}",
            fill=fill,
            region=lateral_region & +z_surfaces[index] & -z_surfaces[index + 1],
        )
        cells.append(cell)
        if round(upper, 10) in distances:
            upstream_cell_for_distance[round(upper, 10)] = cell

    geometry = openmc.Geometry(openmc.Universe(cells=cells))
    source_width = float(config.get("beam_width_cm", 0.0))
    if source_width > 0.0:
        source_space = openmc.stats.Box(
            (
                -source_width / 2.0,
                -source_width / 2.0,
                SOURCE_Z_CM - SOURCE_THICKNESS_CM,
            ),
            (
                source_width / 2.0,
                source_width / 2.0,
                SOURCE_Z_CM + SOURCE_THICKNESS_CM,
            ),
        )
    else:
        source_space = openmc.stats.Point((0.0, 0.0, SOURCE_Z_CM))
    source = openmc.IndependentSource(
        space=source_space,
        angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)),
        energy=openmc.stats.Discrete(
            [float(config["source_energy_mev"]) * 1.0e6], [1.0]
        ),
    )
    batches = int(config.get("batches", 100))
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = batches
    settings.particles = int(config["histories"]) // batches
    settings.source = source
    settings.statepoint = {"batches": [batches]}

    energy_edges_ev = [value * 1.0e6 for value in _energy_edges_mev(
        float(config["source_energy_mev"]), int(config["energy_groups"])
    )]
    energy_filter = openmc.EnergyFilter(energy_edges_ev)
    tallies = openmc.Tallies()
    tally_names: dict[int, tuple[float, int, str, str]] = {}
    for tally_index, distance in enumerate(distances, 1):
        surface = next(
            plane for coordinate, plane in zip(z_nodes, z_surfaces)
            if coordinate == distance
        )
        upstream_cell = upstream_cell_for_distance[distance]
        surface_filter = openmc.SurfaceFilter([surface])
        cell_from_filter = openmc.CellFromFilter([upstream_cell])
        total_tally_name = f"forward-current-total-{tally_index}"
        total_tally = openmc.Tally(name=total_tally_name)
        total_tally.filters = [surface_filter, cell_from_filter]
        total_tally.scores = ["current"]
        tallies.append(total_tally)

        spectrum_tally_name = f"forward-current-spectrum-{tally_index}"
        spectrum_tally = openmc.Tally(name=spectrum_tally_name)
        spectrum_tally.filters = [
            surface_filter,
            cell_from_filter,
            energy_filter,
        ]
        spectrum_tally.scores = ["current"]
        tallies.append(spectrum_tally)
        tally_names[tally_index] = (
            distance,
            upstream_cell.id,
            total_tally_name,
            spectrum_tally_name,
        )

    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials(materials),
        settings=settings,
        tallies=tallies,
    )
    return model, tally_names


def _energy_bin_centers(edges_mev: list[float]) -> list[float]:
    centers: list[float] = []
    for low, high in zip(edges_mev, edges_mev[1:]):
        centers.append((low + high) / 2.0 if low == 0.0 else math.sqrt(low * high))
    return centers


def _read_scenario_results(
    openmc: Any,
    statepoint_path: Path,
    scenario: Scenario,
    tally_names: dict[int, tuple[float, int, str, str]],
    config: dict[str, Any],
) -> list[PointResult]:
    if not statepoint_path.is_file():
        raise FileNotFoundError(
            f"OpenMC did not produce {statepoint_path}; inspect the scenario "
            "output files for the engine error."
        )
    energy_edges_mev = _energy_edges_mev(
        float(config["source_energy_mev"]), int(config["energy_groups"])
    )
    centers = _energy_bin_centers(energy_edges_mev)
    results: list[PointResult] = []
    with openmc.StatePoint(str(statepoint_path)) as statepoint:
        for _, (distance, _, total_name, spectrum_name) in tally_names.items():
            total_tally = statepoint.get_tally(name=total_name)
            spectrum_tally = statepoint.get_tally(name=spectrum_name)
            total = float(total_tally.mean.ravel()[0])
            standard_deviation = float(total_tally.std_dev.ravel()[0])
            if total < -1.0e-14:
                raise RuntimeError(
                    f"OpenMC returned negative forward current {total} at "
                    f"{distance:g} cm. Verify the CellFromFilter direction."
                )
            total = max(total, 0.0)
            relative_error = (
                standard_deviation / total
                if total > 0.0
                else 0.0
            )
            spectrum = [float(value) for value in spectrum_tally.mean.ravel()]
            if len(spectrum) != len(centers):
                raise RuntimeError(
                    f"Energy tally at {distance:g} cm has {len(spectrum)} bins; "
                    f"expected {len(centers)}."
                )
            if any(value < -1.0e-14 for value in spectrum):
                raise RuntimeError(
                    "OpenMC returned a negative energy-binned forward current; "
                    "verify tally filters and energy bins."
                )
            spectrum = [max(value, 0.0) for value in spectrum]
            spectrum_sum = sum(spectrum)
            mean_energy = (
                sum(center * value for center, value in zip(centers, spectrum))
                / spectrum_sum
                if spectrum_sum > 0.0
                else None
            )
            results.append(
                PointResult(
                    scenario.name,
                    distance,
                    total,
                    relative_error,
                    mean_energy,
                )
            )
    return results


def _save_results(
    output_dir: Path,
    results: list[PointResult],
    config: dict[str, Any],
    source_directories: dict[str, str],
    converted_data: dict[str, str],
) -> None:
    source_rate = config.get("source_rate_neutrons_per_s")
    csv_path = output_dir / "neutron_transport_results.csv"
    columns = [
        "scenario",
        "distance_cm",
        "forward_neutrons_per_source_neutron",
        "relative_error",
        "mean_forward_energy_mev",
        "forward_neutrons_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "distance_cm": f"{result.distance_cm:.10g}",
                    "forward_neutrons_per_source_neutron": (
                        f"{result.forward_neutrons_per_source_neutron:.10g}"
                    ),
                    "relative_error": f"{result.relative_error:.8g}",
                    "mean_forward_energy_mev": (
                        ""
                        if result.mean_energy_mev is None
                        else f"{result.mean_energy_mev:.10g}"
                    ),
                    "forward_neutrons_per_second": (
                        ""
                        if source_rate is None
                        else f"{result.forward_neutrons_per_source_neutron * source_rate:.10g}"
                    ),
                }
            )

    metadata = {
        "transport_code": "OpenMC continuous-energy fixed-source neutron transport",
        "nuclear_data_library": config.get("nuclear_data_library", "JEFF-4.0"),
        "source_data_directories": source_directories,
        "openmc_hdf5_data": converted_data,
        "source_energy_mev": config["source_energy_mev"],
        "histories_per_scenario": config["histories"],
        "batches": config.get("batches", 100),
        "source_rate_neutrons_per_s": source_rate,
        "temperature_k": DATA_TEMPERATURE_K,
        "result_definition": (
            "OpenMC surface current tallies filtered by the upstream CellFromFilter "
            "and neutron energy. These are forward crossings per source neutron; "
            "repeated forward crossings by a particle are included. Mean energies "
            "are estimated from energy-bin geometric representative values."
        ),
        "geometry_note": (
            "Planar layers with reflective lateral boundaries approximate an "
            "infinite slab and intentionally exclude lateral leakage."
        ),
        "validation_note": (
            "JEFF provenance is recorded from the configured nuclear data "
            "directories; confirm processing records and validation benchmarks "
            "independently."
        ),
        "results_csv": csv_path.name,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for result plots; install it with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    figure, (count_axis, energy_axis) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True
    )
    for scenario_name in dict.fromkeys(result.scenario for result in results):
        points = [r for r in results if r.scenario == scenario_name]
        distances = [r.distance_cm for r in points]
        counts = [
            r.forward_neutrons_per_source_neutron
            if source_rate is None
            else r.forward_neutrons_per_source_neutron * source_rate
            for r in points
        ]
        errors = [r.relative_error * value for r, value in zip(points, counts)]
        count_axis.errorbar(
            distances, counts, yerr=errors, marker="o", markersize=3,
            linewidth=1.6, capsize=2, label=scenario_name,
        )
        energy_points = [
            (point.distance_cm, point.mean_energy_mev)
            for point in points
            if point.mean_energy_mev is not None
        ]
        if energy_points:
            energy_axis.plot(
                [item[0] for item in energy_points],
                [item[1] for item in energy_points],
                marker="o", markersize=3, linewidth=1.6, label=scenario_name,
            )

    count_axis.set_ylabel(
        "Forward neutron crossings (n/s)"
        if source_rate is not None
        else "Forward neutron crossings per source neutron"
    )
    count_axis.set_title(
        f"{config['source_energy_mev']:g} MeV beam through material slabs "
        f"({config['histories']:,} histories)"
    )
    count_axis.grid(True, which="both", alpha=0.35)
    count_axis.legend()
    if source_rate is not None:
        count_axis.set_yscale("log")
    energy_axis.set_xlabel("Depth through shield (cm)")
    energy_axis.set_ylabel("Mean forward neutron energy (MeV)")
    energy_axis.grid(True, which="both", alpha=0.35)
    energy_axis.set_yscale("log")
    energy_axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "neutron_transport_vs_depth.png", dpi=180)
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenMC continuous-energy neutron transport through planar "
            "material layers using the supplied JEFF-4.0 data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON settings file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--neutron-hdf5-dir",
        type=Path,
        help="Override the prepared 293.6 K JEFF OpenMC neutron HDF5 directory.",
    )
    parser.add_argument(
        "--neutron-ace-dir",
        type=Path,
        help="Override the processed 293.6 K JEFF neutron ACE directory (fallback).",
    )
    parser.add_argument(
        "--thermal-ace-dir",
        type=Path,
        help="Override the processed JEFF thermal ACE directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Run output directory (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="OpenMC CPU thread count; defaults to the settings file or OpenMC default.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare nuclear data and write OpenMC XML without running transport.",
    )
    args = parser.parse_args(argv)

    try:
        config_path = args.config.resolve(strict=True)
        config = _load_config(config_path)
        if args.neutron_ace_dir is not None:
            config["neutron_ace_directory"] = str(args.neutron_ace_dir.resolve())
            config.pop("neutron_hdf5_directory", None)
        if args.neutron_hdf5_dir is not None:
            config["neutron_hdf5_directory"] = str(args.neutron_hdf5_dir.resolve())
        if args.thermal_ace_dir is not None:
            config["thermal_ace_directory"] = str(args.thermal_ace_dir.resolve())
        _validate_config(config)
        scenarios = _make_scenarios(config)
        openmc = _load_openmc()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "openmc_hdf5_data"
        cross_sections, converted = _convert_ace_data(
            openmc, config, config_path, scenarios, data_dir
        )
        source_dirs = {
            "processed_thermal_ace": str(
                _resolve_config_path(
                    config["thermal_ace_directory"],
                    "'thermal_ace_directory'",
                    config_path,
                )
            ),
        }
        if config.get("neutron_hdf5_directory") is not None:
            source_dirs["prepared_neutron_hdf5"] = str(
                _resolve_config_path(
                    config["neutron_hdf5_directory"],
                    "'neutron_hdf5_directory'",
                    config_path,
                )
            )
        else:
            source_dirs["processed_neutron_ace"] = str(
                _resolve_config_path(
                    config["neutron_ace_directory"],
                    "'neutron_ace_directory'",
                    config_path,
                )
            )
        (output_dir / "configuration.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        all_results: list[PointResult] = []
        for scenario in scenarios:
            case_dir = output_dir / _safe_name(scenario.name)
            case_dir.mkdir(parents=True, exist_ok=True)
            model, tally_names = _build_scenario_model(
                openmc, scenario, config
            )
            model.export_to_xml(directory=str(case_dir))
            print(f"Wrote OpenMC model for {scenario.name}: {case_dir}")
            if args.prepare_only:
                continue
            os.environ["OPENMC_CROSS_SECTIONS"] = str(cross_sections)
            openmc.config["cross_sections"] = str(cross_sections)
            thread_count = (
                args.threads if args.threads is not None else config.get("threads")
            )
            if thread_count is not None and (
                isinstance(thread_count, bool)
                or not isinstance(thread_count, int)
                or thread_count < 1
            ):
                raise ValueError("'threads' must be a positive integer.")
            openmc.run(
                cwd=str(case_dir),
                output=True,
                **({} if thread_count is None else {"threads": thread_count}),
            )
            batches = int(config.get("batches", 100))
            statepoint_path = case_dir / f"statepoint.{batches}.h5"
            all_results.extend(
                _read_scenario_results(
                    openmc, statepoint_path, scenario, tally_names, config
                )
            )
            print(f"Completed {scenario.name}: {config['histories']:,} histories")

        if args.prepare_only:
            print(
                "OpenMC model XML and converted HDF5 nuclear data are ready. "
                "No transport histories were run."
            )
            return 0
        _save_results(output_dir, all_results, config, source_dirs, converted)
        print(f"Saved OpenMC results and plots in {output_dir}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
