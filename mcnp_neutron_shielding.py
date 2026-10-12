"""Generate MCNP models for a normally incident 14.1 MeV neutron beam.

This program uses JEFF-derived continuous-energy ACE neutron data through the
MCNP XSDIR. Raw JEFF ENDF evaluations must be processed separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


JEFF_MAX_ENERGY_MEV = 20.0
LOWEST_ENERGY_MEV = 1.0e-11
SOURCE_POSITION_CM = -0.05
UPSTREAM_VACUUM_PLANE_CM = -0.1
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "settings" / "mcnp_neutron_shielding.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mcnp_neutron_shielding"


# Natural-isotope atom abundances are expanded explicitly so MCNP uses the
# isotope-specific JEFF ACE tables instead of non-JEFF elemental defaults.
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
        "element_atom_fractions": {
            "N": 1.56168,
            "O": 0.41976,
            "Ar": 0.00934,
            "C": 0.00042,
        },
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

_TABLE_RE = re.compile(r"^\s*(\d{1,6}\.\d+[a-z])\s+", re.IGNORECASE)
_THERMAL_TABLE_RE = re.compile(
    r"^\s*([a-z][a-z0-9_-]*\.\d+t)\s+", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")
_IMPLICIT_EXPONENT_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$"
)
_MCTAL_CARDS = {
    f"{card}{flag}"
    for card in "fdusmcet"
    for flag in ("", "t", "c")
}
_MCTAL_COUNT_RE = re.compile(r"^(\d+)([a-z]*)$", re.IGNORECASE)


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
class Monitor:
    distance_cm: float
    surface_id: int
    tally_id: int


@dataclass(frozen=True)
class TallyResult:
    value: float
    relative_error: float
    energy_bin_values: tuple[float, ...]


@dataclass(frozen=True)
class PointResult:
    scenario: str
    distance_cm: float
    neutrons_per_source_neutron: float
    relative_error: float
    mean_energy_mev: Optional[float]


def _positive_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a positive finite number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{description} must be a positive finite number.")
    return number


def _canonical_material_name(name: str) -> str:
    compact_name = re.sub(r"[\s_()-]+", "", name.strip().casefold())
    return MATERIAL_ALIASES.get(compact_name, compact_name)


def _find_custom_material(
    custom_materials: dict[str, Any], material_name: str
) -> str | None:
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
    source_rate = config.get("source_rate_neutrons_per_s")
    if source_rate is not None:
        _positive_number(source_rate, "'source_rate_neutrons_per_s'")
    if not isinstance(config.get("thermal_scattering_tables", {}), dict):
        raise ValueError("'thermal_scattering_tables' must be a JSON object.")


def _make_scenarios(config: dict[str, Any]) -> tuple[Scenario, ...]:
    custom_materials = config.get("custom_materials", {})
    if not isinstance(custom_materials, dict):
        raise ValueError("'custom_materials' must be a JSON object.")

    raw_scenarios = config.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("'scenarios' must be a non-empty JSON array.")

    scenarios: list[Scenario] = []
    seen_names: set[str] = set()
    seen_directories: set[str] = set()
    for raw_scenario in raw_scenarios:
        if not isinstance(raw_scenario, dict):
            raise ValueError("Each scenario must be a JSON object.")
        name = raw_scenario.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every scenario needs a non-empty 'name'.")
        if name.strip().casefold() in seen_names:
            raise ValueError(f"Scenario names must be unique: {name!r}.")
        seen_names.add(name.strip().casefold())
        directory_name = _safe_name(name).casefold()
        if directory_name in seen_directories:
            raise ValueError(
                f"Scenario names produce the same output folder: {name!r}."
            )
        seen_directories.add(directory_name)

        raw_layers = raw_scenario.get("layers")
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ValueError(f"Scenario {name!r} must contain at least one layer.")

        layers: list[Layer] = []
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict):
                raise ValueError(f"Layers in scenario {name!r} must be JSON objects.")
            material_name = raw_layer.get("material")
            if not isinstance(material_name, str) or not material_name.strip():
                raise ValueError(
                    f"Every layer in scenario {name!r} needs a material name."
                )
            canonical = _canonical_material_name(material_name)
            if canonical in BUILTIN_MATERIALS:
                default_density = BUILTIN_MATERIALS[canonical]["density_g_cm3"]
            else:
                matching_key = _find_custom_material(custom_materials, canonical)
                if matching_key is None:
                    known = ", ".join(sorted(BUILTIN_MATERIALS))
                    raise ValueError(
                        f"Unknown material {material_name!r}. Built-in materials "
                        f"are {known}; define other materials in 'custom_materials'."
                    )
                material_data = custom_materials[matching_key]
                if not isinstance(material_data, dict):
                    raise ValueError(
                        f"Custom material {matching_key!r} must be a JSON object."
                    )
                default_density = _positive_number(
                    material_data.get("density_g_cm3"),
                    f"Density for custom material {matching_key!r}",
                )

            density = _positive_number(
                raw_layer.get("density_g_cm3", default_density),
                f"Density for {material_name!r}",
            )
            thickness = _positive_number(
                raw_layer.get("thickness_cm"),
                f"Thickness for {material_name!r}",
            )
            layers.append(Layer(canonical, thickness, density))
        scenarios.append(Scenario(name.strip(), tuple(layers)))

    return tuple(scenarios)


def _material_nuclides(
    material_name: str, custom_materials: dict[str, Any]
) -> dict[int, float]:
    if material_name in BUILTIN_MATERIALS:
        element_fractions = BUILTIN_MATERIALS[material_name]["element_atom_fractions"]
        nuclide_amounts: dict[int, float] = {}
        for element, atom_fraction in element_fractions.items():
            for zaid, isotope_fraction in NATURAL_ISOTOPES[element].items():
                nuclide_amounts[zaid] = (
                    nuclide_amounts.get(zaid, 0.0)
                    + atom_fraction * isotope_fraction
                )
    else:
        matching_key = _find_custom_material(custom_materials, material_name)
        if matching_key is None:
            raise ValueError(f"Custom material {material_name!r} is not defined.")
        material_data = custom_materials[matching_key]
        raw_nuclides = material_data.get("nuclide_atom_fractions")
        if not isinstance(raw_nuclides, dict) or not raw_nuclides:
            raise ValueError(
                f"Custom material {matching_key!r} needs a non-empty "
                "'nuclide_atom_fractions' object with isotope ZAIDs as keys."
            )
        nuclide_amounts = {}
        for zaid_text, fraction in raw_nuclides.items():
            try:
                zaid = int(zaid_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid isotope ZAID {zaid_text!r} in {matching_key!r}."
                ) from exc
            if not 1000 <= zaid <= 999999:
                raise ValueError(
                    f"Isotope ZAID {zaid!r} must be an MCNP nuclide ZAID."
                )
            nuclide_amounts[zaid] = _positive_number(
                fraction,
                f"Atom fraction for isotope {zaid} in {matching_key!r}",
            )

    total = sum(nuclide_amounts.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"Material {material_name!r} has an invalid composition.")
    return {zaid: amount / total for zaid, amount in nuclide_amounts.items()}


def _make_energy_bounds(source_energy_mev: float, group_count: int) -> list[float]:
    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise ValueError("'energy_groups' must be an integer.")
    if not 24 <= group_count <= 997:
        raise ValueError("'energy_groups' must be between 24 and 997.")

    low_exponent = math.log10(LOWEST_ENERGY_MEV)
    high_exponent = math.log10(JEFF_MAX_ENERGY_MEV)
    bounds = [
        10.0 ** (low_exponent + i * (high_exponent - low_exponent) / (group_count - 1))
        for i in range(group_count)
    ]
    for near_source_energy in (
        source_energy_mev * (1.0 - 1.0e-6),
        source_energy_mev,
        source_energy_mev * (1.0 + 1.0e-6),
    ):
        if near_source_energy <= JEFF_MAX_ENERGY_MEV:
            bounds.append(near_source_energy)
    bounds.sort()

    unique_bounds: list[float] = []
    for bound in bounds:
        if not unique_bounds or bound > unique_bounds[-1] * (1.0 + 1.0e-12):
            unique_bounds.append(bound)
    return unique_bounds


def _format_number(value: float) -> str:
    return f"{value:.8e}"


def _format_card(keyword: str, tokens: list[str], max_width: int = 78) -> list[str]:
    lines: list[str] = []
    current = keyword
    for token in tokens:
        candidate = f"{current} {token}"
        if len(candidate) > max_width and current != keyword:
            lines.append(current)
            current = f"     {token}"
        elif len(candidate) > max_width:
            raise ValueError(f"MCNP card {keyword} contains an oversized token.")
        else:
            current = candidate
    lines.append(current)
    return lines


def _make_monitors(
    layers: tuple[Layer, ...], sample_interval_cm: float
) -> tuple[list[tuple[float, float, int]], list[Monitor]]:
    sample_positions = {0.0}
    layer_bounds: list[tuple[float, float, int]] = []
    cursor = 0.0

    for layer_index, layer in enumerate(layers):
        end = cursor + layer.thickness_cm
        layer_bounds.append((cursor, end, layer_index))
        next_sample = (math.floor(cursor / sample_interval_cm) + 1) * sample_interval_cm
        while next_sample < end - max(1.0, end) * 1.0e-12:
            sample_positions.add(next_sample)
            next_sample += sample_interval_cm
        sample_positions.add(end)
        cursor = end

    positions = sorted(sample_positions)
    if len(positions) > 999:
        raise ValueError(
            "A scenario creates more than 999 monitor planes. Increase "
            "'sample_interval_cm' to keep the MCNP tally cards manageable."
        )
    monitors = [
        Monitor(distance, index + 2, 10 * (index + 1) + 1)
        for index, distance in enumerate(positions)
    ]
    return layer_bounds, monitors


def _material_ids(scenario: Scenario) -> dict[str, int]:
    return {
        material: index
        for index, material in enumerate(
            dict.fromkeys(layer.material for layer in scenario.layers), start=1
        )
    }


def _thermal_table_for(
    material_name: str, thermal_tables: dict[str, Any]
) -> Optional[str]:
    if material_name not in {"water", "hdpe"}:
        return None
    configured = thermal_tables.get(material_name)
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError(
            f"Set 'thermal_scattering_tables.{material_name}' to a JEFF-derived "
            "MCNP thermal-scattering table. Water and polyethylene require "
            "bound-hydrogen thermal-scattering data."
        )
    return configured.strip()


def _build_deck(
    scenario: Scenario,
    config: dict[str, Any],
    library_suffix: str,
) -> tuple[str, list[Monitor], list[float]]:
    source_energy = _positive_number(
        config.get("source_energy_mev"), "'source_energy_mev'"
    )
    if source_energy > JEFF_MAX_ENERGY_MEV:
        raise ValueError(
            f"The {source_energy:g} MeV source is above the JEFF evaluation "
            f"energy ceiling of {JEFF_MAX_ENERGY_MEV:g} MeV used by this model."
        )
    sample_interval = _positive_number(
        config.get("sample_interval_cm"), "'sample_interval_cm'"
    )
    histories = config.get("histories")
    if isinstance(histories, bool) or not isinstance(histories, int) or histories < 1:
        raise ValueError("'histories' must be a positive integer.")
    energy_bounds = _make_energy_bounds(source_energy, config.get("energy_groups"))
    if len(energy_bounds) > 1000:
        raise ValueError(
            "The configured source energy adds too many boundaries; reduce "
            "'energy_groups'."
        )

    layer_bounds, monitors = _make_monitors(scenario.layers, sample_interval)
    surface_for_distance = {
        monitor.distance_cm: monitor.surface_id for monitor in monitors
    }
    material_ids = _material_ids(scenario)
    custom_materials = config.get("custom_materials", {})
    thermal_tables = config.get("thermal_scattering_tables", {})
    if not isinstance(thermal_tables, dict):
        raise ValueError("'thermal_scattering_tables' must be a JSON object.")

    cells = [f"1 0 1 -2 IMP:N=1"]
    material_for_layer: dict[int, int] = {}

    cell_id = 2
    for layer_start, layer_end, layer_index in layer_bounds:
        layer = scenario.layers[layer_index]
        previous = layer_start
        segment_ends = [
            monitor.distance_cm
            for monitor in monitors
            if layer_start < monitor.distance_cm <= layer_end
        ]
        for segment_end in segment_ends:
            left_surface = surface_for_distance[previous]
            right_surface = surface_for_distance[segment_end]
            mat_id = material_ids[layer.material]
            cells.append(
                f"{cell_id} {mat_id} -{layer.density_g_cm3:.8g} "
                f"{left_surface} -{right_surface} IMP:N=1"
            )
            material_for_layer[layer_index] = mat_id
            cell_id += 1
            previous = segment_end

    surfaces = [f"1 px {_format_number(UPSTREAM_VACUUM_PLANE_CM)}", "2 px 0.0"]
    for monitor in monitors[1:]:
        surfaces.append(
            f"{monitor.surface_id} px {_format_number(monitor.distance_cm)}"
        )

    tally_cards: list[str] = []
    energy_tokens = [_format_number(bound) for bound in energy_bounds] + ["T"]
    for monitor in monitors:
        tally_cards.extend(
            _format_card(
                f"F{monitor.tally_id}:N",
                [f"+{monitor.surface_id}"],
            )
        )
        tally_cards.extend(_format_card(f"E{monitor.tally_id}", energy_tokens))

    data_cards = [
        "MODE N",
        (
            "SDEF CEL=1 PAR=N POS="
            f"{_format_number(SOURCE_POSITION_CM)} 0 0 "
            f"VEC=1 0 0 DIR=1 ERG={_format_number(source_energy)}"
        ),
    ]
    for material_name, material_id in material_ids.items():
        nuclides = _material_nuclides(material_name, custom_materials)
        material_tokens: list[str] = []
        for zaid, atom_fraction in sorted(nuclides.items()):
            material_tokens.extend(
                (f"{zaid}.{library_suffix}", _format_number(atom_fraction))
            )
        data_cards.extend(_format_card(f"M{material_id}", material_tokens))

        thermal_table = _thermal_table_for(material_name, thermal_tables)
        if thermal_table is not None:
            data_cards.append(f"MT{material_id} {thermal_table}")

    data_cards.extend(tally_cards)
    data_cards.append(f"NPS {histories}")

    title = f"{scenario.name} - JEFF continuous-energy neutron transport"
    if len(title) > 80:
        title = title[:80]
    deck_sections = [
        title,
        *cells,
        "",
        *surfaces,
        "",
        *data_cards,
    ]
    return "\n".join(deck_sections) + "\n", monitors, energy_bounds


def _read_xsdir(path: Path, library_suffix: str) -> set[str]:
    if path.name.casefold() != "xsdir":
        raise ValueError(
            f"MCNP expects the cross-section index to be named 'xsdir'; "
            f"received {path.name!r}."
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OSError(f"Cannot read MCNP XSDIR file {path}: {exc}") from exc
    table_names: set[str] = set()
    for line in text.splitlines():
        match = _TABLE_RE.match(line) or _THERMAL_TABLE_RE.match(line)
        if match:
            table_names.add(match.group(1).casefold())
    if not table_names:
        raise ValueError(f"No ACE tables were found in {path}.")

    expected_suffix = library_suffix.casefold()
    wrong_suffix_tables = [
        name for name in table_names if name.endswith("c") and name.endswith(expected_suffix)
    ]
    if not wrong_suffix_tables:
        raise ValueError(
            f"XSDIR {path} contains no continuous-energy neutron tables ending "
            f"in {library_suffix!r}."
        )
    return table_names


def _validate_xsdir(
    xsdir_path: Path,
    library_suffix: str,
    scenarios: tuple[Scenario, ...],
    config: dict[str, Any],
) -> set[str]:
    if not re.fullmatch(r"\d{1,3}c", library_suffix, re.IGNORECASE):
        raise ValueError(
            "'neutron_data_suffix' must be the MCNP continuous-energy suffix "
            "without a dot, for example '80c'. Use the actual JEFF ACE suffix "
            "listed in your XSDIR."
        )
    tables = _read_xsdir(xsdir_path, library_suffix)
    custom_materials = config.get("custom_materials", {})
    material_names = {
        layer.material for scenario in scenarios for layer in scenario.layers
    }
    required_zaids = {
        zaid
        for material in material_names
        for zaid in _material_nuclides(material, custom_materials)
    }
    missing = sorted(
        f"{zaid}.{library_suffix}"
        for zaid in required_zaids
        if f"{zaid}.{library_suffix}".casefold() not in tables
    )
    if missing:
        displayed = ", ".join(missing[:20])
        remainder = len(missing) - min(20, len(missing))
        if remainder:
            displayed += f", and {remainder} more"
        raise ValueError(
            "The XSDIR is missing isotope-specific MCNP tables needed by the "
            f"selected materials: {displayed}. Process matching JEFF evaluations "
            "into ACE tables before running."
        )

    thermal_tables = config.get("thermal_scattering_tables", {})
    for material in material_names:
        thermal_table = _thermal_table_for(material, thermal_tables)
        if thermal_table is not None and thermal_table.casefold() not in tables:
            raise ValueError(
                f"Thermal-scattering table {thermal_table!r} for {material!r} "
                f"is not listed in XSDIR {xsdir_path}."
            )
    return tables


def _parse_mcnp_number(token: str) -> float:
    normalized = token.replace("D", "E").replace("d", "e")
    if _NUMBER_RE.fullmatch(normalized):
        return float(normalized)
    implicit_exponent = _IMPLICIT_EXPONENT_RE.fullmatch(normalized)
    if implicit_exponent:
        return float(f"{implicit_exponent.group(1)}e{implicit_exponent.group(2)}")
    raise ValueError(f"Invalid MCNP numeric value {token!r}.")


def _is_mctal_card(line: str) -> bool:
    return bool(
        line
        and not line[0].isspace()
        and line.split(maxsplit=1)[0].casefold() in _MCTAL_CARDS | {"vals"}
    )


def _parse_card_count(token: str) -> tuple[int, str]:
    match = _MCTAL_COUNT_RE.fullmatch(token.casefold())
    if not match:
        raise ValueError(f"Unsupported MCTAL bin count {token!r}.")
    return int(match.group(1)), match.group(2)


def _parse_mctal(
    path: Path,
    expected_tally_ids: set[int],
    energy_group_count: int,
) -> tuple[int, dict[int, TallyResult]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise OSError(f"Cannot read MCNP MCTAL output {path}: {exc}") from exc
    if len(lines) < 4:
        raise ValueError(f"MCTAL output {path} is incomplete.")
    header_tokens = lines[0].split()
    try:
        histories = int(header_tokens[-2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot determine the completed histories in {path}.") from exc

    tallies: dict[int, TallyResult] = {}
    for start, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 3 or tokens[0].casefold() != "tally":
            continue
        try:
            tally_id = int(tokens[1])
        except ValueError as exc:
            raise ValueError(f"Invalid tally header in {path}: {line!r}.") from exc
        if tally_id not in expected_tally_ids:
            continue

        cards: dict[str, tuple[int, str, list[float]]] = {}
        index = start + 1
        while index < len(lines):
            current = lines[index]
            if current.casefold().startswith("tally ") or current.casefold().startswith(
                "kcode "
            ):
                break
            if not _is_mctal_card(current):
                index += 1
                continue

            card_tokens = current.split()
            card_name = card_tokens[0].casefold()
            if card_name == "vals":
                break
            if len(card_tokens) < 2:
                raise ValueError(
                    f"Malformed {card_name!r} card in MCTAL tally {tally_id}."
                )
            count, count_flags = _parse_card_count(card_tokens[1])
            flags = card_name[1:] + count_flags
            values: list[float] = []
            index += 1
            while index < len(lines) and not _is_mctal_card(lines[index]):
                values.extend(_parse_mcnp_number(value) for value in lines[index].split())
                index += 1
            cards[card_name] = (count, flags, values)
            if card_name == "vals":
                break

        if "et" not in cards:
            raise ValueError(
                f"MCTAL tally {tally_id} has no total energy-binned tally; "
                "verify the MCNP E card ends in T."
            )
        energy_count, flags, energy_labels = cards["et"]
        if "t" not in flags:
            raise ValueError(
                f"MCTAL tally {tally_id} is missing the requested total energy bin."
            )
        group_count = energy_count - 1
        if group_count != energy_group_count or len(energy_labels) != group_count:
            raise ValueError(
                f"MCTAL tally {tally_id} has {group_count} energy groups and "
                f"{len(energy_labels)} boundaries; expected "
                f"{energy_group_count} groups from the input deck."
            )

        while index < len(lines) and not _is_mctal_card(lines[index]):
            if lines[index].casefold().startswith("tally ") or lines[
                index
            ].casefold().startswith("kcode "):
                break
            index += 1
        if index >= len(lines) or lines[index].split()[0].casefold() != "vals":
            raise ValueError(f"MCTAL tally {tally_id} has no 'vals' data.")
        index += 1

        score_tokens: list[float] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if (
                not lines[index][:1].isspace()
                and stripped.casefold().startswith(("tfc", "tally ", "kcode "))
            ):
                break
            score_tokens.extend(
                _parse_mcnp_number(token) for token in lines[index].split()
            )
            index += 1
        if len(score_tokens) != 2 * energy_count:
            raise ValueError(
                f"MCTAL tally {tally_id} contains {len(score_tokens) // 2} "
                f"value/error pairs; expected {energy_count} energy groups "
                "including the total."
            )

        pairs = list(zip(score_tokens[::2], score_tokens[1::2]))
        if any(not math.isfinite(value) for pair in pairs for value in pair):
            raise ValueError(f"MCTAL tally {tally_id} contains non-finite results.")
        total_value, total_error = pairs[-1]
        tallies[tally_id] = TallyResult(
            value=total_value,
            relative_error=total_error,
            energy_bin_values=tuple(value for value, _ in pairs[:-1]),
        )

    missing_tallies = expected_tally_ids - tallies.keys()
    if missing_tallies:
        raise ValueError(
            f"MCTAL output is missing generated tally IDs: "
            f"{', '.join(str(tally) for tally in sorted(missing_tallies))}."
        )
    return histories, tallies


def _mean_energy(
    energy_bounds_mev: list[float], bin_values: tuple[float, ...]
) -> Optional[float]:
    if len(bin_values) != len(energy_bounds_mev):
        raise ValueError("Energy tally values do not match the generated energy bins.")
    total_score = sum(bin_values)
    if total_score <= 0.0:
        return None

    energy_weighted_score = 0.0
    lower_bound = 0.0
    for upper_bound, score in zip(energy_bounds_mev, bin_values):
        representative_energy = (
            upper_bound * 0.5
            if lower_bound == 0.0
            else math.sqrt(lower_bound * upper_bound)
        )
        energy_weighted_score += representative_energy * score
        lower_bound = upper_bound
    return energy_weighted_score / total_score


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    if not safe:
        raise ValueError(f"Scenario name {name!r} cannot be used as a directory name.")
    return safe[:80]


def _write_scenario_deck(
    output_dir: Path,
    scenario: Scenario,
    config: dict[str, Any],
    library_suffix: str,
) -> tuple[Path, list[Monitor], list[float]]:
    deck, monitors, energy_bounds = _build_deck(scenario, config, library_suffix)
    case_dir = output_dir / _safe_name(scenario.name)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "model.i"
    with input_path.open("w", encoding="ascii", newline="\n") as input_file:
        input_file.write(deck)
    return input_path, monitors, energy_bounds


def _resolve_mcnp_executable(configured: Any) -> str:
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("'mcnp_executable' must be a non-empty path or command.")
        executable = configured.strip()
        if Path(executable).is_file():
            return str(Path(executable).resolve())
        located = shutil.which(executable)
        if located:
            return located
        raise FileNotFoundError(f"MCNP executable {executable!r} was not found.")

    for name in ("mcnp6", "mcnp6.exe", "mcnp"):
        located = shutil.which(name)
        if located:
            return located
    raise FileNotFoundError(
        "MCNP was not found on PATH. Install MCNP, or use --prepare-only to "
        "generate decks without starting a simulation."
    )


def _run_scenario(
    executable: str,
    xsdir_path: Path,
    input_path: Path,
    monitors: list[Monitor],
    energy_bounds: list[float],
    histories: int,
) -> dict[int, TallyResult]:
    case_dir = input_path.parent
    mctal_path = case_dir / "mctal"
    if mctal_path.exists():
        raise FileExistsError(
            f"{mctal_path} already exists. Choose a new output directory so an "
            "old tally file cannot be mistaken for this run."
        )
    output_path = case_dir / "mcnp.out"
    environment = os.environ.copy()
    environment["DATAPATH"] = str(xsdir_path.parent)
    command = [
        executable,
        f"i={input_path.name}",
        f"o={output_path.name}",
    ]
    completed = subprocess.run(
        command,
        cwd=case_dir,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"MCNP failed for {input_path.parent.name} "
            f"(exit code {completed.returncode}). See {output_path}."
            + (f"\n{detail[-2000:]}" if detail else "")
        )
    if not mctal_path.is_file():
        raise FileNotFoundError(
            f"MCNP completed without producing {mctal_path}. Check the MCNP "
            f"output in {output_path}."
        )

    completed_histories, tallies = _parse_mctal(
        mctal_path,
        {monitor.tally_id for monitor in monitors},
        len(energy_bounds),
    )
    if completed_histories != histories:
        raise RuntimeError(
            f"MCNP completed {completed_histories} histories, but the input "
            f"requested {histories}; inspect {output_path}."
        )
    return tallies


def _save_results(
    output_dir: Path,
    results: list[PointResult],
    config: dict[str, Any],
    xsdir_path: Path,
    library_suffix: str,
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
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "distance_cm": f"{result.distance_cm:.10g}",
                    "forward_neutrons_per_source_neutron": (
                        f"{result.neutrons_per_source_neutron:.10g}"
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
                        else f"{result.neutrons_per_source_neutron * source_rate:.10g}"
                    ),
                }
            )

    report = {
        "transport_code": "MCNP continuous-energy neutron transport",
        "nuclear_data_library": config.get("nuclear_data_library", "JEFF-4.0"),
        "continuous_energy_table_suffix": library_suffix,
        "xsdir": str(xsdir_path),
        "source_energy_mev": config["source_energy_mev"],
        "histories_per_scenario": config["histories"],
        "source_rate_neutrons_per_s": source_rate,
        "thermal_scattering_tables": config.get("thermal_scattering_tables", {}),
        "result_definition": (
            "F1 surface current in the positive beam direction, per source "
            "neutron; tally includes repeated forward crossings. Mean energy "
            "is estimated from the MCNP energy-binned current using logarithmic "
            "bin representative energies."
        ),
        "validation_note": (
            "XSDIR table names are checked, but the source evaluation for each "
            "ACE table must be verified by the user from the JEFF/NJOY processing "
            "records. MCNP results are not independently benchmarked by this script."
        ),
        "results_csv": csv_path.name,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to plot MCNP results; install it with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    figure, (count_axis, energy_axis) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True
    )
    for scenario_name in dict.fromkeys(result.scenario for result in results):
        case_results = [r for r in results if r.scenario == scenario_name]
        distances = [r.distance_cm for r in case_results]
        counts = [
            r.neutrons_per_source_neutron
            if source_rate is None
            else r.neutrons_per_source_neutron * source_rate
            for r in case_results
        ]
        errors = [r.relative_error * count for r, count in zip(case_results, counts)]
        count_axis.errorbar(
            distances,
            counts,
            yerr=errors,
            marker="o",
            markersize=3,
            linewidth=1.6,
            capsize=2,
            label=scenario_name,
        )
        energy_points = [
            (r.distance_cm, r.mean_energy_mev)
            for r in case_results
            if r.mean_energy_mev is not None
        ]
        if energy_points:
            energy_axis.plot(
                [point[0] for point in energy_points],
                [point[1] for point in energy_points],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=scenario_name,
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
            "Generate MCNP continuous-energy slab-transport input decks from "
            "JEFF-derived ACE data, run them, and plot forward current and "
            "neutron energy versus depth."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "JSON file defining source, layers, histories, and tally settings "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--xsdir",
        type=Path,
        help=(
            "Path to the MCNP XSDIR file for your processed JEFF ACE library; "
            "defaults to 'xsdir' in the settings file."
        ),
    )
    parser.add_argument(
        "--library-suffix",
        help="MCNP continuous-energy suffix from the JEFF ACE XSDIR (e.g. 80c).",
    )
    parser.add_argument(
        "--mcnp-executable",
        help="MCNP executable path or command; defaults to mcnp6 on PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Directory for MCNP inputs, outputs, CSV data, and plots "
            f"(default: {DEFAULT_OUTPUT_PATH})."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate the configured XSDIR and write input decks without running MCNP.",
    )
    args = parser.parse_args(argv)

    try:
        config_path = args.config.resolve()
        config = _load_config(config_path)
        _validate_config(config)
        scenarios = _make_scenarios(config)
        xsdir_value = args.xsdir or config.get("xsdir")
        if not isinstance(xsdir_value, (str, Path)) or not str(xsdir_value).strip():
            raise ValueError(
                "No MCNP XSDIR file is configured. Set 'xsdir' in "
                f"{config_path} to the path of your processed JEFF ACE XSDIR, "
                "or pass --xsdir <path>."
            )
        xsdir_path = Path(xsdir_value).expanduser()
        if not xsdir_path.is_absolute() and args.xsdir is None:
            xsdir_path = config_path.parent / xsdir_path
        suffix = args.library_suffix or config.get("neutron_data_suffix")
        if not isinstance(suffix, str) or not suffix.strip():
            raise ValueError(
                "Specify the actual JEFF-derived ACE suffix in the configuration "
                "or with --library-suffix (for example, 80c)."
            )
        suffix = suffix.strip()
        xsdir_path = xsdir_path.resolve(strict=True)
        _validate_xsdir(xsdir_path, suffix, scenarios, config)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "configuration.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

        prepared: list[tuple[Scenario, Path, list[Monitor], list[float]]] = []
        for scenario in scenarios:
            input_path, monitors, energy_bounds = _write_scenario_deck(
                output_dir, scenario, config, suffix
            )
            prepared.append((scenario, input_path, monitors, energy_bounds))
            print(f"Wrote {input_path}")

        if args.prepare_only:
            print(
                "Input decks prepared. They use the XSDIR-verified MCNP "
                "continuous-energy tables selected by the supplied suffix."
            )
            return 0

        executable = _resolve_mcnp_executable(
            args.mcnp_executable or config.get("mcnp_executable")
        )
        results: list[PointResult] = []
        requested_histories = config.get("histories")
        for scenario, input_path, monitors, energy_bounds in prepared:
            tallies = _run_scenario(
                executable,
                xsdir_path,
                input_path,
                monitors,
                energy_bounds,
                requested_histories,
            )
            for monitor in monitors:
                tally = tallies[monitor.tally_id]
                results.append(
                    PointResult(
                        scenario=scenario.name,
                        distance_cm=monitor.distance_cm - SOURCE_POSITION_CM,
                        neutrons_per_source_neutron=tally.value,
                        relative_error=tally.relative_error,
                        mean_energy_mev=_mean_energy(
                            energy_bounds, tally.energy_bin_values
                        ),
                    )
                )
            print(
                f"Completed {scenario.name}: {requested_histories:,} source histories"
            )

        _save_results(output_dir, results, config, xsdir_path, suffix)
        print(f"Saved results and plots in {output_dir}")
        return 0
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
