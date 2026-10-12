import json
import tempfile
import unittest
from pathlib import Path

import openmc_neutron_shielding as shielding


class ConfigTests(unittest.TestCase):
    def test_default_settings_use_local_preconverted_jeff_hdf5_data(self):
        config = json.loads(
            shielding.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        )

        shielding._validate_config(config)
        scenarios = shielding._make_scenarios(config)
        hdf5_dir = (
            shielding.DEFAULT_CONFIG_PATH.parent
            / config["neutron_hdf5_directory"]
        ).resolve()
        hdf5_files = shielding._index_hdf5_files(hdf5_dir)
        required_zaids = {
            zaid
            for scenario in scenarios
            for layer in scenario.layers
            for zaid in shielding._material_nuclides(
                layer.material, config["custom_materials"]
            )
        }
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(required_zaids - hdf5_files.keys(), set())
        self.assertFalse(
            any(layer.material == "water" for case in scenarios for layer in case.layers)
        )
        self.assertEqual(
            config["thermal_scattering_tables"]["hdpe"]["openmc_name"],
            "c_H_in_CH2",
        )

    def test_missing_water_thermal_data_is_not_silently_accepted(self):
        scenario = shielding.Scenario(
            "water",
            (shielding.Layer("water", 1.0, 0.9982),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neutron_dir = root / "neutrons"
            thermal_dir = root / "thermal"
            neutron_dir.mkdir()
            thermal_dir.mkdir()
            for filename in (
                "1-H-1g-293.6",
                "1-H-2g-293.6",
                "8-O-16g-293.6",
                "8-O-17g-293.6",
                "8-O-18g-293.6",
            ):
                (neutron_dir / filename).touch()
            config_path = root / "settings.json"
            config_path.write_text("{}", encoding="utf-8")
            config = {
                "neutron_ace_directory": str(neutron_dir),
                "thermal_ace_directory": str(thermal_dir),
                "thermal_scattering_tables": {},
                "custom_materials": {},
            }
            with self.assertRaisesRegex(ValueError, "bound-hydrogen thermal scattering"):
                shielding._convert_ace_data(
                    object(), config, config_path, (scenario,), root / "cache"
                )


class MaterialTests(unittest.TestCase):
    def test_dry_air_alias_uses_air_preset(self):
        self.assertEqual(shielding._canonical_material_name("Air (dry)"), "air")

    def test_water_isotope_fractions_are_normalized(self):
        fractions = shielding._material_nuclides("water", {})

        self.assertAlmostEqual(sum(fractions.values()), 1.0)
        self.assertAlmostEqual(fractions[1001], 2 * 0.999885 / 3)
        self.assertAlmostEqual(fractions[8016], 0.99757 / 3)

    def test_custom_material_accepts_normalized_names(self):
        custom = {
            "Stainless Steel": {
                "density_g_cm3": 7.9,
                "nuclide_atom_fractions": {"26054": 0.06, "26056": 0.94},
            }
        }
        scenarios = shielding._make_scenarios(
            {
                "scenarios": [
                    {
                        "name": "steel",
                        "layers": [
                            {"material": "stainless_steel", "thickness_cm": 1.0}
                        ],
                    }
                ],
                "custom_materials": custom,
            }
        )
        self.assertEqual(scenarios[0].layers[0].material, "stainlesssteel")
        self.assertEqual(
            shielding._material_nuclides("stainlesssteel", custom),
            {26054: 0.06, 26056: 0.94},
        )

    def test_zaid_converts_to_openmc_gnds_name(self):
        self.assertEqual(shielding._nuclide_name(1001), "H1")
        self.assertEqual(shielding._nuclide_name(82208), "Pb208")


class DataTests(unittest.TestCase):
    def test_indexes_only_supported_temperature_ace_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1-H-1g-293.6").touch()
            (root / "1-H-1g-300.0").touch()
            (root / "1-H-2g-293.6").touch()
            (root / "readme.txt").touch()

            indexed = shielding._index_ace_files(root)

        self.assertEqual(set(indexed), {1001, 1002})
        self.assertEqual(indexed[1001].name, "1-H-1g-293.6")

    def test_indexes_preconverted_hdf5_files_at_supported_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "13-Al-27g-293.6.h5").touch()
            (root / "13-Al-27g-300.0.h5").touch()
            (root / "13-Al-27m-293.6.h5").touch()
            (root / "readme.txt").touch()

            indexed = shielding._index_hdf5_files(root)

        self.assertEqual(set(indexed), {13027})
        self.assertEqual(indexed[13027].name, "13-Al-27g-293.6.h5")

    def test_preconverted_hdf5_data_is_registered_without_ace_conversion(self):
        class FakeLibrary:
            def __init__(self):
                self.files = []

            def register_file(self, path):
                self.files.append(path)

            def export_to_xml(self, path):
                Path(path).write_text("<cross_sections />", encoding="utf-8")

        class FakeData:
            DataLibrary = FakeLibrary

        class FakeOpenMC:
            data = FakeData()

        scenario = shielding.Scenario(
            "lead",
            (shielding.Layer("lead", 1.0, 11.34),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5_dir = root / "hdf5"
            hdf5_dir.mkdir()
            for zaid in shielding._material_nuclides("lead", {}):
                atomic_number, mass_number = divmod(zaid, 1000)
                symbol = shielding.ELEMENT_SYMBOLS[atomic_number - 1]
                (hdf5_dir / f"{atomic_number}-{symbol}-{mass_number}g-293.6.h5").touch()
            config_path = root / "settings.json"
            config_path.write_text("{}", encoding="utf-8")
            config = {
                "neutron_hdf5_directory": str(hdf5_dir),
                "thermal_scattering_tables": {},
                "custom_materials": {},
            }
            cross_sections, registered = shielding._convert_ace_data(
                FakeOpenMC(),
                config,
                config_path,
                (scenario,),
                root / "cache",
            )
            self.assertTrue(cross_sections.is_file())
            self.assertEqual(
                len(registered), len(shielding._material_nuclides("lead", {}))
            )
            self.assertTrue(
                all(Path(path).parent == hdf5_dir for path in registered.values())
            )

    def test_missing_ace_directory_has_actionable_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "Processed JEFF neutron ACE"):
            shielding._index_ace_files(Path("missing-ace-directory"))

    def test_energy_edges_include_zero_and_source_energy(self):
        edges = shielding._energy_edges_mev(14.1, 24)

        self.assertEqual(edges[0], 0.0)
        self.assertIn(14.1, edges)
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(shielding._energy_bin_centers(edges)), len(edges) - 1)

    def test_monitor_positions_include_material_interfaces(self):
        scenario = shielding.Scenario(
            "two layers",
            (
                shielding.Layer("lead", 1.5, 11.34),
                shielding.Layer("aluminium", 2.0, 2.7),
            ),
        )

        self.assertEqual(
            shielding._monitor_distances(scenario, 1.0),
            [0.0, 1.0, 1.5, 2.0, 3.0, 3.5],
        )


if __name__ == "__main__":
    unittest.main()
