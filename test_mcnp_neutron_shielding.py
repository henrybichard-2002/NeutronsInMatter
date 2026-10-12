import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import mcnp_neutron_shielding as shielding


class CliTests(unittest.TestCase):
    def test_missing_xsdir_reports_setup_instructions_without_argparse_failure(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = shielding.main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("No MCNP XSDIR file is configured", stderr.getvalue())
        self.assertIn("settings\\mcnp_neutron_shielding.json", stderr.getvalue())
        self.assertIn("--xsdir <path>", stderr.getvalue())

    def test_xsdir_can_be_configured_relative_to_settings_file(self):
        config = {
            "nuclear_data_library": "JEFF-4.0",
            "xsdir": "xsdir",
            "neutron_data_suffix": "80c",
            "source_energy_mev": 14.1,
            "histories": 1000,
            "sample_interval_cm": 1.0,
            "energy_groups": 24,
            "thermal_scattering_tables": {},
            "custom_materials": {},
            "scenarios": [
                {
                    "name": "aluminium",
                    "layers": [{"material": "aluminium", "thickness_cm": 1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            settings_dir = Path(directory) / "settings"
            settings_dir.mkdir()
            settings_path = settings_dir / "transport.json"
            settings_path.write_text(json.dumps(config), encoding="utf-8")
            isotope_tables = {
                f"{zaid}.80c"
                for zaid in shielding._material_nuclides("aluminium", {})
            }
            (settings_dir / "xsdir").write_text(
                "directory=.\n"
                + "".join(
                    f"{table} 1.0 data 0 1 1 1 0 0 293.6\n"
                    for table in isotope_tables
                ),
                encoding="utf-8",
            )
            output_dir = Path(directory) / "output"

            exit_code = shielding.main(
                [
                    "--config",
                    str(settings_path),
                    "--output-dir",
                    str(output_dir),
                    "--prepare-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(list(output_dir.glob("*/model.i"))), 1)


class MaterialTests(unittest.TestCase):
    def test_dry_air_alias_uses_air_preset(self):
        self.assertEqual(shielding._canonical_material_name("Air (dry)"), "air")

    def test_water_expands_to_jeff_isotope_zaids(self):
        fractions = shielding._material_nuclides("water", {})

        self.assertAlmostEqual(sum(fractions.values()), 1.0)
        self.assertAlmostEqual(fractions[1001], 2 * 0.999885 / 3)
        self.assertAlmostEqual(fractions[8016], 0.99757 / 3)

    def test_custom_material_accepts_normalized_names(self):
        custom_materials = {
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
                "custom_materials": custom_materials,
            }
        )

        self.assertEqual(scenarios[0].layers[0].material, "stainlesssteel")
        self.assertEqual(
            shielding._material_nuclides(
                scenarios[0].layers[0].material, custom_materials
            ),
            {26054: 0.06, 26056: 0.94},
        )


class DeckTests(unittest.TestCase):
    def test_layered_deck_has_surfaces_tallies_and_bound_thermal_scattering(self):
        scenario = shielding.Scenario(
            "water then lead",
            (
                shielding.Layer("water", 2.0, 0.9982),
                shielding.Layer("lead", 1.0, 11.34),
            ),
        )
        config = {
            "source_energy_mev": 14.1,
            "histories": 10000,
            "sample_interval_cm": 1.0,
            "energy_groups": 24,
            "custom_materials": {},
            "thermal_scattering_tables": {"water": "lwtr.80t"},
        }
        deck, monitors, bounds = shielding._build_deck(scenario, config, "80c")

        self.assertEqual([monitor.distance_cm for monitor in monitors], [0, 1, 2, 3])
        self.assertEqual(len({monitor.tally_id for monitor in monitors}), 4)
        self.assertIn("SDEF CEL=1 PAR=N", deck)
        self.assertIn("F11:N +2", deck)
        self.assertIn("E11", deck)
        self.assertIn("T\n", deck)
        self.assertIn("MT1 lwtr.80t", deck)
        self.assertIn("M2", deck)
        self.assertIn("NPS 10000", deck)
        self.assertGreater(len(bounds), config["energy_groups"])
        self.assertIn(config["source_energy_mev"], bounds)


class MctalTests(unittest.TestCase):
    def test_reads_total_forward_current_and_energy_spectrum(self):
        fixture = """\
mcnp       6     10/12/26 12:00:00     1          100        12345
Small slab test
ntal     1
    11
tally    11                    1    0
f        1
    2
et       5
  1.00000E-01  1.00000E+00  1.00000E+01  1.40999859E+01
vals
  1.00000E+00  1.0000E-01  2.00000E+00  2.0000E-01
  3.00000E+00  3.0000E-01  4.00000E+00  4.0000E-01
  1.00000E+01  5.0000E-02
tfc   0
"""
        with tempfile.TemporaryDirectory() as directory:
            mctal_path = Path(directory) / "mctal"
            mctal_path.write_text(fixture, encoding="utf-8")
            histories, tallies = shielding._parse_mctal(mctal_path, {11}, 4)

        self.assertEqual(histories, 100)
        self.assertEqual(tallies[11].value, 10.0)
        self.assertEqual(tallies[11].relative_error, 0.05)
        self.assertEqual(tallies[11].energy_bin_values, (1.0, 2.0, 3.0, 4.0))
        self.assertGreater(
            shielding._mean_energy([0.1, 1.0, 10.0, 14.1], (1.0, 2.0, 3.0, 4.0)),
            0.0,
        )

    def test_rejects_missing_total_energy_bin(self):
        fixture = """\
mcnp 6 10/12/26 12:00:00 1 100 12345
No total bin
ntal 1
11
tally 11 1 0
f 1
  2
e 4
  0.1 1 10 14.1
vals
  1 0.1 2 0.2 3 0.3 4 0.4
"""
        with tempfile.TemporaryDirectory() as directory:
            mctal_path = Path(directory) / "mctal"
            mctal_path.write_text(fixture, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no total energy-binned tally"):
                shielding._parse_mctal(mctal_path, {11}, 4)


class XsdirTests(unittest.TestCase):
    def test_requires_isotope_tables_and_water_thermal_table(self):
        scenario = shielding.Scenario(
            "water",
            (shielding.Layer("water", 1.0, 0.9982),),
        )
        config = {
            "custom_materials": {},
            "thermal_scattering_tables": {"water": "lwtr.80t"},
        }
        with tempfile.TemporaryDirectory() as directory:
            xsdir_path = Path(directory) / "xsdir"
            tables = {
                f"{zaid}.80c"
                for zaid in shielding._material_nuclides("water", {})
            }
            tables.add("lwtr.80t")
            xsdir_path.write_text(
                "directory=.\n"
                + "".join(f"{table} 1.0 data 0 1 1 1 0 0 293.6\n" for table in tables),
                encoding="utf-8",
            )
            available = shielding._validate_xsdir(
                xsdir_path, "80c", (scenario,), config
            )

        self.assertIn("lwtr.80t", available)


if __name__ == "__main__":
    unittest.main()
