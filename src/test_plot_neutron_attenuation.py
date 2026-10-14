import csv
import json
import tempfile
import unittest
from pathlib import Path

from plot_neutron_attenuation import plot_run


class AttenuationPlotTests(unittest.TestCase):
    def test_plots_transport_and_spectrum_csvs_into_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "run"
            output_dir = Path(directory) / "plots"
            input_dir.mkdir()
            (input_dir / "configuration.json").write_text(
                json.dumps(
                    {
                        "source_energy_mev": 14.1,
                        "scenarios": [
                            {
                                "name": "test",
                                "layers": [
                                    {
                                        "material": "air",
                                        "thickness_mm": 2.0,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (input_dir / "neutron_transport_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "scenario",
                        "distance_mm",
                        "forward_neutron_transmission_fraction",
                        "forward_energy_transmission_fraction",
                        "mean_forward_energy_mev",
                        "mean_forward_energy_std_dev_mev",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "scenario": "test",
                            "distance_mm": 0.0,
                            "forward_neutron_transmission_fraction": 1.0,
                            "forward_energy_transmission_fraction": 1.0,
                            "mean_forward_energy_mev": 14.1,
                            "mean_forward_energy_std_dev_mev": 0.0,
                        },
                        {
                            "scenario": "test",
                            "distance_mm": 2.0,
                            "forward_neutron_transmission_fraction": 0.8,
                            "forward_energy_transmission_fraction": 0.6,
                            "mean_forward_energy_mev": 10.575,
                            "mean_forward_energy_std_dev_mev": 0.2,
                        },
                    ]
                )
            with (input_dir / "neutron_energy_spectrum.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "scenario",
                        "distance_mm",
                        "energy_min_mev",
                        "energy_max_mev",
                        "forward_neutrons_per_source_neutron",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "scenario": "test",
                            "distance_mm": distance,
                            "energy_min_mev": lower,
                            "energy_max_mev": upper,
                            "forward_neutrons_per_source_neutron": value,
                        }
                        for distance, values in (
                            (0.0, (0.9, 0.1)),
                            (2.0, (0.7, 0.1)),
                        )
                        for (lower, upper), value in zip(
                            ((1.0e-5, 0.1), (0.1, 20.0)), values
                        )
                    ]
                )

            paths = plot_run(input_dir, output_dir, "test")

            self.assertEqual(len(paths), 2)
            self.assertTrue(
                (output_dir / "neutron_attenuation_test.png").is_file()
            )
            self.assertTrue((output_dir / "neutron_spectrum_test.png").is_file())


if __name__ == "__main__":
    unittest.main()
