# Neutrons in Matter

Python scripts for plotting neutron cross sections and estimating neutron
attenuation and energy loss in materials.

## Setup

Use Python 3.9 or newer. From the project directory, create and activate a
virtual environment, then install the core dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Scripts

Run scripts from any working directory; they locate project input files next
to the scripts and save generated plots under `outputs/`:

```powershell
python n_XsectionLookup.py
python N_attenuation_energyLoss.py
```

`n_XsectionLookup.py` reads `air(dry)_nXsection_data.json` and writes
`outputs/all_cross_sections_plot.png`. `N_attenuation_energyLoss.py` reads
`H20(liq)_nXsection_data.json` and writes
`outputs/neutron_analysis_results.png`. `TEMP.py` also saves its plot in
`outputs/`. Generated results are ignored by Git.

`TEMP.py` is an optional experiment that reads
`JEFF40/n_82-Pb-206g.jeff`. The raw JEFF evaluation files are not included in
this repository; obtain them separately and place the required file in a local
`JEFF40` directory. Install its additional dependency with
`python -m pip install -r requirements-jeff.txt`.

## MCNP shield transport

`mcnp_neutron_shielding.py` generates and runs MCNP continuous-energy
simulations for a normally incident, monoenergetic 14.1 MeV neutron beam
passing through configurable planar material layers. The example compares
air, lead, aluminium, water, and HDPE. Output includes forward surface current
per source neutron, its MCNP statistical relative error, and an energy-binned
estimate of the mean forward-neutron energy versus depth. If the source rate is
known, set `source_rate_neutrons_per_s` in the settings file to also
report neutrons per second.

**The raw JEFF `.jeff` files in `JEFF40` are ENDF evaluations, not MCNP
cross-section tables.** Process the JEFF neutron evaluations into continuous-
energy ACE tables with NJOY, and make the resulting MCNP tables available
through an `xsdir` file. Supply the suffix used by those actual tables; the
program checks that the selected isotope tables are listed in `xsdir`. It
cannot independently verify their nuclear-data provenance, so retain and check
the NJOY processing records. Water and HDPE also require corresponding
JEFF-derived bound-hydrogen thermal-scattering ACE tables, configured under
`thermal_scattering_tables`. The program deliberately stops instead of
silently substituting free-gas scattering data when those tables are missing.

Install MCNP separately under its applicable license; it is not bundled with
this repository. After processing JEFF into MCNP-readable ACE data and
configuring the suffix and thermal table names in
`settings/mcnp_neutron_shielding.json`, run:

```powershell
python mcnp_neutron_shielding.py `
  --xsdir "C:\path\to\jeff-ace\xsdir"
```

Edit `settings/mcnp_neutron_shielding.json` to change the source, histories,
material layers, scenarios, or thermal-scattering tables. The default run
directory is `outputs/mcnp_neutron_shielding/`; each scenario gets a
subdirectory containing its MCNP input and output files, while the CSV, plot,
and metadata are written at the run-directory root. Use `--config` to select a
different settings file or `--output-dir` to choose another results location.
Relative paths supplied to those options are resolved from the current working
directory; the defaults are based on the project location.

Set `xsdir` to the path of the processed JEFF ACE `xsdir` file and
`neutron_data_suffix` to its continuous-energy table suffix in that settings
file to run without command-line overrides. Relative `xsdir` paths in the
settings file are resolved relative to the settings file itself. The
`--xsdir` and `--library-suffix` command-line options override these values.
Until an XSDIR is supplied, running without `--xsdir` reports this setup
requirement directly rather than failing during argument parsing.

Use the suffix (shown here as `80c`) and thermal-table names actually present
in your JEFF-processed `xsdir`; they are examples, not claims about any
particular processed library. Add or edit scenarios and layer thicknesses in
the settings JSON; set a layer's `density_g_cm3` when its density differs from
the built-in room-temperature reference. Materials beyond the built-in presets
can be added with explicit isotope `ZAID` atom fractions in `custom_materials`.
For the default room-temperature material densities, use appropriately
temperature-processed isotope and thermal-scattering ACE tables (about
293.6 K); changing a material density does not change the nuclear-data
temperature.

`--prepare-only` validates the configured tables and writes the MCNP input
decks without running transport. Completed MCNP output is saved as
`neutron_transport_results.csv`, `neutron_transport_vs_depth.png`, and
`run_metadata.json` in `outputs/mcnp_neutron_shielding/` by default. Tally
values represent forward
surface crossings per source neutron; a neutron that scatters back and crosses
a tally plane forward again can contribute another crossing. Mean energies use
logarithmic energy-bin representative values. Raise the history count and
refine the energy bins to study statistical uncertainty and energy resolution.
The transport model is a one-dimensional infinite slab (no lateral escape),
not a finite shield or a point-source geometry.

Run the transport-script tests with:

```powershell
python -m unittest test_mcnp_neutron_shielding
```

## Data

The raw JEFF40 directory and `JEFF40-Evaluations-Neutron-593.zip` are
intentionally excluded from version control. The smaller JSON and CSV data
files already in the project directory are retained as analysis inputs.
