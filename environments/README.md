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

## OpenMC shield transport

`src/openmc_neutron_shielding.py` runs continuous-energy, fixed-source OpenMC
transport for a directional neutron beam through a one-dimensional stack of
planar layers. The editable example configuration is
`settings/openmc_neutron_shielding.json`; its default stack is 70 mm air,
2 mm aluminium, 10 mm water, 2 mm aluminium, then 100 mm air. Edit each
layer's `material` and `thickness_mm`, or add/remove layer objects to change
the material order and thicknesses. Built-in materials include air, aluminium,
HDPE, lead, and water; define additional materials in `custom_materials`.

All user-facing geometry settings and reported distances use millimetres:
`thickness_mm`, `sample_interval_mm`, `beam_width_mm`, and
`lateral_half_width_mm`. Layer boundaries are always scored in addition to
regular sample positions. OpenMC internally expects coordinates in centimetres,
so the model converts configured millimetres to centimetres only when creating
its transport geometry. Material density remains in g/cm^3.

**OpenMC is not available as a native Windows conda package.** Run the
simulation from Linux, WSL2, or Docker. For WSL2, install Miniforge/Conda in
your Linux distribution, then from the project root:

```bash
conda env create -f environments/environment-openmc.yml
conda activate neutrons-openmc
python src/openmc_neutron_shielding.py
```

If WSL2 is not installed, install it from an elevated Windows PowerShell with
`wsl --install -d Ubuntu`, restart if Windows asks, then install Miniforge in
Ubuntu before running the commands above.

The local JEFF data paths in the default settings are relative to that settings
file and resolve to the `Npaths/JEFF40-Processed-OPENMC-HDF5`,
`Npaths/JEFF40-Processed-Neutron-ACE-293K`, and
`Npaths/JEFF40-Processed-TSL-selected-ACE/data` directories. Prepared HDF5
neutron tables are preferred; use `--neutron-hdf5-dir` to override their
location. The ACE fallback can be selected with `--neutron-ace-dir`, and the
thermal ACE path can be overridden with `--thermal-ace-dir`.

The input directory contains processed 293.6 K JEFF neutron ACE tables; the
original `.jeff` evaluations alone are not transport-ready. The supplied
thermal ACE set includes polyethylene (`hch2`) and light water (`hh2o`) tables,
configured as OpenMC's `c_H_in_CH2` and `c_H_in_H2O`.
All built-in materials use the available 293.6 K isotope data. The program
does not claim to independently benchmark the processed nuclear data.

The settings JSON also controls the source energy/rate, history count, batches,
energy-group count, and sample interval. Each scenario gets its own folder in
`outputs/openmc_neutron_shielding/`; model files are saved in those scenario
folders. Use
`--output-dir` to select a different output folder, `--threads` to set the CPU
thread count, or `--prepare-only` to convert data and export model XML without
running particle histories.

Five 10-batch run configurations are provided in `settings/`:
`reactorpath1.json`, `reactorpath2.json`, `reactorpath3.json`,
`DDModerator.json`, and `DTModerator.json`. Run one at a time, giving each a
separate output directory so the root-level summary files are kept distinct:

```bash
python src/openmc_neutron_shielding.py --config settings/reactorpath1.json --output-dir outputs/reactorpath1
python src/openmc_neutron_shielding.py --config settings/reactorpath2.json --output-dir outputs/reactorpath2
python src/openmc_neutron_shielding.py --config settings/reactorpath3.json --output-dir outputs/reactorpath3
python src/openmc_neutron_shielding.py --config settings/DDModerator.json --output-dir outputs/DDModerator
python src/openmc_neutron_shielding.py --config settings/DTModerator.json --output-dir outputs/DTModerator
```

The moderator settings model the helium-3 as pure He-3 gas at 293.6 K and
2 bar absolute, with density calculated using the ideal-gas law
(`0.000247101462395 g/cm^3`). Their configurations use the available JEFF
293.6 K He-3 neutron table. The configured history total remains 1,000,000 per
run; 10 batches means 100,000 histories per batch.

`neutron_energy_flux_heatmap.png` is the main plot: its horizontal axis is
distance in mm, its **linear** vertical axis is neutron energy in MeV, and its
color scale is per-energy-bin forward neutron flux density in N/s/cm^2. The
mean-forward-neutron-energy curve is overlaid on the same energy axis with
its one-sigma statistical uncertainty shown as a band. A labeled layer strip
below the heatmap shows the materials, thicknesses, and interface positions.

The aggregate `neutron_transport_results.csv` reports neutron flux, energy
flux, mean forward energy, and one-standard-deviation Monte Carlo uncertainty
at each scored distance. `neutron_energy_spectrum.csv` reports every energy
bin at every scored distance, with the bin's energy limits, neutron-flux
estimate, and lower/upper one-sigma ranges. CSV flux values are divided by the
modeled plane area (mm^2); the main heatmap instead uses cm^2 to report the
requested N/s/cm^2 when a source rate is configured. Repeated forward
crossings are included. Reflective lateral boundaries approximate an infinite
slab and intentionally prevent lateral leakage; this is not a finite-shield
or point-source geometry.

Run the transport-script tests with:

```powershell
python -m unittest discover -s src -p test_openmc_neutron_shielding.py
```

## Data

The raw JEFF40 and processed ACE data directories are local nuclear-data
inputs and should not be committed. Keep your existing `Npaths` data files
available alongside the project when running the transport model.
