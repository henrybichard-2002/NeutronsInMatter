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

`openmc_neutron_shielding.py` runs a continuous-energy, fixed-source OpenMC
model for a directional 14.1 MeV neutron beam through planar material layers.
The settings at `settings/openmc_neutron_shielding.json` uses the preconverted 293.6 K neutron
HDF5 tables under `Npaths/JEFF40-Processed-OPENMC-HDF5` directly, avoiding
reconversion and duplicate data files. If that setting is removed, the program
can instead convert only required neutron ACE tables and cache them under
`outputs/openmc_neutron_shielding/openmc_hdf5_data/`. The selected thermal
scattering ACE table is converted and cached when required.

**OpenMC is not available as a native Windows conda package.** Run the
simulation from Linux, WSL2, or Docker. For WSL2, install Miniforge/Conda in
your Linux distribution, then from the project directory mounted under
`/mnt/c`:

```bash
cd /mnt/c/Users/Romul/Analysis/NeutronsInMatter
conda env create -f environment-openmc.yml
conda activate neutrons-openmc
python openmc_neutron_shielding.py
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
thermal ACE set includes the polyethylene (`hch2`) table, configured as
OpenMC's `c_H_in_CH2`. It does **not** include a water bound-hydrogen table, so
the default scenarios omit water rather than silently using a less accurate
free-gas approximation. Add a JEFF-derived water thermal table and configure
it under `thermal_scattering_tables` before adding water to a scenario.
All built-in materials use the available 293.6 K isotope data. The program
does not claim to independently benchmark the processed nuclear data.

Edit the settings JSON to change the source, history count, batches, materials,
layer thicknesses, sampling interval, or source rate. Each scenario gets its
own folder in `outputs/openmc_neutron_shielding/`; CSV results, the plot,
copied settings and run metadata are saved at that folder's root. Use
`--output-dir` to select a different output folder, `--threads` to set the CPU
thread count, or `--prepare-only` to convert data and export model XML without
running particle histories.

Tally results are positive-direction surface crossings per source neutron,
using OpenMC's surface-current score filtered to particles coming from the
upstream cell. Repeated forward crossings are included. Mean energy is
estimated from logarithmic energy-bin representative values. Reflective
lateral boundaries approximate an infinite slab and intentionally prevent
lateral leakage; this is not a finite-shield or point-source geometry.

Run the transport-script tests with:

```powershell
python -m unittest test_openmc_neutron_shielding
```

## Data

The raw JEFF40 and processed ACE data directories are local nuclear-data
inputs and should not be committed. Keep your existing `Npaths` data files
available alongside the project when running the transport model.
