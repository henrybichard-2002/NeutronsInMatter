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

Run the scripts from the project directory so their input data files can be
found:

```powershell
python n_XsectionLookup.py
python N_attenuation_energyLoss.py
```

`n_XsectionLookup.py` reads `air(dry)_nXsection_data.json` and writes
`all_cross_sections_plot.png`. `N_attenuation_energyLoss.py` reads
`H20(liq)_nXsection_data.json` and writes `neutron_analysis_results.png`.
Generated PNG files are ignored by Git.

`TEMP.py` is an optional experiment that reads
`JEFF40/n_82-Pb-206g.jeff`. The raw JEFF evaluation files are not included in
this repository; obtain them separately and place the required file in a local
`JEFF40` directory. Install its additional dependency with
`python -m pip install -r requirements-jeff.txt`.

## Data

The raw JEFF40 directory and `JEFF40-Evaluations-Neutron-593.zip` are
intentionally excluded from version control. The smaller JSON and CSV data
files already in the project directory are retained as analysis inputs.
