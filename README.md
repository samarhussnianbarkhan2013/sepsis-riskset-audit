# Timing leakage and transportability in early sepsis prediction

Code and results for the paper *Timing Leakage and the Transportability Gap in Early Sepsis Prediction: A Risk-Set Audit of the PhysioNet/CinC 2019 Benchmark*.

## Data

All analyses use the training release (version 1.0.0) of the PhysioNet/Computing in Cardiology Challenge 2019 database, which is openly available from PhysioNet under the CC BY 4.0 licence. Each notebook downloads it from the PhysioNet open-data mirror; no credentials are needed.

## Running the analyses

Open a notebook in Google Colab, select a CPU runtime and choose Runtime > Run all. Each notebook writes one JSON file of results. The files used for the paper are in `results/`. Random seeds are fixed in the code (2026-2030 for model fitting and splits; 2019 + h and 2119 + h for landmark sampling in systems A and B).

| Notebook | Results file | Content in the paper |
|---|---|---|
| `01_record_timing_and_weighting.ipynb` | `results_01.json` | Record timing after onset (Section IV-A); domain-classifier AUROC with all features (Section IV-E) |
| `02_exact_risk_sets.ipynb` | `results_02.json` | Timing probe of the time-matched design (Table II); effect of adding the landmark time (Section IV-B) |
| `03_matching_ratio_feature_families_horizons.ipynb` | `results_03.json` | Fig. 2; full-window rows of Tables II and III; Table IV; Table VI; cross-evaluation (Section IV-B) |
| `04_common_support_stacked_and_hourly.ipynb` | `results_04.json` | Common-support rows of Tables II and III; stacked ensemble and hourly designs (Table III, Fig. 3) |
| `05_primary_cohort_analyses.ipynb` | `results_05.json` | Primary cohort (Table II); Table V; Fig. 4; Table VII; Fig. 5 |

`figures/make_figures.py` regenerates Figs. 1-5 from `results/`:

```
cd figures
python make_figures.py
```

## Software

Python 3 with numpy, pandas, scikit-learn and matplotlib as provided by Google Colab; awscli is installed by the download cell.

## Licence

The code is released under the MIT licence. The PhysioNet/CinC 2019 data remain under their CC BY 4.0 licence.
