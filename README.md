# Onion Price Retraining Benchmark (Indian Mandis)

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A data-validated MLOps benchmark comparing calendar-based and drift-triggered retraining policies for 4-week-ahead wholesale onion price forecasting across 223 Indian agricultural markets (*mandis*) over 362 evaluation weeks under realistic 4-week label latency.

## Key Experimental Findings

| Benchmark Dimension | Verified Finding |
| :--- | :--- |
| **Six-Check Data Validation Gate (G1–G6)** | Screens **965,314** raw daily rows (818 markets, 2014–2024), removing **1.02%** invalid range/modal-price records |
| **Value-Based Lag Audit (G6)** | Pre-computed `modal_price_lag_7` ($n=959{,}893$ non-null) matches market-row shifts on **95.9%** of non-null values, but matches 7-calendar-day market-date medians on only **33.0%** (**82.5%** do not reference 7 calendar days earlier) |
| **Metric-Dependent Policy Rankings** | Default **Page–Hinkley** achieves lowest MAE (**451.6 Rs/quintal**, 7 models) and Log-MAE (**0.2204**); **Weekly** (362 models) and **Skill-loss** (66 models) tie for lowest MAPE (**23.35%**); static/ADWIN (**24.58%**) trails persistence (**23.72%**) on MAPE |
| **Circular Block Bootstrap (108 Intervals)** | Across all 9 non-reference policies $\times$ 3 metrics (`MAE`, `MAPE`, `Log-MAE`) $\times$ 4 block lengths (`4, 13, 26, 52` weeks), **108 / 108** 95% CIs vs. quarterly retraining include zero (84 distinct intervals; persistence $\Delta\text{MAE} = +21.2$ `[-35.2, +85.8]` Rs/quintal at 13w) |
| **Threshold Sensitivity Grid** | 9-setting Page–Hinkley grid spans **5–14 models** (MAE **442.3–454.9** Rs/quintal, MAPE **22.62–23.40%**), with **9 / 9** 13-week MAE CIs vs. quarterly including zero. 4-setting PSI grid (`0.10–0.30`) fires at every eligible check (**91 models**) and cannot discriminate in this 4-vs-156-week design |
| **Kaplan–Meier Time to Next Retrain** | Over 12 merged conditional shock episodes: **PSI** 1.0w (`12/12`), **Skill** 1.5w midpoint / 1.0w smallest-$t$ (`12/12`), **Quarterly** 5.0w (`12/12`), **Page–Hinkley** 17.5w midpoint / 17.0w smallest-$t$ (`11/12` followed by a retrain before evaluation end), **ADWIN (ours)** **not reached** (`0/12`) |

## Repository Structure

```text
.
├── convert_mendeley_to_parquet.py          # Converts raw Mendeley V1 CSVs -> typed Parquet + SHA-256
├── Paper3_OnionRetraining.ipynb            # Core data-validation, leakage ablation & retraining notebook
├── Paper3_OnionRetraining_EXECUTED.ipynb   # Executed benchmark notebook
├── make_notebook.py                        # Notebook generator
├── run_round2_p3.py                        # Round 2 bootstrap, grid, delay & G6 value-audit pipeline
├── run_round3_p3.py                        # Final Kaplan–Meier, 108-CI block sensitivity & PH grid CIs
├── gen_tables.py                           # Programmatic summary & table generator
└── results/
    ├── table_gate.csv                      # Six-check validation gate summary
    ├── table_g6_value_audit.csv            # Value-based lag audit (market-row vs. market-date median)
    ├── table_leakage.csv                   # Provided vs. strictly calendar-causal lag ablation
    ├── table_policies.csv                  # 10-policy MAE, MAPE, Log-MAE, shock/normal MAE
    ├── table_bootstrap_vs_quarterly.csv    # 13-week circular block-bootstrap CIs (MAE/MAPE/Log-MAE)
    ├── table_bootstrap_block_sensitivity.csv # Full 108-interval (4/13/26/52w) sensitivity grid
    ├── table_threshold_grid.csv            # 9-setting Page–Hinkley + 4-setting PSI grid with 95% CIs
    ├── table_detection_delay.csv           # Kaplan–Meier time-to-next-retrain on 12 merged episodes
    ├── models/                             # Cached weekly out-of-sample predictions (`pred_T*.npy`)
    └── fig_pareto.png/.pdf, fig_timeline.png/.pdf # Pareto frontier & shock-retrain timeline
```

## Quick Start & Reproducibility

Because `data/onion_2014_2024.parquet` (108 MB) exceeds GitHub's 100 MB file limit, download the public Mendeley Data V1 CSV release first and convert it locally:

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib pyarrow

# 2. Convert Mendeley Data V1 CSVs (DOI: 10.17632/ds9jmxp9zy.1) to Parquet
python convert_mendeley_to_parquet.py --csv-dir /path/to/mendeley_csvs --out data/onion_2014_2024.parquet

# 3. Replay Kaplan–Meier delays, 108-CI block bootstrap, and 9-setting PH threshold grid on cached predictions
python run_round3_p3.py

# 4. Regenerate summary tables and outputs
python gen_tables.py
```

## Data Attribution

Uses the public spatio-temporal Indian onion price, weather, and event dataset by Shelar, R., Hajare, A., and Mangarule, S. (Mendeley Data, Version 1, 2025, CC BY 4.0, DOI: `10.17632/ds9jmxp9zy.1`), integrating Agmarknet wholesale mandi records and IMD gridded weather features.

## Topics / Tags

`mlops` · `concept-drift` · `retraining-policy` · `time-series-forecasting` · `data-validation` · `agricultural-prices` · `page-hinkley` · `block-bootstrap` · `kaplan-meier`

## License

Code and analysis scripts are released under the **MIT License**. Underlying public market datasets remain under **CC BY 4.0**.
