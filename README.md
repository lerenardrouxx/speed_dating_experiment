# Reciprocal Recommendation for Speed Dating

Predicting mutual second-date decisions **before the first date**, on the Columbia speed-dating
experiment (Fisman & Iyengar, 2002–2004). Because a match requires *both* people to say yes, this
is a reciprocal recommendation problem: a good pairing has to satisfy two preference models at
once, using only what is known at signup.

**Course:** Recommender Systems — Master Applied Data Science and AI, HSLU
**Authors:** Barnard Alison · Kohler Andrin · Tabalae Petronela · Sever Alin

## Read in this order

| # | File | What it is |
|---|---|---|
| 1 | `01_Speed_Dating_EDA.ipynb` | Exploratory analysis. Establishes the constraints everything else follows: the pre-date feature boundary, the leave-one-wave-out protocol and its total cold start, the scale audit, and what the popularity baseline can actually be |
| 2 | `02_Speed_Dating_Feature_Engineering.ipynb` | Builds and verifies the modelling-ready feature table. Section 15 is the hand-off contract for the modelling and evaluation steps |
| 3 | `03_Baseline_Desirability_Model.ipynb` | Baseline **Z**: popularity ranking by desirability predicted from signup profiles, with a chance-level reference and paired significance tests |
| 4a | `04_X1_Clustering.ipynb` | Method **X₁**: gender-asymmetric cluster-based reciprocal recommender, scored head-to-head against Z |
| 4b | `04_Speed_Dating_Factorization_Machine.ipynb` | Method **X₂**: factorization machine with harmonic-mean reciprocal scoring and a no-interaction ablation |
| 5 | `report.qmd` | The written report (Quarto → `report.pdf`) |

The two `04_` notebooks are parallel methods built on the same feature table; neither depends on
the other.

## Files

| Path | Role |
|---|---|
| `sd_features.py` | Feature-engineering module — the single source of truth. Notebook 02 narrates it; downstream notebooks import it |
| `baseline_desirability.py` | Baseline Z (desirability predicted from profiles, trained on a Wilson lower bound); narrated by notebook 03 |
| `baseline_results_by_wave.csv` | Baseline Z metrics per held-out wave, written by notebook 03 |
| `x1_cluster.py` | Method X₁: per-fold clustering and cluster-pair compatibility tables |
| `evaluation.py` | Ranking metrics (Recall@3, NDCG@3, Coverage@3) used by notebook `04_X1` to score X₁ and Z identically |
| `sd_fm.py` | Method X₂: factorization machine, training loop, hyperparameter search and its ranking metrics (NDCG@3, Precision@3, MRR, Coverage@3) |
| `Data/Speed_Dating_Data.csv` | Raw dataset, 8,378 rows × 195 columns (Latin-1 encoded) |
| `Data/features/speed_dating_features.csv` | Built by notebook 02 and **committed** — 8,368 directed encounters × 102 columns, deliberately unscaled. Load it directly; running notebook 02 is only needed to rebuild it |
| `Data/features/feature_dictionary.csv` | Built by notebook 02 and **committed** — every exported column with its definition, source, range and coverage |
| `make_report_figures.py`, `figures/` | Regenerates the three report figures |
| `report.qmd`, `references.bib` | The report and its bibliography |
| `dataset_key_description.qmd` | Column dictionary for all 195 raw columns, with the pre-date / leakage boundary |
| `reciprocal_rs_slide.pptx` | Project proposal deck — fixes the baseline, the two methods, the protocol and the metrics |
| `requirements.txt` | Python dependencies, pinned to the environment the notebooks were executed in |

## How to run

Python **3.11.9**.

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                   # includes PyTorch, needed by notebook 04 (X₂)
jupyter lab
```

Open the notebooks **from the repository root** — data paths are relative and the notebooks import
the modules above from there. All notebooks run top to bottom and are deterministic (`SEED = 42`).
To re-execute a notebook without opening Jupyter:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace 01_Speed_Dating_EDA.ipynb
```

Measured runtimes of the last full execution: notebook 01 about 90 s, 02 about 25 s, 03 about 60 s,
`04_X1` about 80 s and `04_Speed_Dating_Factorization_Machine` about 75 s. Running notebook 02
rewrites `Data/features/`; running notebook 03 rewrites `baseline_results_by_wave.csv`.

## Reproducibility

All analysis code used in `report.qmd` lives in this repository:

- `01_Speed_Dating_EDA.ipynb` — the exploratory analysis, cited in the report as **EDA §n**;
- `02_Speed_Dating_Feature_Engineering.ipynb` — the feature construction, cited as **FE §n**. It
  narrates and verifies `sd_features.py`, the module holding the feature logic, so that the
  modelling and evaluation steps import one definition rather than reimplementing it;
- `03_Baseline_Desirability_Model.ipynb` — the popularity baseline
  (`baseline_desirability.py`);
- `04_X1_Clustering.ipynb` and `04_Speed_Dating_Factorization_Machine.ipynb` — the two candidate
  methods, developed in parallel and sharing the `04` prefix: the cluster-based method
  (`x1_cluster.py`, ranking metrics in `evaluation.py`) and the factorization machine (`sd_fm.py`).

All notebooks run top to bottom from a cold kernel on Python 3.11.9 with the pinned
`requirements.txt` and a fixed seed (`SEED = 42`); `make_report_figures.py` regenerates the report figures, and the exported feature table is committed alongside a generated data dictionary giving the
definition, source, documented range and missingness of every column.

## Status

Done: EDA, feature engineering, baseline **Z**, method **X₁** (cluster-based) and method **X₂**
(factorization machine), each evaluated under 21-fold leave-one-wave-out cross-validation.

In progress: a single shared evaluator and a side-by-side comparison of Z, X₁ and X₂ (they currently
report different metric sets, see `evaluation.py` and `sd_fm.py`), and the remaining report
sections.
