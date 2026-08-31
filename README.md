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
| 3 | `report.qmd` | The written report (Quarto → `report.pdf`) |

## Files

| Path | Role |
|---|---|
| `sd_features.py` | Feature-engineering module — the single source of truth. Notebook 02 narrates it; downstream notebooks import it |
| `Data/Speed_Dating_Data.csv` | Raw dataset, 8,378 rows × 195 columns (Latin-1 encoded) |
| `Data/features/speed_dating_features.csv` | Built by notebook 02 and **committed** — 8,368 directed encounters × 102 columns, deliberately unscaled. Load it directly; running notebook 02 is only needed to rebuild it |
| `Data/features/feature_dictionary.csv` | Built by notebook 02 and **committed** — every exported column with its definition, source, range and coverage |
| `dataset_key_description.qmd` | Column dictionary for all 195 raw columns, with the pre-date / leakage boundary |
| `reciprocal_rs_slide.pptx` | Project proposal deck — fixes the baseline, the two methods, the protocol and the metrics |
| `requirements.txt` | Python dependencies |

## How to run

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open the notebooks **from the repository root** — data paths are relative and notebook 02 imports
`sd_features.py` from there. Both run top to bottom and are deterministic (`SEED = 42`): notebook
01 takes roughly two minutes, notebook 02 well under one. Running notebook 02 rewrites
`Data/features/`.

## Status

The EDA and the feature engineering are complete. Still to come: the popularity baseline (**Z**),
the two proposed methods (**X₁** cluster-based, **X₂** factorization machine), and the 21-fold
leave-one-wave-out evaluation on NDCG@3, Precision@3, MRR and Coverage@3. Notebook 02's Section 15
describes exactly how to consume the feature table for those steps.
