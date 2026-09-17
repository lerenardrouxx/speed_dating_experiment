"""Script version of the comparison run -- verified here, then turned into 05_Comparison_Evaluation.ipynb.

Runs random / Z (baseline) / X1 (clustering) / FM (factorization machine) through the ONE shared
evaluator (evaluation.py), all under the same decider_iid/candidate_id/score/match schema, and
produces the results table + figure for report.qmd's Results section.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

import sd_features as sf
import x1_cluster as x1
import sd_fm as fm
import evaluation as ev

SEED = 42
K = 3


def to_shared_schema(df: pd.DataFrame, score_col: str, decider_col: str = "iid",
                      candidate_col: str = "pid") -> pd.DataFrame:
    """Rename any method's own (decider, candidate, score, match) columns onto the shared
    evaluator's contract, without touching the method's own module."""
    out = df.rename(columns={decider_col: "decider_iid", candidate_col: "candidate_id",
                              score_col: "score"})
    return out[["decider_iid", "candidate_id", "score", "match", "wave"]]


# --------------------------------------------------------------------------------------------
# Z: popularity baseline (predicted desirability), reshaped onto the shared schema
# --------------------------------------------------------------------------------------------

def run_z(raw: pd.DataFrame, person: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    import baseline_desirability as bd

    waves = sorted(raw["wave"].dropna().unique())
    parts = []
    for w in waves:
        train_iids = raw.loc[raw["wave"] != w, "iid"].unique()
        test_iids = raw.loc[raw["wave"] == w, "iid"].unique()

        y_train = bd.wilson_desirability(raw, train_iids).dropna()
        X_train = person.loc[person.index.isin(y_train.index)]
        y_train = y_train.loc[X_train.index]

        prep = bd.make_profile_preprocessor()
        X_train_t = prep.fit_transform(X_train)
        model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                       random_state=SEED)
        model.fit(X_train_t, y_train.to_numpy())

        X_test = person.loc[person.index.isin(test_iids)]
        X_test_t = prep.transform(X_test)
        pred = pd.Series(model.predict(X_test_t), index=X_test.index)

        test_pairs = raw.loc[(raw["wave"] == w) & raw["pid"].notna(),
                              ["iid", "pid", "match"]].copy()
        test_pairs["pid"] = test_pairs["pid"].astype("int64")
        test_pairs["score"] = test_pairs["pid"].map(pred)
        test_pairs["wave"] = w
        parts.append(test_pairs)

    return to_shared_schema(pd.concat(parts, ignore_index=True), score_col="score")


# --------------------------------------------------------------------------------------------
# X1: clustering, reshaped onto the shared schema
# --------------------------------------------------------------------------------------------

def run_x1(feat: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for w in sorted(feat["wave"].unique()):
        train_df = feat[feat["wave"] != w]
        test_df = feat[feat["wave"] == w]
        model = x1.fit_x1(train_df, seed=SEED)
        pred = x1.predict_x1(model, test_df)
        rows = test_df[["iid", "pid", "match"]].copy()
        rows["score"] = pred["x1_score"].to_numpy()
        rows["wave"] = w
        parts.append(rows)
    return to_shared_schema(pd.concat(parts, ignore_index=True), score_col="score")


# --------------------------------------------------------------------------------------------
# FM: reuses sd_fm.py's own 21-fold loop and hyperparameter search, reshaped onto the shared
# schema -- this is the "adapter" approach: sd_fm.py itself is untouched, only its OUTPUT is
# relabelled here so it can go through the one shared evaluator instead of the module's own
# duplicate metric functions.
# --------------------------------------------------------------------------------------------

def run_fm(feat: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if cfg is None:
        cfg, search_results = fm.select_hyperparams(feat, seed=SEED)
    else:
        search_results = None
    scored_all, fold_diag = fm.run_lowo_fm(feat, cfg, seed=SEED)
    scored_all = scored_all.rename(columns={"iid": "decider_iid", "pid": "candidate_id",
                                             "p_match": "score"})
    return scored_all[["decider_iid", "candidate_id", "score", "match", "wave"]], {
        "cfg": cfg, "search_results": search_results, "fold_diag": fold_diag,
    }


def summarize(name: str, scored_all: pd.DataFrame) -> dict:
    res = ev.summarize_folds_and_users(scored_all, k=K)
    by_fold, chance = res["by_fold"], res["chance_by_fold"]
    row = {"method": name}
    for metric in [f"Recall@{K}", f"NDCG@{K}", f"Precision@{K}", "MRR", f"Coverage@{K}"]:
        row[f"{metric}_mean"] = by_fold[metric].mean()
        row[f"{metric}_std"] = by_fold[metric].std()
    tests = ev.paired_tests(by_fold[f"NDCG@{K}"], chance[f"NDCG@{K}"])
    row["vs_chance_mean_diff_ndcg3"] = tests["mean_diff"]
    row["vs_chance_folds_won"] = f"{tests['n_favouring_method']}/{tests['n_pairs']}"
    row["vs_chance_p_ttest"] = tests["p_ttest"]
    row["vs_chance_p_wilcoxon"] = tests["p_wilcoxon"]
    return row, res


if __name__ == "__main__":
    t0 = time.time()
    raw = sf.load_raw()
    person = sf.build_person_table(raw)
    feat = sf.load_features()
    print(f"Data loaded ({time.time()-t0:.1f}s)")

    scored_z = run_z(raw, person)
    print(f"Z scored ({time.time()-t0:.1f}s), rows={len(scored_z)}")

    scored_x1 = run_x1(feat)
    print(f"X1 scored ({time.time()-t0:.1f}s), rows={len(scored_x1)}")

    scored_fm, fm_info = run_fm(feat)
    print(f"FM scored ({time.time()-t0:.1f}s), rows={len(scored_fm)}, cfg={fm_info['cfg']}")

    rows = []
    results_by_method = {}
    for name, scored in [("Z (popularity baseline)", scored_z),
                          ("X1 (clustering)", scored_x1),
                          ("FM (factorization machine)", scored_fm)]:
        row, res = summarize(name, scored)
        rows.append(row)
        results_by_method[name] = res
        print(f"{name} summarized ({time.time()-t0:.1f}s)")

    table = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.width", 160)
    print()
    print(table)
    table.to_csv("comparison_results.csv")
    print(f"\nSaved comparison_results.csv. Total time: {time.time()-t0:.1f}s")

    # FM vs Z, the actual research-question comparison
    fm_by_fold = results_by_method["FM (factorization machine)"]["by_fold"]
    z_by_fold = results_by_method["Z (popularity baseline)"]["by_fold"]
    fm_vs_z = ev.paired_tests(fm_by_fold[f"NDCG@{K}"], z_by_fold[f"NDCG@{K}"])
    print("\nFM vs Z, NDCG@3:", fm_vs_z)
