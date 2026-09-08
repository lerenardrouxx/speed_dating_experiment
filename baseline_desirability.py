"""
Baseline model: predicted desirability from signup profile features.

RQ1 baseline (report.qmd, sec-baseline): a popularity ranking where each
candidate's score is their *predicted* desirability (acceptance rate),
predicted from their signup profile alone -- not their real acceptance
history, because leave-one-wave-out CV is a total cold start (0 of 551
held-out participants have training history in any fold).

Target: a Wilson lower-bound on the acceptance rate a training-wave
participant actually received (dec_o), so a person accepted 4/4 times
doesn't outrank one accepted 40/50 times just because they had fewer dates.
This matches what report.qmd already says the baseline does ("a Wilson
lower bound is applied") -- sd_features.add_behavioural_features() instead
uses additive/Bayesian shrinkage, a different formula for the same idea,
which is the discrepancy flagged in the PR review. This script implements
the Wilson version directly so the code matches the report.

Pipeline per fold (wave w held out):
  1. Target: Wilson lower bound of dec_o for every training-wave participant.
  2. Features: their signup profile only (build_person_table) -- no partner
     info, since desirability is supposed to be a property of the person,
     not of a specific pair.
  3. Fit a RandomForestRegressor: profile features -> Wilson-shrunk desirability.
     Hyperparameters (300 trees, max_depth=6, min_samples_leaf=5) are fixed by
     convention, not tuned -- this is a baseline, and tuning it against the
     evaluation metric would make it a hidden second "improved method" rather
     than a fair, simple reference point.
  4. Predict desirability for held-out-wave participants from their profile.
  5. Rank each test participant's met candidates by the CANDIDATE's predicted
     desirability (this is what makes it a popularity baseline: the score
     does not depend on who is choosing). Candidates are shuffled per-decider
     before the sort so ties in predicted score break randomly and
     reproducibly, not by whatever order rows happen to sit in.
  6. Score against match==1 (mutual yes) at rank 3: Recall@3, NDCG@3, coverage.

Also reports two things a plain results table would hide:
  - A signal diagnostic: how well predicted desirability actually tracks a
    test participant's *own* true (Wilson) desirability, computed
    transductively within their own held-out wave for this check only --
    never used to build the ranking itself, so it isn't leakage into the
    baseline, just a post-hoc sanity check on the baseline.
  - A random-ranking chance level for Recall@K / NDCG@K, so "beats chance"
    is a demonstrated number instead of an assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sd_features import (
    CATEGORY_LEVELS,
    INTERESTS,
    PREF,
    PROFILE_CAT,
    PROFILE_NUM,
    SEED,
    SELFRATE,
    build_person_table,
    load_raw,
    wilson_desirability as _wilson_desirability_features,
)

K = 3  # the group's decided evaluation cutoff (short candidate lists, median 16)
N_RANDOM_REPEATS = 50  # repeats per fold for the chance-level estimate

# Own-side-only profile features (no "_p" partner columns): this model predicts
# a person's OWN desirability from their OWN signup answers.
PROFILE_NUMERIC = PROFILE_NUM + INTERESTS + SELFRATE + PREF
PROFILE_CATEGORICAL = PROFILE_CAT


def wilson_desirability(raw: pd.DataFrame, iids: np.ndarray) -> pd.Series:
    """Wilson lower-bound acceptance rate RECEIVED (dec_o) for each iid, using only
    that person's own encounters. Thin wrapper around sd_features.wilson_desirability
    (the shared, single-source-of-truth implementation) shaped for this script's
    per-fold train/apply split."""
    train_df = raw.loc[raw["iid"].isin(iids)]
    out = _wilson_desirability_features(train_df, train_df)
    return out.drop_duplicates("iid").set_index("iid")["desirability_wilson"]


def make_profile_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[CATEGORY_LEVELS[c] for c in PROFILE_CATEGORICAL],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, PROFILE_NUMERIC), ("cat", categorical, PROFILE_CATEGORICAL)]
    )


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def _rank_by_score(g: pd.DataFrame, rng: np.random.Generator) -> list:
    """Sort a decider's candidates by predicted score, descending, breaking ties
    randomly (via a pre-shuffle) instead of by row order -- predicted desirability
    is compressed toward the mean (see the signal diagnostic below), so ties are
    common and an unbroken sort would silently favor whoever appears first in the
    raw data on every single tie."""
    shuffled = g.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
    return shuffled.sort_values("score", ascending=False, kind="stable")["pid"].tolist()


def run_leave_one_wave_out(raw: pd.DataFrame, k: int = K, seed: int = SEED) -> pd.DataFrame:
    person = build_person_table(raw)
    waves = sorted(raw["wave"].dropna().unique())
    rows = []
    rng = np.random.default_rng(seed)

    for w in waves:
        train_iids = raw.loc[raw["wave"] != w, "iid"].unique()
        test_iids = raw.loc[raw["wave"] == w, "iid"].unique()

        y_train = wilson_desirability(raw, train_iids).dropna()
        X_train = person.loc[person.index.isin(y_train.index)]
        y_train = y_train.loc[X_train.index]

        prep = make_profile_preprocessor()
        X_train_t = prep.fit_transform(X_train)
        model = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=seed
        )
        model.fit(X_train_t, y_train.to_numpy())

        X_test = person.loc[person.index.isin(test_iids)]
        X_test_t = prep.transform(X_test)
        pred_desirability = pd.Series(model.predict(X_test_t), index=X_test.index)

        # --- signal diagnostic (not used for ranking -- transductive, held-out-wave
        # only, purely to check whether the model differentiates people at all) ---
        y_test_true = wilson_desirability(raw, test_iids).dropna()
        common = X_test.index.intersection(y_test_true.index)
        if len(common) > 2:
            signal_corr = np.corrcoef(
                pred_desirability.loc[common], y_test_true.loc[common]
            )[0, 1]
        else:
            signal_corr = np.nan
        pred_std = pred_desirability.loc[common].std() if len(common) else np.nan
        true_std = y_test_true.loc[common].std() if len(common) else np.nan

        test_pairs = raw.loc[
            (raw["wave"] == w) & raw["pid"].notna(), ["iid", "pid", "match"]
        ].copy()
        test_pairs["pid"] = test_pairs["pid"].astype("int64")
        test_pairs["score"] = test_pairs["pid"].map(pred_desirability)

        recalls, ndcgs, topk_union = [], [], set()
        random_recalls, random_ndcgs = [], []
        for _, g in test_pairs.groupby("iid"):
            ranked = _rank_by_score(g, rng)
            relevant = set(g.loc[g["match"] == 1, "pid"])

            r, n = recall_at_k(ranked, relevant, k), ndcg_at_k(ranked, relevant, k)
            if not np.isnan(r):
                recalls.append(r)
            if not np.isnan(n):
                ndcgs.append(n)
            topk_union.update(ranked[:k])

            # chance level: same candidates, fully random order, averaged over
            # many repeats so one lucky/unlucky shuffle doesn't skew the fold
            if relevant:
                pids = g["pid"].tolist()
                rr, rn = [], []
                for _ in range(N_RANDOM_REPEATS):
                    perm = rng.permutation(pids).tolist()
                    rr.append(recall_at_k(perm, relevant, k))
                    rn.append(ndcg_at_k(perm, relevant, k))
                random_recalls.append(np.mean(rr))
                random_ndcgs.append(np.mean(rn))

        coverage = len(topk_union) / test_pairs["pid"].nunique()
        rows.append(
            {
                "wave": w,
                "n_test_participants": len(test_iids),
                f"recall@{k}": np.mean(recalls) if recalls else np.nan,
                f"ndcg@{k}": np.mean(ndcgs) if ndcgs else np.nan,
                "coverage": coverage,
                f"chance_recall@{k}": np.mean(random_recalls) if random_recalls else np.nan,
                f"chance_ndcg@{k}": np.mean(random_ndcgs) if random_ndcgs else np.nan,
                "signal_corr": signal_corr,
                "pred_std": pred_std,
                "true_std": true_std,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = load_raw()
    results = run_leave_one_wave_out(raw)
    results.to_csv("baseline_results_by_wave.csv", index=False)

    print(results.to_string(index=False))
    print()

    metric_cols = [f"recall@{K}", f"ndcg@{K}", "coverage", f"chance_recall@{K}", f"chance_ndcg@{K}"]
    print("=== Summary across 21 folds (mean +/- std) ===")
    print(results[metric_cols].agg(["mean", "std"]).to_string())
    print()

    print("=== Signal diagnostic (does predicted desirability track the truth?) ===")
    print(f"Mean corr(predicted, true Wilson desirability): "
          f"{results['signal_corr'].mean():.3f} (std {results['signal_corr'].std():.3f})")
    print(f"Mean predicted-score std dev per fold: {results['pred_std'].mean():.4f}")
    print(f"Mean true-desirability std dev per fold: {results['true_std'].mean():.4f}")
    print(f"Ratio (predicted spread / true spread): "
          f"{results['pred_std'].mean() / results['true_std'].mean():.3f}")
    print()

    print("=== Does the model beat chance? (paired across the 21 folds) ===")
    from scipy import stats

    for metric, chance_col in [(f"recall@{K}", f"chance_recall@{K}"), (f"ndcg@{K}", f"chance_ndcg@{K}")]:
        diff = results[metric] - results[chance_col]
        t_stat, p_t = stats.ttest_rel(results[metric], results[chance_col])
        _, p_w = stats.wilcoxon(results[metric], results[chance_col])
        print(
            f"{metric}: model={results[metric].mean():.3f} vs chance={results[chance_col].mean():.3f}, "
            f"mean paired diff={diff.mean():.3f}, beats chance in {(diff > 0).sum()}/{len(diff)} folds, "
            f"paired t-test p={p_t:.4f}, wilcoxon p={p_w:.4f}"
        )
