"""Ranking evaluation for the reciprocal recommender: Recall@3, NDCG@3, Precision@3, MRR,
Coverage@3, a chance-level reference, and paired significance tests.

This is the single shared evaluator every method (baseline Z, X1, X2/FM) is scored by, the same
way `sd_features.py` is the single source of truth for features. Before this extension, the FM
(`sd_fm.py`) carried its own separate copy of NDCG/Precision/MRR/Coverage under different column
names (`iid`/`pid`/`p_match` instead of `decider_iid`/`candidate_id`/`score`), and the baseline
had its own chance-level and paired-test code inline in `baseline_desirability.py` rather than
here -- three evaluators computing overlapping numbers, none guaranteed to agree with the others
on edge cases (tie-breaking, which deciders get skipped, whether MRR is capped at k). Everything
below is meant to replace all three call sites; nothing here changes an existing metric's
definition, it only adds the metrics/references that were previously computed elsewhere (or not
at all) and gives every caller one place to get them.

The ranking unit is one decider in one wave: everyone they met that night is their candidate
list, ordered by the model's score for that row. Relevance is `match` -- the symmetric label,
because the point of a RECIPROCAL recommender is to rank people who would actually match
highly, not just people the decider alone would say yes to.

Coverage@3 is defined exactly as the baseline defines it: the fraction of DISTINCT CANDIDATE
PEOPLE (by `candidate_id`, i.e. `pid`) who show up in *someone's* top-3 that wave, out of all
distinct candidate people available that wave. This is a people-coverage number, not a
row-coverage number -- a candidate met by several deciders only needs to be recommended once
to count.

**Tie-breaking is randomized, not row-order**, matching the baseline's own `_rank_by_score`:
predicted scores are often heavily compressed (the baseline's signal diagnostic found predicted
desirability varies only ~1/4 as much as true desirability does), and X1's cluster lookup is
more compressed still -- at most k_men x k_women distinct values, so most candidates in a
decider's list tie exactly. Breaking ties by whichever row happens to sit first in the raw data
would silently favor that row on every tie, for every method, which is a bigger effect for a
coarse method like X1 than for a continuous one. Each candidate list is shuffled (seeded, so
reproducible) before the stable sort that ranks by score.

**Which deciders get averaged over.** Recall, NDCG, Precision and MRR are all undefined (or
trivially zero in a way that would misrepresent the method) for a decider with no true `match`
in their candidate list -- there is nothing for the method to have found. Every per-decider
metric below therefore skips deciders with zero relevant candidates, consistently, so a fold's
Recall@3, NDCG@3, Precision@3 and MRR are all averaged over the *same* set of deciders and stay
comparable to each other. Coverage is a population-level quantity (which candidates ever get
surfaced), not a per-decider one, so it is unaffected by this and uses every decider.

**Two ways to average across the 21 leave-one-wave-out folds.** `evaluate()` scores one fold (one
wave) at a time, averaging over that fold's deciders -- stacking 21 calls and taking mean/std
across folds (as `baseline_desirability.py` already does) weights every WAVE equally regardless
of how many people attended it. `evaluate_per_user()` instead returns one row per decider with
that decider's own metrics, pooled across every fold at once; averaging that table's columns
weights every PERSON equally instead. The two can disagree when wave sizes vary a lot (EDA: they
do), so both are worth reporting rather than picking one silently -- `summarize_folds_and_users`
below returns both from a single `scored_all` table spanning every fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_CHANCE_REPEATS = 50  # repeats per fold for the chance-level reference


def _dcg(relevance: np.ndarray) -> float:
    ranks = np.arange(1, len(relevance) + 1)
    return float((relevance / np.log2(ranks + 1)).sum())


def _ranked_groups(scored: pd.DataFrame, score_col: str, seed: int, decider_col: str = "decider_iid"):
    """One (decider, ranked-group) pair per decider, tie-broken by a seeded shuffle before a
    stable sort -- see the module docstring. Same seed each call, so repeated metric calls on
    the same `scored` frame agree with each other."""
    rng = np.random.default_rng(seed)
    for decider, group in scored.groupby(decider_col):
        shuffled = group.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        yield decider, shuffled.sort_values(score_col, ascending=False, kind="stable")


def ndcg_at_k(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
              label_col: str = "match", seed: int = SEED) -> float:
    """Mean NDCG@k over deciders, skipping deciders with zero relevant items (NDCG undefined)."""
    values = []
    for _, ranked in _ranked_groups(scored, score_col, seed):
        rel = ranked[label_col].to_numpy(dtype="float64")
        if rel.sum() == 0:
            continue
        dcg = _dcg(rel[:k])
        ideal = _dcg(np.sort(rel)[::-1][:k])
        values.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(values)) if values else float("nan")


def recall_at_k(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
                 label_col: str = "match", id_col: str = "candidate_id",
                 seed: int = SEED) -> float:
    """Mean Recall@k over deciders, skipping deciders with zero relevant items (recall needs
    a non-empty relevant set to divide by)."""
    values = []
    for _, ranked in _ranked_groups(scored, score_col, seed):
        relevant = set(ranked.loc[ranked[label_col] == 1, id_col])
        if not relevant:
            continue
        top = ranked[id_col].head(k)
        values.append(len(set(top) & relevant) / len(relevant))
    return float(np.mean(values)) if values else float("nan")


def precision_at_k(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
                    label_col: str = "match", seed: int = SEED) -> float:
    """Mean Precision@k over deciders, skipping deciders with zero relevant items (see the
    module docstring: kept consistent with Recall/NDCG/MRR rather than counting a decider who
    could never have a hit against every method equally). The denominator is capped to the
    decider's actual candidate-list length when it is shorter than k, so a decider met by only
    1 or 2 people that night is not penalised for a list that could never reach length k."""
    values = []
    for _, ranked in _ranked_groups(scored, score_col, seed):
        rel = ranked[label_col].to_numpy(dtype="float64")
        if rel.sum() == 0:
            continue
        n = min(k, len(rel))
        values.append(float(rel[:n].sum()) / n if n > 0 else float("nan"))
    return float(np.mean(values)) if values else float("nan")


def mrr(scored: pd.DataFrame, score_col: str = "score", label_col: str = "match",
        seed: int = SEED) -> float:
    """Mean Reciprocal Rank over deciders: 1 / (rank of the first true match), over the FULL
    ranked candidate list (not cut at k) -- matching the FM module's convention, since capping
    MRR's search at k would silently turn "found on the list, just not near the top" into
    "not found at all". Deciders with zero relevant items are skipped, same as the other
    per-decider metrics above."""
    values = []
    for _, ranked in _ranked_groups(scored, score_col, seed):
        rel = ranked[label_col].to_numpy(dtype="float64")
        if rel.sum() == 0:
            continue
        hits = np.flatnonzero(rel > 0)
        values.append(1.0 / (hits[0] + 1))
    return float(np.mean(values)) if values else float("nan")


def coverage_at_k(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
                   id_col: str = "candidate_id", seed: int = SEED) -> float:
    """Fraction of distinct candidate PEOPLE recommended in *someone's* top-k this wave, out
    of all distinct candidate people available -- the baseline's exact definition."""
    recommended = set()
    for _, ranked in _ranked_groups(scored, score_col, seed):
        recommended.update(ranked[id_col].head(k))
    total = scored[id_col].nunique()
    return len(recommended) / total if total else float("nan")


def evaluate(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
             label_col: str = "match", id_col: str = "candidate_id",
             seed: int = SEED) -> dict[str, float]:
    """Recall@k, NDCG@k, Precision@k, MRR and Coverage@k on one wave's (or one fold's) scored
    rows -- the full metric set `report.qmd`'s research question and the EDA/FE hand-off name.

    `scored` needs one row per directed encounter with columns `decider_iid`, `id_col`
    (the candidate's own identity, `pid`), `score_col` and `label_col`.
    """
    return {
        f"Recall@{k}": recall_at_k(scored, k, score_col, label_col, id_col, seed),
        f"NDCG@{k}": ndcg_at_k(scored, k, score_col, label_col, seed),
        f"Precision@{k}": precision_at_k(scored, k, score_col, label_col, seed),
        "MRR": mrr(scored, score_col, label_col, seed),
        f"Coverage@{k}": coverage_at_k(scored, k, score_col, id_col, seed),
    }


# --------------------------------------------------------------------------------------------
# Per-user metrics -- pooled across folds, one row per decider (see module docstring)
# --------------------------------------------------------------------------------------------

def evaluate_per_user(scored: pd.DataFrame, k: int = 3, score_col: str = "score",
                       label_col: str = "match", id_col: str = "candidate_id",
                       decider_col: str = "decider_iid", seed: int = SEED) -> pd.DataFrame:
    """One row per decider with their own Recall@k/NDCG@k/Precision@k/reciprocal-rank.

    `scored` can span every fold at once (unlike `evaluate()`, which is meant for one fold) --
    each decider only appears in the fold that held their wave out, so grouping by decider here
    is safe and gives a person-weighted table rather than a wave-weighted one. Deciders with no
    true match are excluded, same convention as the per-fold metrics.
    """
    rows = []
    for decider, ranked in _ranked_groups(scored, score_col, seed, decider_col):
        rel = ranked[label_col].to_numpy(dtype="float64")
        if rel.sum() == 0:
            continue
        relevant = set(ranked.loc[ranked[label_col] == 1, id_col])
        top = ranked[id_col].head(k)
        n = min(k, len(rel))
        hits = np.flatnonzero(rel > 0)
        dcg = _dcg(rel[:k])
        ideal = _dcg(np.sort(rel)[::-1][:k])
        rows.append({
            decider_col: decider,
            "n_candidates": len(rel),
            f"Recall@{k}": len(set(top) & relevant) / len(relevant),
            f"NDCG@{k}": dcg / ideal if ideal > 0 else 0.0,
            f"Precision@{k}": float(rel[:n].sum()) / n if n > 0 else float("nan"),
            "MRR": 1.0 / (hits[0] + 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Chance-level reference: same candidates, fully random order
# --------------------------------------------------------------------------------------------

def evaluate_chance(scored: pd.DataFrame, k: int = 3, label_col: str = "match",
                     id_col: str = "candidate_id", decider_col: str = "decider_iid",
                     n_repeats: int = N_CHANCE_REPEATS, seed: int = SEED) -> dict[str, float]:
    """What Recall@k/NDCG@k/Precision@k/MRR/Coverage@k would be if candidates were ranked in a
    fully random order instead of by any model's score -- the same candidates and the same
    relevance labels, so this isolates "does the model's ranking beat pure chance" from "is the
    task itself easy or hard". Averaged over `n_repeats` independent random orderings per
    decider so one lucky/unlucky shuffle doesn't decide the number (mirrors the baseline's own
    chance-level diagnostic in `baseline_desirability.py`, generalised to every metric here).

    A method's real score should be compared to this, not to zero -- e.g. with a median
    candidate-list length of ~16 and k=3, even a uniformly random ranking recovers a relevant
    candidate reasonably often just from list length, so "beats zero" is a much weaker claim
    than "beats chance".
    """
    rng = np.random.default_rng(seed)
    recalls, ndcgs, precisions, mrrs = [], [], [], []
    recommended = set()

    for _, group in scored.groupby(decider_col):
        rel_all = group[label_col].to_numpy(dtype="float64")
        relevant_ids = set(group.loc[group[label_col] == 1, id_col])
        ids = group[id_col].to_numpy()

        recommended.update(ids[:k])  # coverage reference: which IDs a decider even has access to

        if not relevant_ids:
            continue
        for _ in range(n_repeats):
            perm = rng.permutation(len(rel_all))
            rel = rel_all[perm]
            ids_perm = ids[perm]

            n = min(k, len(rel))
            dcg = _dcg(rel[:k])
            ideal = _dcg(np.sort(rel)[::-1][:k])

            recalls.append(len(set(ids_perm[:k]) & relevant_ids) / len(relevant_ids))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
            precisions.append(float(rel[:n].sum()) / n if n > 0 else float("nan"))
            hits = np.flatnonzero(rel > 0)
            mrrs.append(1.0 / (hits[0] + 1))

    total = scored[id_col].nunique()
    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        f"Precision@{k}": float(np.mean(precisions)) if precisions else float("nan"),
        "MRR": float(np.mean(mrrs)) if mrrs else float("nan"),
        # Coverage under random ranking, in expectation, is just k / candidates-per-decider
        # averaged sensibly across deciders; approximated the same way as the real metric would
        # be, by actually recommending each decider's first-k-after-shuffle IDs once.
        f"Coverage@{k}": len(recommended) / total if total else float("nan"),
    }


# --------------------------------------------------------------------------------------------
# Paired significance tests across folds
# --------------------------------------------------------------------------------------------

def paired_tests(method_values, reference_values) -> dict[str, float]:
    """Paired t-test and Wilcoxon signed-rank test between two equal-length sequences of
    per-fold (or per-user) metric values -- e.g. a method's 21 per-wave NDCG@3 values against
    the chance-level or baseline NDCG@3 for those same 21 waves. Both tests are reported because
    they answer slightly different questions (mean difference vs. consistency of direction) and
    21 folds is few enough that it is worth not leaning on a single test's assumptions.

    Returns the mean paired difference, how many folds/users favoured `method_values`, and each
    test's statistic and p-value. `NaN`-valued folds (a metric that was undefined for that fold,
    e.g. no relevant items anywhere that wave) are dropped from both inputs together before
    testing, since a fold missing from one side has nothing to pair against.
    """
    a = np.asarray(method_values, dtype="float64")
    b = np.asarray(reference_values, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"paired_tests needs equal-length sequences, got {a.shape} vs {b.shape}")

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b

    result = {
        "n_pairs": int(mask.sum()),
        "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
        "n_favouring_method": int((diff > 0).sum()),
        "t_stat": float("nan"), "p_ttest": float("nan"),
        "wilcoxon_stat": float("nan"), "p_wilcoxon": float("nan"),
    }
    if len(diff) < 2 or np.allclose(diff, 0):
        return result  # not enough pairs, or no variation at all -- both tests are undefined

    t_stat, p_ttest = stats.ttest_rel(a, b)
    result["t_stat"], result["p_ttest"] = float(t_stat), float(p_ttest)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(a, b)
        result["wilcoxon_stat"], result["p_wilcoxon"] = float(w_stat), float(p_wilcoxon)
    except ValueError:
        pass  # e.g. all differences are zero after masking -- leave the wilcoxon fields as NaN
    return result


# --------------------------------------------------------------------------------------------
# Convenience: both ways of summarising across the 21 folds, from one call
# --------------------------------------------------------------------------------------------

def summarize_folds_and_users(scored_all: pd.DataFrame, k: int = 3, score_col: str = "score",
                               label_col: str = "match", id_col: str = "candidate_id",
                               decider_col: str = "decider_iid", fold_col: str = "wave",
                               seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Run one method's full leave-one-wave-out `scored_all` table (every fold's rows, already
    concatenated, with a `fold_col` identifying which wave each row's fold was) through both
    aggregation views described in the module docstring.

    Returns a dict with:
        "by_fold" -- one row per fold, mean over that fold's deciders (wave-weighted).
        "per_user" -- one row per decider, pooled across all folds (person-weighted).
        "chance_by_fold" -- the same shape as "by_fold", but for the chance-level reference,
            so the two can be paired directly with `paired_tests`.
    """
    fold_rows, chance_rows = [], []
    for fold, group in scored_all.groupby(fold_col):
        metrics = evaluate(group, k, score_col, label_col, id_col, seed)
        metrics[fold_col] = fold
        fold_rows.append(metrics)

        chance = evaluate_chance(group, k, label_col, id_col, decider_col, seed=seed)
        chance[fold_col] = fold
        chance_rows.append(chance)

    return {
        "by_fold": pd.DataFrame(fold_rows).set_index(fold_col),
        "per_user": evaluate_per_user(scored_all, k, score_col, label_col, id_col, decider_col, seed),
        "chance_by_fold": pd.DataFrame(chance_rows).set_index(fold_col),
    }
