"""Ranking evaluation for the reciprocal recommender: Recall@3, NDCG@3, Coverage@3.

This is the metric set actually implemented for the baseline (`baseline_desirability.py`,
`03_Baseline_Desirability_Model.ipynb`) -- `report.qmd`'s evaluation section is a rubric
("Recall@K and NDCG@K, and/or MRR@K... coverage or novelty"), not a fixed spec, and the
baseline is what made the concrete choice: Recall@3 + NDCG@3 + Coverage, no Precision@3 or MRR.
This module exists so every method (baseline Z, X1, X2) is scored by the exact same code, the
same way `sd_features.py` is the single source of truth for features.

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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


def _dcg(relevance: np.ndarray) -> float:
    ranks = np.arange(1, len(relevance) + 1)
    return float((relevance / np.log2(ranks + 1)).sum())


def _ranked_groups(scored: pd.DataFrame, score_col: str, seed: int):
    """One (decider, ranked-group) pair per decider, tie-broken by a seeded shuffle before a
    stable sort -- see the module docstring. Same seed each call, so repeated metric calls on
    the same `scored` frame agree with each other."""
    rng = np.random.default_rng(seed)
    for decider, group in scored.groupby("decider_iid"):
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
    """Mean Recall@k over deciders, skipping deciders with zero relevant items (as the
    baseline's `recall_at_k` does: recall needs a non-empty relevant set to divide by)."""
    values = []
    for _, ranked in _ranked_groups(scored, score_col, seed):
        relevant = set(ranked.loc[ranked[label_col] == 1, id_col])
        if not relevant:
            continue
        top = ranked[id_col].head(k)
        values.append(len(set(top) & relevant) / len(relevant))
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
    """All three metrics on one wave's (or one fold's) scored rows.

    `scored` needs one row per directed encounter with columns `decider_iid`, `id_col`
    (the candidate's own identity, `pid`), `score_col` and `label_col`.
    """
    return {
        f"Recall@{k}": recall_at_k(scored, k, score_col, label_col, id_col, seed),
        f"NDCG@{k}": ndcg_at_k(scored, k, score_col, label_col, seed),
        f"Coverage@{k}": coverage_at_k(scored, k, score_col, id_col, seed),
    }
