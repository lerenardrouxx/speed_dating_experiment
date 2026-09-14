"""X1: gender-asymmetric cluster-based reciprocal recommender.

Design (notebook 02, Section 15 hand-off): men are clustered on ``FEATURES_SELFRATE`` --
what they say they are like -- and women are clustered on ``FEATURES_PREF`` -- what they say
they want. Both are one row per person.

Those two cluster-label sets are then used to build TWO directional compatibility tables, each
indexed by (male self-rating cluster, female preference cluster):

* ``compat_female_dec`` -- among training pairs that fall in a given cell, how often did the
  WOMAN say yes to the man? This is the natural read of the cell: "does a woman who wants what
  this cluster wants say yes to a man who is what that cluster is like".
* ``compat_male_dec`` -- the same cell, but how often did the MAN say yes to the woman? A cell
  can predict either person's decision; nothing requires the predicted party to be the one the
  clustering axis was built from.
* ``compat_match`` -- how often did the pair mutually match.

Both preference-share clustering inputs are pulled down to 4 dimensions (``pref_attr``,
``pref_sinc``, ``pref_intel``, ``pref_fun``): exploration (see ``explore_clr.py``) found that
``pref_amb`` and ``pref_shar`` are zero for 12 and 14 of 274 women respectively, and including
them in a compositional-share K-Means (or its CLR-transformed equivalent) lets that zero
inflation dominate the cluster split rather than a genuine preference typology. The two zero
flags are kept as separate per-person indicator features instead, exported for the caller
alongside the cluster id, rather than folded into the K-Means distance.

Every table is fit **inside the caller's CV fold**, on training-wave rows only, exactly like
``sd_features.fit_preprocessor`` -- this module never looks at more than one wave at a time
during evaluation. Cells are shrunk toward the fold's global yes-rate with ``m`` pseudo-counts,
the same convention ``sd_features.add_behavioural_features`` uses for the same reason: most
cells have plenty of training rows (8,368 rows / 2 directions / up to 16 cells is a few hundred
each), but shrinkage costs nothing and protects the rare small cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 42

SELFRATE_COLS = ["selfrate_attr", "selfrate_sinc", "selfrate_intel", "selfrate_fun",
                 "selfrate_amb"]                                       # men, 5 dims
PREF_CLUSTER_COLS = ["pref_attr", "pref_sinc", "pref_intel", "pref_fun"]   # women, 4 dims
PREF_ZERO_FLAG_COLS = ["pref_amb", "pref_shar"]                        # women, kept as flags

K_MEN = 4
K_WOMEN = 4
SHRINK_M = 20.0


@dataclass
class X1Model:
    men_imputer: SimpleImputer
    men_scaler: StandardScaler
    men_kmeans: KMeans
    women_imputer: SimpleImputer
    women_scaler: StandardScaler
    women_kmeans: KMeans
    compat_female_dec: pd.DataFrame     # index: male_cluster, columns: female_cluster
    compat_male_dec: pd.DataFrame
    compat_match: pd.DataFrame
    global_female_yes: float
    global_male_yes: float
    global_match: float
    k_men: int
    k_women: int


def _person_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("iid").set_index("iid")


def _fit_cluster(person: pd.DataFrame, cols: list[str], k: int, seed: int = SEED):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(person[cols]))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    return imputer, scaler, km


def _assign_cluster(person: pd.DataFrame, cols: list[str], imputer, scaler, km) -> pd.Series:
    X = scaler.transform(imputer.transform(person[cols]))
    return pd.Series(km.predict(X), index=person.index, dtype="float64")


def pref_zero_flags(person: pd.DataFrame) -> pd.DataFrame:
    """Per-person indicators for a zero stated-preference share -- see module docstring."""
    return pd.DataFrame({f"{c}_zero": (person[c] == 0).astype("float64") for c in PREF_ZERO_FLAG_COLS},
                        index=person.index)


def _shrunk_table(cells: pd.DataFrame, col: str, prior: float, k_men: int, k_women: int) -> pd.DataFrame:
    agg = cells.groupby(["male_cluster", "female_cluster"])[col].agg(["sum", "count"])
    shrunk = (agg["sum"] + SHRINK_M * prior) / (agg["count"] + SHRINK_M)
    table = shrunk.unstack("female_cluster")
    table = table.reindex(index=range(k_men), columns=range(k_women))
    return table.fillna(prior)


def fit_x1(train_df: pd.DataFrame, k_men: int = K_MEN, k_women: int = K_WOMEN,
           seed: int = SEED) -> X1Model:
    """Fit both cluster models and both compatibility tables on training-wave rows only."""
    person = _person_table(train_df)
    men = person[person["gender"] == 1]
    women = person[person["gender"] == 0]

    men_imputer, men_scaler, men_km = _fit_cluster(men, SELFRATE_COLS, k_men, seed)
    women_imputer, women_scaler, women_km = _fit_cluster(women, PREF_CLUSTER_COLS, k_women, seed)

    men_cluster = _assign_cluster(men, SELFRATE_COLS, men_imputer, men_scaler, men_km)
    women_cluster = _assign_cluster(women, PREF_CLUSTER_COLS, women_imputer, women_scaler, women_km)

    is_female_decider = (train_df["gender"] == 0).to_numpy()
    male_cluster = np.where(~is_female_decider, train_df["iid"].map(men_cluster).to_numpy(),
                            train_df["pid"].map(men_cluster).to_numpy())
    female_cluster = np.where(is_female_decider, train_df["iid"].map(women_cluster).to_numpy(),
                              train_df["pid"].map(women_cluster).to_numpy())

    cells = pd.DataFrame({
        "male_cluster": male_cluster,
        "female_cluster": female_cluster,
        "is_female_decider": is_female_decider,
        "dec": train_df["dec"].to_numpy(dtype="float64"),
        "match": train_df["match"].to_numpy(dtype="float64"),
    }).dropna(subset=["male_cluster", "female_cluster"])
    cells["male_cluster"] = cells["male_cluster"].astype(int)
    cells["female_cluster"] = cells["female_cluster"].astype(int)

    global_female_yes = cells.loc[cells["is_female_decider"], "dec"].mean()
    global_male_yes = cells.loc[~cells["is_female_decider"], "dec"].mean()
    global_match = cells["match"].mean()

    compat_female_dec = _shrunk_table(cells[cells["is_female_decider"]], "dec",
                                       global_female_yes, k_men, k_women)
    compat_male_dec = _shrunk_table(cells[~cells["is_female_decider"]], "dec",
                                     global_male_yes, k_men, k_women)
    compat_match = _shrunk_table(cells, "match", global_match, k_men, k_women)

    return X1Model(men_imputer, men_scaler, men_km, women_imputer, women_scaler, women_km,
                   compat_female_dec, compat_male_dec, compat_match,
                   global_female_yes, global_male_yes, global_match, k_men, k_women)


def _lookup(table: pd.DataFrame, male_cluster: np.ndarray, female_cluster: np.ndarray,
            prior: float) -> np.ndarray:
    result = np.full(len(male_cluster), prior)
    ok = ~(np.isnan(male_cluster) | np.isnan(female_cluster))
    arr = table.to_numpy()
    result[ok] = arr[male_cluster[ok].astype(int), female_cluster[ok].astype(int)]
    return result


def predict_x1(model: X1Model, df: pd.DataFrame) -> pd.DataFrame:
    """Score every directed row of `df` (typically one held-out wave).

    Cluster assignment only needs each person's own pre-date profile -- available even for a
    cold-start participant nobody in `model` has ever met. Returns a frame aligned to `df`'s
    index with both people's cluster ids, both directional compatibility scores, the predicted
    match probability, and `x1_score`: the score relevant to THIS row's own decider (her score
    on a female-decider row, his on a male-decider row).
    """
    person = _person_table(df)
    men = person[person["gender"] == 1]
    women = person[person["gender"] == 0]

    men_cluster = _assign_cluster(men, SELFRATE_COLS, model.men_imputer, model.men_scaler,
                                   model.men_kmeans)
    women_cluster = _assign_cluster(women, PREF_CLUSTER_COLS, model.women_imputer,
                                     model.women_scaler, model.women_kmeans)

    is_female_decider = (df["gender"] == 0).to_numpy()
    male_cluster = np.where(~is_female_decider, df["iid"].map(men_cluster).to_numpy(),
                            df["pid"].map(men_cluster).to_numpy())
    female_cluster = np.where(is_female_decider, df["iid"].map(women_cluster).to_numpy(),
                              df["pid"].map(women_cluster).to_numpy())

    out = pd.DataFrame({"male_cluster": male_cluster, "female_cluster": female_cluster},
                       index=df.index)
    out["x1_female_dec"] = _lookup(model.compat_female_dec, male_cluster, female_cluster,
                                    model.global_female_yes)
    out["x1_male_dec"] = _lookup(model.compat_male_dec, male_cluster, female_cluster,
                                  model.global_male_yes)
    out["x1_match"] = _lookup(model.compat_match, male_cluster, female_cluster,
                               model.global_match)
    out["x1_score"] = np.where(is_female_decider, out["x1_female_dec"], out["x1_male_dec"])
    return out
