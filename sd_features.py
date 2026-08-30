"""Feature engineering for the Speed Dating reciprocal recommender.

This module is the single source of truth for the feature table. Notebook
``02_Speed_Dating_Feature_Engineering.ipynb`` narrates and verifies what happens here; the
modelling and evaluation notebooks import from here rather than re-deriving anything.

One row of the output = one **directed encounter**: participant ``iid`` meeting partner ``pid``.
The decider's columns carry no suffix, the partner's carry ``_p``. Every meeting appears twice,
once from each side, so the mirror of ``(a, b)`` is the separate row ``(b, a)``.

Typical use in a modelling notebook::

    from sd_features import load_features, fit_preprocessor, transform, FEATURES_ALL

    feat = load_features()
    for w in sorted(feat["wave"].unique()):          # the CV loop belongs to the caller
        train, test = feat[feat["wave"] != w], feat[feat["wave"] == w]
        prep = fit_preprocessor(train)               # fitted on TRAINING waves only
        X_tr, X_te = transform(prep, train), transform(prep, test)
        y_tr, y_te = train["dec"], test["dec"]

Design decisions inherited from ``01_Speed_Dating_EDA.ipynb`` (sections in brackets):

1. Features are **strictly pre-date** [13]. The in-date scorecard, every ``_o`` event rating,
   ``dec_o`` as an input and all ``_s``/``_2``/``_3`` surveys are banned; see ``BANNED_LEAKY``.
2. Leave-one-wave-out CV is a **total cold start** [15]: 0 of 551 held-out participants have any
   training history. Person-level behavioural features are constant at test time, so they are not
   in the feature table -- see ``add_behavioural_features`` for what they are still good for.
3. ``attr1_1..shar1_1`` needs **no** wave 6-9 rescale [12]; all 21 waves already sum to ~100. We
   still normalise to shares because the per-wave totals run 90-148.
4. The compatibility gap uses the partner's ``3_1`` **self-ratings** [15.4], not their in-date
   ratings, which are unavailable for a cold-start partner. Five attributes, not six: there is no
   ``shar3_1``, so the sixth stated weight is paired with ``int_corr`` instead.
5. Clip to documented ranges before scaling [14.1], and fit every scaler on training waves only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42

RAW_PATH = Path("Data") / "Speed_Dating_Data.csv"
FEATURES_DIR = Path("Data") / "features"
FEATURES_CSV = FEATURES_DIR / "speed_dating_features.csv"
DICTIONARY_CSV = FEATURES_DIR / "feature_dictionary.csv"

# Used only if the CSV is absent (e.g. a bare Colab runtime).
FALLBACK_URL = (
    "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/"
    "Speed%20Dating%20Data.csv"
)

# --------------------------------------------------------------------------------------------
# Raw column groups, named as the data key names them
# --------------------------------------------------------------------------------------------

INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
             "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping",
             "yoga"]

ATTR6 = ["attr", "sinc", "intel", "fun", "amb", "shar"]     # the six attributes
ATTR5 = ["attr", "sinc", "intel", "fun", "amb"]             # the five that have a self-rating

PREF1_RAW = [f"{a}1_1" for a in ATTR6]      # stated preference: what I want (100-point allocation)
SELF3_RAW = [f"{a}3_1" for a in ATTR5]      # self-rating: what I am like (1-10)
PF_O_RAW = ["pf_o_att", "pf_o_sin", "pf_o_int", "pf_o_fun", "pf_o_amb", "pf_o_sha"]

PROFILE_NUM = ["age", "imprace", "imprelig", "exphappy", "go_out", "date"]
PROFILE_CAT = ["race", "field_cd", "career_c", "goal"]

# Renamed forms used in the feature table.
PREF = [f"pref_{a}" for a in ATTR6]
SELFRATE = [f"selfrate_{a}" for a in ATTR5]

# Documented code lists, from the data key. Hard-coded on purpose: deriving categories from a
# training fold would give a different column layout in every fold of a 21-fold CV loop.
CATEGORY_LEVELS = {
    "race": [1, 2, 3, 4, 5, 6],
    "field_cd": list(range(1, 19)),
    "career_c": list(range(1, 18)),
    "goal": [1, 2, 3, 4, 5, 6],
}

# Documented ranges, for clipping and for the fold-independent min-max.
RANGE_1_10 = (1.0, 10.0)
RANGE_1_7 = (1.0, 7.0)

# --------------------------------------------------------------------------------------------
# Output column groups
# --------------------------------------------------------------------------------------------

KEYS = ["iid", "pid", "wave", "gender", "order"]
LABELS = ["dec", "dec_o", "match"]

FEATURES_PROFILE = (PROFILE_NUM + [f"{c}_p" for c in PROFILE_NUM]
                    + PROFILE_CAT + [f"{c}_p" for c in PROFILE_CAT])
FEATURES_PREF = PREF + [f"{c}_p" for c in PREF]
FEATURES_SELFRATE = SELFRATE + [f"{c}_p" for c in SELFRATE]
FEATURES_INTERESTS = INTERESTS + [f"{c}_p" for c in INTERESTS]
FEATURES_DYADIC = ["age_diff", "age_gap", "samerace", "race_importance",
                   "same_field", "same_goal", "same_career", "go_out_diff", "date_diff",
                   "int_corr", "int_sim_pearson", "int_sim_cosine"]
FEATURES_COMPAT = ["compat_a2b", "compat_b2a", "compat_harmonic", "compat_gap_l1",
                   "shar_x_intcorr", "shar_x_intcorr_p"]

# `gender` is both a key and a legitimate pre-date feature (the yes-rate gap is 10.9pp, EDA 5).
FEATURES_ALL = (["gender"] + FEATURES_PROFILE + FEATURES_PREF + FEATURES_SELFRATE
                + FEATURES_INTERESTS + FEATURES_DYADIC + FEATURES_COMPAT)

# Every column a pre-date model may NOT see. Checked programmatically by `leakage_report`.
BANNED_LEAKY = (
    ATTR6 + ["like", "prob"]                                   # in-date scorecard I fill in
    + [f"{a}_o" for a in ATTR6] + ["like_o", "prob_o"]         # in-date scorecard about me
    + ["dec_o", "met", "met_o"]                                # decisions / met-before, in-date
    + ["condtn"]                                               # wave-constant designed treatment
    + ["expnum", "mn_sat", "tuition", "income"]                # pre-date but 49-77% missing
)
BANNED_SUFFIXES = ("_s", "_2", "_3")     # mid-event and follow-up surveys, in every block


# --------------------------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------------------------

def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Speed Dating CSV (8,378 x 195). Latin-1: the file is not UTF-8."""
    path = Path(path)
    source = path if path.exists() else FALLBACK_URL
    return pd.read_csv(source, encoding="latin-1")


def load_features(path: str | Path = FEATURES_CSV) -> pd.DataFrame:
    """Load the exported feature table produced by :func:`build_features`."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------------
# 2. Person table -- one row per participant
# --------------------------------------------------------------------------------------------

def build_person_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per participant (551), carrying every pre-date signup signal we use.

    Clipping happens here, once, so that every downstream feature inherits clean inputs. The
    stated-preference block is converted to shares that sum to 1 (EDA 12) and the `3_1` block is
    renamed to `selfrate_*` so nobody has to remember what `attr3_1` means.
    """
    p = raw.drop_duplicates("iid").set_index("iid").copy()

    person = pd.DataFrame(index=p.index)

    # Profile: 1-10 importance/expectation scales and the two 1-7 frequency ordinals.
    person["age"] = p["age"]
    for c in ["imprace", "imprelig", "exphappy"]:
        person[c] = p[c].clip(*RANGE_1_10)
    for c in ["go_out", "date"]:
        person[c] = p[c].clip(*RANGE_1_7)
    for c in PROFILE_CAT:
        person[c] = p[c]

    # 17 interests, 1-10. `gaming` reaches 14 and `reading` 13 in the raw file (EDA 14.1).
    for c in INTERESTS:
        person[c] = p[c].clip(*RANGE_1_10)

    # Self-ratings: the pre-date stand-in for "what this person is like".
    for a, raw_col in zip(ATTR5, SELF3_RAW):
        person[f"selfrate_{a}"] = p[raw_col].clip(*RANGE_1_10)

    # Stated preferences -> shares. Not because of the documented wave 6-9 scale problem (that
    # block is already converted, EDA 12) but because the per-wave totals run 90 to 148.
    total = p[PREF1_RAW].sum(axis=1, min_count=len(PREF1_RAW))
    for a, raw_col in zip(ATTR6, PREF1_RAW):
        person[f"pref_{a}"] = p[raw_col] / total

    return person


# --------------------------------------------------------------------------------------------
# 3. Directed pair table -- the feature matrix
# --------------------------------------------------------------------------------------------

def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    """1.0 if the two codes agree, 0.0 if they differ, NaN if either side is missing.

    NaN rather than 0 on purpose: "we do not know their field" is not "their field differs".
    """
    return (a == b).astype("float64").where(a.notna() & b.notna())


def _row_similarity(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Pearson correlation and cosine similarity between two equal-shaped matrices."""
    ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
    pearson = np.full(len(x), np.nan)
    cosine = np.full(len(x), np.nan)

    xs, ys = x[ok], y[ok]
    xc, yc = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
    denom_p = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    denom_c = np.sqrt((xs ** 2).sum(axis=1) * (ys ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson[ok] = np.where(denom_p > 0, (xc * yc).sum(axis=1) / denom_p, np.nan)
        cosine[ok] = np.where(denom_c > 0, (xs * ys).sum(axis=1) / denom_c, np.nan)
    return pearson, cosine


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the directed-encounter feature table (8,368 rows).

    Rows without a `pid` (10 of 8,378) are dropped: they name no partner, so no dyadic feature
    exists for them and they have no mirror row either.

    Every column returned is raw (unscaled, un-encoded) on purpose -- scaling belongs inside the
    caller's CV fold, via :func:`fit_preprocessor`.
    """
    person = build_person_table(raw)

    # Select the keys FIRST. `raw` already contains `age`, `race`, `field_cd`... so joining the
    # person table onto the full frame raises "columns overlap but no suffix specified".
    pairs = raw.loc[raw["pid"].notna(), KEYS + LABELS + ["samerace", "int_corr"]].copy()
    pairs["pid"] = pairs["pid"].astype("int64")

    out = (pairs
           .join(person, on="iid")
           .join(person.add_suffix("_p"), on="pid")
           .reset_index(drop=True))

    # --- dyadic ------------------------------------------------------------------------------
    out["age_diff"] = (out["age"] - out["age_p"]).abs()
    out["age_gap"] = out["age"] - out["age_p"]          # signed: direction is the point of X1
    out["race_importance"] = out["imprace"] * out["samerace"]
    out["same_field"] = _same(out["field_cd"], out["field_cd_p"])
    out["same_goal"] = _same(out["goal"], out["goal_p"])
    out["same_career"] = _same(out["career_c"], out["career_c_p"])
    # `go_out`/`date` are reverse-coded (1 = several times a week ... 7 = almost never), so the
    # absolute difference is a lifestyle-frequency MISMATCH, not "how often they go out".
    out["go_out_diff"] = (out["go_out"] - out["go_out_p"]).abs()
    out["date_diff"] = (out["date"] - out["date_p"]).abs()

    # Our own interest similarity, alongside the ready-made `int_corr`. They are close but NOT
    # the same number (r ~ 0.92): `int_corr` is precomputed and we cannot reproduce it exactly.
    mine = out[INTERESTS].to_numpy(dtype="float64")
    theirs = out[[f"{c}_p" for c in INTERESTS]].to_numpy(dtype="float64")
    out["int_sim_pearson"], out["int_sim_cosine"] = _row_similarity(mine, theirs)

    # --- compatibility, both directions, cold-start-safe ---------------------------------------
    # My stated weights over the five attributes that have a self-rating, renormalised to sum to
    # 1, dotted with the partner's self-ratings. Both sides come from the signup survey, so both
    # exist for a participant nobody has ever met (EDA 15.4).
    w_self = out[[f"pref_{a}" for a in ATTR5]]
    w_self = w_self.div(w_self.sum(axis=1), axis=0)
    w_part = out[[f"pref_{a}_p" for a in ATTR5]]
    w_part = w_part.div(w_part.sum(axis=1), axis=0)

    attr_self = out[[f"selfrate_{a}" for a in ATTR5]]
    attr_part = out[[f"selfrate_{a}_p" for a in ATTR5]]

    out["compat_a2b"] = (w_self.to_numpy() * attr_part.to_numpy()).sum(axis=1)
    out["compat_b2a"] = (w_part.to_numpy() * attr_self.to_numpy()).sum(axis=1)
    denom = out["compat_a2b"] + out["compat_b2a"]
    out["compat_harmonic"] = np.where(denom > 0, 2 * out["compat_a2b"] * out["compat_b2a"] / denom,
                                      np.nan)

    # Distance form of the same idea: how far my weight profile is from what they actually are.
    # The partner's self-ratings are turned into shares first so both vectors sum to 1.
    part_share = attr_part.div(attr_part.sum(axis=1), axis=0)
    out["compat_gap_l1"] = np.abs(w_self.to_numpy() - part_share.to_numpy()).sum(axis=1)

    # The sixth stated weight has no `shar3_1` to meet -- nobody self-rates "shared interests",
    # which is a property of a pair. Pair it with the pre-date shared-interest proxy instead.
    out["shar_x_intcorr"] = out["pref_shar"] * out["int_corr"]
    out["shar_x_intcorr_p"] = out["pref_shar_p"] * out["int_corr"]

    # Rows where an input was missing produce NaN throughout; nothing is silently filled here.
    for col in ["compat_a2b", "compat_b2a", "compat_harmonic", "compat_gap_l1"]:
        out[col] = out[col].where(out[col].notna())

    ordered = KEYS + LABELS + [c for c in FEATURES_ALL if c not in KEYS]
    return out[ordered]


# --------------------------------------------------------------------------------------------
# 4. Leakage guard
# --------------------------------------------------------------------------------------------

def leakage_report(features: list[str] | None = None) -> pd.DataFrame:
    """Check the feature list against `BANNED_LEAKY`. Returns one row per violation (ideally 0)."""
    features = FEATURES_ALL if features is None else features
    hits = [
        {"column": c,
         "reason": ("in-date or label-adjacent" if c not in
                    ("condtn", "expnum", "mn_sat", "tuition", "income")
                    else "excluded by decision (wave-constant or >49% missing)")}
        for c in features
        if c in set(BANNED_LEAKY) or c.endswith(BANNED_SUFFIXES)
    ]
    return pd.DataFrame(hits, columns=["column", "reason"])


# --------------------------------------------------------------------------------------------
# 5. Fold-aware preprocessing
# --------------------------------------------------------------------------------------------

# Fold-independent min-max onto the DOCUMENTED range (not the observed one): a value then means
# the same thing in every fold, and no statistic is learned from the data, so nothing can leak.
_MINMAX_1_10 = (["imprace", "imprelig", "exphappy"] + INTERESTS + SELFRATE)
_MINMAX_1_10 = _MINMAX_1_10 + [f"{c}_p" for c in _MINMAX_1_10]
_MINMAX_1_7 = ["go_out", "date"]
_MINMAX_1_7 = _MINMAX_1_7 + [f"{c}_p" for c in _MINMAX_1_7]

# Unbounded or non-obviously-bounded: standardised, and THIS is fitted on the training waves.
_STANDARDISE = (["age", "age_p", "age_diff", "age_gap", "go_out_diff", "date_diff",
                 "int_sim_pearson", "int_sim_cosine",
                 "compat_a2b", "compat_b2a", "compat_harmonic", "compat_gap_l1",
                 "shar_x_intcorr", "shar_x_intcorr_p"])

# Already on a comparable scale: preference shares sum to 1, indicators are 0/1, int_corr is a
# correlation. Passed through untouched (imputation aside).
_PASSTHROUGH = (["gender", "samerace", "same_field", "same_goal", "same_career", "int_corr"]
                + FEATURES_PREF)

_ONEHOT = PROFILE_CAT + [f"{c}_p" for c in PROFILE_CAT]


def _scale_1_10(x):
    return (x - 1.0) / 9.0


def _scale_1_7(x):
    return (x - 1.0) / 6.0


def fit_preprocessor(train_df: pd.DataFrame, impute: bool = True) -> ColumnTransformer:
    """Fit the preprocessing pipeline **on training waves only** and return it.

    Call this inside your CV loop with the training folds, then apply the returned transformer to
    both train and test. Fitting it on the full table would let the held-out fold's extremes and
    medians influence the scaling -- the same leak as fitting a model on test data, in a quieter
    form (EDA 14.1).

    Parameters
    ----------
    train_df : the training rows of the feature table.
    impute : median-impute numeric gaps (default). Pass ``False`` for models that handle NaN
        natively, such as ``HistGradientBoostingClassifier``.

    What each block gets: documented-range min-max (fold-independent), z-scoring (fitted here),
    passthrough for the already-comparable columns, and one-hot with fixed category lists so the
    output layout is identical in every fold.
    """
    def numeric(*steps):
        pipe = []
        if impute:
            pipe.append(("impute", SimpleImputer(strategy="median")))
        pipe.extend(steps)
        return Pipeline(pipe) if pipe else "passthrough"

    prep = ColumnTransformer(
        transformers=[
            ("minmax_1_10", numeric(("scale", FunctionTransformer(
                _scale_1_10, feature_names_out="one-to-one"))), _MINMAX_1_10),
            ("minmax_1_7", numeric(("scale", FunctionTransformer(
                _scale_1_7, feature_names_out="one-to-one"))), _MINMAX_1_7),
            ("standardise", numeric(("scale", StandardScaler())), _STANDARDISE),
            ("passthrough", numeric(), _PASSTHROUGH),
            ("onehot", OneHotEncoder(
                categories=[CATEGORY_LEVELS[c.removesuffix("_p")] for c in _ONEHOT],
                handle_unknown="ignore", sparse_output=False), _ONEHOT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return prep.fit(train_df)


def transform(prep: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor and keep the column names (sklearn returns a bare array)."""
    return pd.DataFrame(prep.transform(df), columns=prep.get_feature_names_out(), index=df.index)


# --------------------------------------------------------------------------------------------
# 6. Behavioural statistics -- NOT features under this protocol
# --------------------------------------------------------------------------------------------

def add_behavioural_features(train_df: pd.DataFrame, apply_df: pd.DataFrame,
                             m: float = 20.0) -> pd.DataFrame:
    """Estimate per-person behavioural rates on `train_df` and map them onto `apply_df`.

    **These are deliberately not in the feature table.** Under leave-one-wave-out CV every
    participant attends exactly one wave, so a held-out person has no training history at all: 0
    of 551, in all 21 folds (EDA 15). Every test row therefore receives the cold-start fallback,
    and a constant cannot reorder a ranked list.

    The function survives for two jobs that do need it:

    * the training-side ``desirability`` target for the re-specified popularity baseline Z
      ("predicted desirability from profile features", EDA 15.1);
    * a deliberately-labelled optimistic within-fold reference.

    Note it takes **raw** date rows, not the feature table: ``avg_attr_received`` and
    ``avg_like_received`` come from ``attr_o``/``like_o``, columns the feature table excludes as
    in-date leakage.

    Rates are shrunk toward the training mean with ``m`` pseudo-counts. m = 20 is the interior
    optimum of split-half reliability (0.545 raw -> 0.608, EDA 13.3); a third of participants have
    10 dates or fewer, where one flipped decision moves the raw rate by 10 points.
    """
    out = apply_df.copy()

    specs = {
        "selectivity": ("iid", "dec"),              # how often this person says yes
        "desirability": ("iid", "dec_o"),           # how often this person is said yes to
        "avg_attr_received": ("iid", "attr_o"),
        "avg_like_received": ("iid", "like_o"),
    }

    for name, (key, col) in specs.items():
        if col not in train_df.columns:
            continue
        prior = train_df[col].mean()
        agg = train_df.groupby(key)[col].agg(["sum", "count"])
        shrunk = (agg["sum"] + m * prior) / (agg["count"] + m)

        out[name] = out["iid"].map(shrunk).fillna(prior)          # my own rate
        out[f"{name}_p"] = out["pid"].map(shrunk).fillna(prior)   # the partner's rate

    # Honest bookkeeping: which rows got a constant because the person is unseen in training.
    seen = set(train_df["iid"].unique())
    out["behav_cold_start"] = ~out["iid"].isin(seen)
    out["behav_cold_start_p"] = ~out["pid"].isin(seen)
    return out


# --------------------------------------------------------------------------------------------
# 7. Data dictionary
# --------------------------------------------------------------------------------------------

_ATTR_WORDS = {"attr": "attractiveness", "sinc": "sincerity", "intel": "intelligence",
               "fun": "being fun", "amb": "ambition", "shar": "shared interests"}

_INTEREST_NOTE = "Self-reported interest in the activity at signup"


def _definitions() -> dict[str, tuple[str, str, str, str]]:
    """column -> (block, definition, source columns, documented range)."""
    d: dict[str, tuple[str, str, str, str]] = {
        "iid": ("key", "Participant id, unique across the study", "iid", "-"),
        "pid": ("key", "Partner's iid for this date -- who they met", "pid", "-"),
        "wave": ("key", "Session 1-21; the unit the CV splits on", "wave", "1-21"),
        "gender": ("key / feature", "0 = female, 1 = male (also a feature: the yes-rate gap is "
                                    "10.9pp)", "gender", "0/1"),
        "order": ("key / diagnostic", "Position of this date in the night. Pre-date but a "
                                      "scheduling artefact (|r| = 0.03); excluded from "
                                      "FEATURES_ALL", "order", "1-22"),
        "dec": ("label", "This participant's yes/no about the partner -- the DIRECTIONAL "
                         "training target", "dec", "0/1"),
        "dec_o": ("label", "The partner's yes/no about this participant, i.e. the reverse "
                           "direction's label. LABEL ONLY, never a feature", "dec_o", "0/1"),
        "match": ("label", "1 iff both said yes -- the symmetric relevance label", "match", "0/1"),
        "age_diff": ("dyadic", "Absolute age difference", "age, age_p", "0+"),
        "age_gap": ("dyadic", "Signed age difference (decider minus partner)", "age, age_p", "+/-"),
        "samerace": ("dyadic", "1 if both are the same race", "samerace", "0/1"),
        "race_importance": ("dyadic", "imprace x samerace: same-race matters only to people who "
                                      "say it does. No religion analogue exists (no religion "
                                      "identity column in the file)", "imprace, samerace", "0-10"),
        "same_field": ("dyadic", "1 if both study the same field, NaN if either is missing",
                       "field_cd, field_cd_p", "0/1"),
        "same_goal": ("dyadic", "1 if both attend for the same reason", "goal, goal_p", "0/1"),
        "same_career": ("dyadic", "1 if both intend the same career", "career_c, career_c_p",
                        "0/1"),
        "go_out_diff": ("dyadic", "Going-out frequency mismatch (the scale is reverse-coded: "
                                  "1 = several times a week, 7 = almost never)",
                        "go_out, go_out_p", "0-6"),
        "date_diff": ("dyadic", "Dating frequency mismatch, same reverse coding",
                      "date, date_p", "0-6"),
        "int_corr": ("dyadic", "Precomputed interest correlation for the pair (ships with the "
                               "file)", "int_corr", "-1 to 1"),
        "int_sim_pearson": ("dyadic", "Pearson correlation over the 17 interests, recomputed by "
                                      "us. Close to int_corr but NOT identical (r ~ 0.92)",
                            "17 interests, both sides", "-1 to 1"),
        "int_sim_cosine": ("dyadic", "Cosine similarity over the 17 interests",
                           "17 interests, both sides", "-1 to 1"),
        "compat_a2b": ("compatibility", "My stated weights (5 attributes, renormalised) dotted "
                                        "with the partner's SELF-ratings. Pre-date on both sides",
                       "pref_*, selfrate_*_p", "1-10"),
        "compat_b2a": ("compatibility", "The partner's stated weights dotted with my "
                                        "self-ratings -- the reciprocal direction",
                       "pref_*_p, selfrate_*", "1-10"),
        "compat_harmonic": ("compatibility", "Harmonic mean of the two directions: a ready-made "
                                             "symmetric reciprocal score",
                            "compat_a2b, compat_b2a", "1-10"),
        "compat_gap_l1": ("compatibility", "L1 distance between my weight vector and the "
                                           "partner's self-rating profile as shares -- the "
                                           "'distance' form of the same idea",
                          "pref_*, selfrate_*_p", "0-2"),
        "shar_x_intcorr": ("compatibility", "My shared-interest weight x int_corr. The sixth "
                                            "stated weight has no self-rating to meet (there is "
                                            "no shar3_1), so it pairs with the interest proxy",
                           "pref_shar, int_corr", "-1 to 1"),
        "shar_x_intcorr_p": ("compatibility", "The partner's shared-interest weight x int_corr",
                             "pref_shar_p, int_corr", "-1 to 1"),
    }

    profile_defs = {
        "age": ("Age in years", "age", "18-55"),
        "imprace": ("How important a shared racial background is (1-10)", "imprace", "1-10"),
        "imprelig": ("How important a shared religious background is (1-10)", "imprelig", "1-10"),
        "exphappy": ("How happy they expect to be with the people they meet (1-10)", "exphappy",
                     "1-10"),
        "go_out": ("How often they go out generally (1 = several/week ... 7 = almost never)",
                   "go_out", "1-7"),
        "date": ("How often they go on dates, same reverse coding", "date", "1-7"),
        "race": ("Race code 1-6 (categorical -- one-hot, never scaled)", "race", "1-6"),
        "field_cd": ("Field of study, coded 1-18 (categorical)", "field_cd", "1-18"),
        "career_c": ("Intended career, coded 1-17 (categorical)", "career_c", "1-17"),
        "goal": ("Reason for attending, coded 1-6 (categorical)", "goal", "1-6"),
    }
    for col, (definition, source, rng) in profile_defs.items():
        d[col] = ("profile", definition, source, rng)

    for a in ATTR6:
        d[f"pref_{a}"] = ("stated preference",
                          f"Share of the 100-point allocation this person gave to "
                          f"{_ATTR_WORDS[a]} at signup; the six shares sum to 1",
                          f"{a}1_1", "0-1")
    for a in ATTR5:
        d[f"selfrate_{a}"] = ("self-rating",
                              f"How this person rates their own {_ATTR_WORDS[a]} at signup. The "
                              f"pre-date stand-in for the in-date attribute ratings",
                              f"{a}3_1", "1-10")
    for c in INTERESTS:
        d[c] = ("interest", f"{_INTEREST_NOTE}: {c}", c, "1-10")

    # The partner's side of every person-level column.
    for col in list(d):
        if col in KEYS + LABELS or col in FEATURES_DYADIC + FEATURES_COMPAT:
            continue
        block, definition, source, rng = d[col]
        d[f"{col}_p"] = (block, f"Partner's value -- {definition[0].lower()}{definition[1:]}",
                         f"{source} joined on pid", rng)
    return d


def build_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    """Data dictionary for the exported table: one row per column, with definition and coverage."""
    defs = _definitions()
    missing = [c for c in features.columns if c not in defs]
    if missing:
        raise ValueError(f"No dictionary entry for: {missing}")

    rows = []
    for col in features.columns:
        block, definition, source, rng = defs[col]
        rows.append({
            "column": col,
            "block": block,
            "definition": definition,
            "source_columns": source,
            "documented_range": rng,
            "nan_fraction": round(float(features[col].isna().mean()), 4),
            "in_features_all": col in set(FEATURES_ALL),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 8. Export
# --------------------------------------------------------------------------------------------

def export(features: pd.DataFrame, dictionary: pd.DataFrame,
           out_dir: str | Path = FEATURES_DIR) -> list[Path]:
    """Write the feature table (CSV, plus parquet when pyarrow is available) and the dictionary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    csv_path = out_dir / FEATURES_CSV.name
    features.to_csv(csv_path, index=False)
    written.append(csv_path)

    try:
        parquet_path = out_dir / (FEATURES_CSV.stem + ".parquet")
        features.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    except (ImportError, ValueError):
        pass          # pyarrow absent: the CSV is the committed artefact anyway

    dict_path = out_dir / DICTIONARY_CSV.name
    dictionary.to_csv(dict_path, index=False)
    written.append(dict_path)
    return written
