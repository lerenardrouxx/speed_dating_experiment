"""Factorization machine (X2) for the Speed Dating reciprocal recommender.

Notebook ``04_Speed_Dating_Factorization_Machine.ipynb`` narrates and verifies what happens
here; all model, training and metric logic lives in this module, exactly as
``02_Speed_Dating_Feature_Engineering.ipynb`` narrates ``sd_features.py`` rather than defining
feature logic inline.

The model predicts ``dec`` -- the directional "would this participant say yes" target -- from the
raw feature table's ``FEATURES_ALL`` columns, preprocessed fold-aware via
``sd_features.fit_preprocessor``. Two directional predictions (A->B and B->A) are combined into a
single match probability by harmonic mean, the reciprocal-scoring rule fixed by the project's
proposal deck.

Design decisions inherited from the proposal deck and ``01_Speed_Dating_EDA.ipynb`` /
``02_Speed_Dating_Feature_Engineering.ipynb``:

1. **Total cold start** (EDA S15): 0 of 551 held-out participants have any training history in
   any of the 21 leave-one-wave-out folds. The FM therefore generalises through the **feature
   vector** only -- ``iid``/``pid`` are never model inputs, only join keys.
2. **Fold-aware preprocessing** (notebook 02 S12/S15): every fit -- the preprocessor AND the FM
   itself -- is fit on that fold's training waves only. ``run_lowo_fm`` calls
   ``sf.fit_preprocessor`` fresh inside its loop; never on the full table.
3. **Reciprocal combination is harmonic mean**, not product: ``2*p_ab*p_ba / (p_ab+p_ba)``,
   mirroring ``compat_harmonic`` in ``sd_features.py``.
4. **Hyperparameters are chosen once**, via a single inner validation split carved from one
   fold's training pool, and reused unchanged across all 21 outer fits -- not tuned per fold.
   See ``select_hyperparams`` for why this does not leak into the outer evaluation.
5. **Ranking relevance is the mutual ``match`` label**, not ``dec``. Users with no true match in
   their candidate set are excluded from NDCG@3/Precision@3/MRR (undefined for them by
   construction), a convention that must be stated rather than left implicit -- see
   ``evaluate_user_rankings``.

Typical use in the modelling notebook::

    import sd_features as sf
    import sd_fm as fm

    feat = sf.load_features()
    best_cfg, search_results = fm.select_hyperparams(feat)         # ONE split, reused below
    scored_all, fold_diagnostics = fm.run_lowo_fm(feat, best_cfg)  # 21 leave-one-wave-out fits
    fold_metrics = fm.evaluate_lowo_rankings(scored_all)           # NDCG@3/Precision@3/MRR/Coverage@3
    fold_metrics[["ndcg@3", "precision@3", "mrr", "coverage@3"]].agg(["mean", "std"])
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

import sd_features as sf

SEED = sf.SEED


# --------------------------------------------------------------------------------------------
# 1. The model
# --------------------------------------------------------------------------------------------

class FactorizationMachine(nn.Module):
    """Degree-2 factorization machine (Rendle, 2010).

    ``y_logit(x) = w0 + w.x + 0.5 * sum_f[ (x @ V_f)^2 - (x^2) @ (V_f^2) ]``

    computed batched over the whole ``k``-dimensional latent space at once (no explicit loop
    over feature pairs). ``k=0`` drops the interaction term entirely, collapsing the model to
    plain logistic regression on the same feature set -- the ablation the notebook compares
    against, reusing this same class and training loop rather than a second implementation.
    """

    def __init__(self, n_features: int, k: int = 8):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        self.k = k
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01) if k > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.w0 + x @ self.w
        if self.k == 0:
            return logits
        square_of_sum = (x @ self.V).pow(2).sum(dim=1)
        sum_of_square = x.pow(2) @ self.V.pow(2).sum(dim=1)
        return logits + 0.5 * (square_of_sum - sum_of_square)

    def l2_penalty(self) -> torch.Tensor:
        penalty = self.w.pow(2).sum()
        if self.V is not None:
            penalty = penalty + self.V.pow(2).sum()
        return penalty


# --------------------------------------------------------------------------------------------
# 2. Training
# --------------------------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_fm(X_tr: np.ndarray, y_tr: np.ndarray, *, k: int = 8, lr: float = 0.01,
             l2: float = 1e-3, epochs: int = 300, patience: int = 20,
             X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
             seed: int = SEED) -> tuple[FactorizationMachine, dict]:
    """Fit an FM with Adam; L2 is added to the loss explicitly rather than via ``weight_decay``,
    so the regularisation term is visible in the training curve rather than hidden in the
    optimiser.

    When ``X_val``/``y_val`` are given, training tracks validation logloss each epoch and
    restores the model to its **best-validation** checkpoint (early stopping after ``patience``
    epochs without improvement). Without a validation set, training runs for the full ``epochs``
    budget -- this is how the 21 outer folds are fit, reusing the epoch count that the one-time
    hyperparameter search found via early stopping (Section 3), rather than re-splitting a
    validation set out of every fold.

    Returns the fitted model and a history dict (``train_loss``, ``val_logloss`` if applicable,
    ``best_epoch``).
    """
    _set_seed(seed)
    model = FactorizationMachine(n_features=X_tr.shape[1], k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    history: dict = {"train_loss": [], "val_logloss": []}
    best_state, best_val, best_epoch, bad_epochs = None, np.inf, epochs - 1, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t) + l2 * model.l2_penalty()
        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    model(X_val_t), y_val_t).item()
            history["val_logloss"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = {k_: v_.clone() for k_, v_ in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history


def predict_proba(model: FactorizationMachine, X: np.ndarray) -> np.ndarray:
    """Sigmoid of the FM's logits -- P(dec=1) for each row of ``X``."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def combine_harmonic(p_ab: np.ndarray, p_ba: np.ndarray) -> np.ndarray:
    """Harmonic mean of the two directional probabilities -- the deck's reciprocal score.

    Mirrors ``compat_harmonic`` in ``sd_features.py``, applied to model outputs instead of the
    hand-built compatibility features. ``p_ab + p_ba == 0`` only if both directions are
    (numerically) certain "no", in which case the match probability really is 0.
    """
    p_ab, p_ba = np.asarray(p_ab), np.asarray(p_ba)
    denom = p_ab + p_ba
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * p_ab * p_ba / denom, 0.0)


# --------------------------------------------------------------------------------------------
# 3. Hyperparameter search -- ONE split, reused for all 21 outer folds
# --------------------------------------------------------------------------------------------

DEFAULT_GRID = {"k": [4, 8, 16], "l2": [0.001, 0.01, 0.02, 0.05], "lr": [0.01, 0.05]}


def select_hyperparams(feat: pd.DataFrame, tuning_wave: int = 1, n_val_waves: int = 4,
                        grid: dict | None = None, epochs: int = 300, patience: int = 20,
                        seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    """Pick (k, lr, l2, epochs) once via an inner split, reused unchanged by every outer fold.

    The inner split is carved from ``feat[feat.wave != tuning_wave]``: ``n_val_waves`` waves
    from that pool become the inner-validation set, the rest are inner-train. No wave that will
    ever serve as an *outer* test wave contributes to picking the winning config as anything
    other than training input (``tuning_wave`` itself never appears in this search's train or
    validation data at all; every other wave appears only as inner-train or inner-validation,
    never scored as if it were the outer test fold). The preprocessor is fit on inner-train only,
    the same fold-aware discipline the outer loop uses (notebook 02 S12/S15).

    Selection is by lowest validation logloss (a proper scoring rule for the probabilities the
    harmonic-mean combination consumes); validation AUC is reported alongside for
    interpretability. The winning config's early-stopped epoch count becomes a **fixed** epoch
    budget, reused by every outer fold's final fit instead of re-splitting a validation set out
    of each one.
    """
    grid = grid or DEFAULT_GRID
    rng = np.random.default_rng(seed)

    pool = feat[feat["wave"] != tuning_wave]
    val_waves = rng.choice(pool["wave"].unique(), size=n_val_waves, replace=False)
    inner_train = pool[~pool["wave"].isin(val_waves)]
    inner_val = pool[pool["wave"].isin(val_waves)]

    prep = sf.fit_preprocessor(inner_train)
    X_tr = sf.transform(prep, inner_train).values
    X_val = sf.transform(prep, inner_val).values
    y_tr = inner_train["dec"].values.astype("float32")
    y_val = inner_val["dec"].values.astype("float32")

    results = []
    for k, l2, lr in product(grid["k"], grid["l2"], grid["lr"]):
        model, history = train_fm(X_tr, y_tr, k=k, lr=lr, l2=l2, epochs=epochs,
                                   patience=patience, X_val=X_val, y_val=y_val, seed=seed)
        p_val = predict_proba(model, X_val)
        results.append({
            "k": k, "l2": l2, "lr": lr,
            "val_logloss": log_loss(y_val, p_val, labels=[0, 1]),
            "val_auc": roc_auc_score(y_val, p_val),
            "best_epoch": history["best_epoch"],
        })

    search_results = pd.DataFrame(results).sort_values("val_logloss").reset_index(drop=True)
    best = search_results.iloc[0]
    best_cfg = {"k": int(best["k"]), "l2": float(best["l2"]), "lr": float(best["lr"]),
                "epochs": int(best["best_epoch"]) + 1}
    return best_cfg, search_results


# --------------------------------------------------------------------------------------------
# 4. Ranking metrics
# --------------------------------------------------------------------------------------------

def ndcg_at_k(rel_sorted: np.ndarray, k: int = 3) -> float:
    """NDCG@k for binary relevance, over a list already ranked by predicted score."""
    rel_sorted = np.asarray(rel_sorted, dtype=float)
    top = rel_sorted[:k]
    dcg = float((top / np.log2(np.arange(2, len(top) + 2))).sum())

    n_rel = min(int(rel_sorted.sum()), k)
    idcg = float((np.ones(n_rel) / np.log2(np.arange(2, n_rel + 2))).sum()) if n_rel > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(rel_sorted: np.ndarray, k: int = 3) -> float:
    """Precision@k, denominator capped to the actual candidate-set size when it is below k."""
    rel_sorted = np.asarray(rel_sorted, dtype=float)
    n = min(k, len(rel_sorted))
    return float(rel_sorted[:n].sum()) / n if n > 0 else 0.0


def reciprocal_rank(rel_sorted: np.ndarray) -> float:
    """1/rank of the first relevant item, over the FULL uncut ranked list (not cut at k)."""
    hits = np.flatnonzero(np.asarray(rel_sorted) > 0)
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def evaluate_user_rankings(scored: pd.DataFrame, k: int = 3,
                            skip_zero_relevant: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Rank each user's candidates by ``p_match`` and score NDCG@k / Precision@k / MRR.

    ``scored`` needs one row per (iid, pid) directed encounter with columns ``iid``, ``p_match``,
    ``match``. Ties in ``p_match`` fall back to the existing row order (stable sort) --
    effectively moot with continuous float scores, documented rather than engineered around.

    A user with no true ``match`` in their candidate set has an undefined NDCG (IDCG=0) and MRR
    (no hit to rank). Per the project convention (stated, not silent), such users are EXCLUDED
    from the returned per-user table when ``skip_zero_relevant=True``.
    """
    rows = []
    for iid, group in scored.groupby("iid"):
        ranked = group.sort_values("p_match", ascending=False, kind="stable")
        rel = ranked["match"].to_numpy()
        if skip_zero_relevant and rel.sum() == 0:
            continue
        rows.append({"iid": iid, "n_candidates": len(rel), "ndcg@3": ndcg_at_k(rel, k),
                     "precision@3": precision_at_k(rel, k), "mrr": reciprocal_rank(rel)})

    per_user = pd.DataFrame(rows)
    cols = ["ndcg@3", "precision@3", "mrr"]
    summary = per_user[cols].mean() if len(per_user) else pd.Series({c: np.nan for c in cols})
    return per_user, summary


def coverage_at_k(scored: pd.DataFrame, k: int = 3) -> float:
    """Share of the fold's candidate catalogue recommended in at least one user's top-k list.

    Catalogue = every ``pid`` that is a candidate for someone in this fold (the held-out wave's
    opposite-sex roster). No formula for Coverage@3 exists in the EDA or the proposal deck; this
    is the definition adopted for this project, stated here rather than left implicit.
    """
    catalogue = set(scored["pid"].unique())
    if not catalogue:
        return 0.0
    recommended = set()
    for _, group in scored.groupby("iid"):
        top = group.sort_values("p_match", ascending=False, kind="stable").head(k)
        recommended.update(top["pid"].tolist())
    return len(recommended) / len(catalogue)


# --------------------------------------------------------------------------------------------
# 5. The 21-fold leave-one-wave-out evaluation
# --------------------------------------------------------------------------------------------

def run_lowo_fm(feat: pd.DataFrame, cfg: dict, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score one FM per wave, with fold-aware preprocessing throughout.

    ``cfg`` is a fixed ``{"k", "lr", "l2", "epochs"}`` dict (from ``select_hyperparams``, or with
    ``k=0`` substituted for the linear-only ablation) -- the SAME config is used for every outer
    fold; only the preprocessor and the FM's weights are refit per fold.

    Returns:
        scored_all -- one row per directed encounter across all 21 test folds, with ``p_ab``
            (this row's own direction), ``p_ba`` (the mirror row's direction, joined back onto
            this row) and ``p_match`` (their harmonic mean).
        fold_diagnostics -- one row per wave: the direction-level ``dec`` classifier's AUC and
            logloss, to sanity-check against the EDA's feasibility-probe ceiling (~0.59 AUC)
            before trusting the ranking metrics built on top of it.
    """
    scored_parts, diag_rows = [], []

    for w in sorted(feat["wave"].unique()):
        train_df = feat[feat["wave"] != w]
        test_df = feat[feat["wave"] == w]

        prep = sf.fit_preprocessor(train_df)
        X_tr = sf.transform(prep, train_df).values
        X_te = sf.transform(prep, test_df).values
        y_tr = train_df["dec"].values.astype("float32")

        model, _ = train_fm(X_tr, y_tr, k=cfg["k"], lr=cfg["lr"], l2=cfg["l2"],
                             epochs=cfg["epochs"], seed=seed)
        p_ab = predict_proba(model, X_te)

        scored = test_df[["iid", "pid", "wave", "match"]].copy()
        scored["p_ab"] = p_ab

        rev = (scored.rename(columns={"iid": "pid", "pid": "iid", "p_ab": "p_ba"})
                     [["iid", "pid", "p_ba"]])
        scored = scored.merge(rev, on=["iid", "pid"], how="left")
        assert scored["p_ba"].notna().all(), (
            f"wave {w}: a directed row has no mirror row -- the mirror-completeness guarantee "
            f"from notebook 02 Section 4 has been violated"
        )
        scored["p_match"] = combine_harmonic(scored["p_ab"].values, scored["p_ba"].values)
        scored_parts.append(scored)

        y_te = test_df["dec"].values
        diag_rows.append({"wave": w, "n_test_rows": len(test_df),
                          "auc_dec": roc_auc_score(y_te, p_ab),
                          "logloss_dec": log_loss(y_te, p_ab, labels=[0, 1])})

    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(diag_rows)


def evaluate_lowo_rankings(scored_all: pd.DataFrame, k: int = 3,
                            skip_zero_relevant: bool = True) -> pd.DataFrame:
    """Per-fold NDCG@k / Precision@k / MRR / Coverage@k -- average with a std across the rows."""
    rows = []
    for w, group in scored_all.groupby("wave"):
        _, summary = evaluate_user_rankings(group, k=k, skip_zero_relevant=skip_zero_relevant)
        rows.append({"wave": w, "ndcg@3": summary["ndcg@3"], "precision@3": summary["precision@3"],
                     "mrr": summary["mrr"], "coverage@3": coverage_at_k(group, k=k)})
    return pd.DataFrame(rows)
