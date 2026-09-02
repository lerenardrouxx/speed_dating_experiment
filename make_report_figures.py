"""Regenerate the figures used in `report.qmd`.

Run from the repository root:

    python make_report_figures.py

Writes three 300-dpi PNGs into `figures/`. Everything is computed from
`Data/Speed_Dating_Data.csv` through `sd_features`, the same module notebook 02 uses, so a figure
cannot drift away from the feature table it describes. The numbers behind each panel are the ones
reported in `01_Speed_Dating_EDA.ipynb` and `02_Speed_Dating_Feature_Engineering.ipynb`.

Styling is for print, not for a notebook: text-width figures, ~9 pt labels, no chart titles (the
report caption carries them) and a tight bounding box.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sd_features as sf

OUT_DIR = Path("figures")
GREEN, RED, LIGHT = "#2f7d58", "#c44e34", "#7bb59a"
TEXT_WIDTH_IN = 6.2          # A4 with 1-inch margins

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig1_structure(raw: pd.DataFrame) -> None:
    """Wave sizes and candidate-set sizes: the CV unit, and why the cut-off is @3."""
    people_per_wave = raw.groupby("wave")["iid"].nunique()
    partners_per_person = raw.groupby("iid").size()

    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.0))
    fig.subplots_adjust(wspace=0.32)

    people_per_wave.plot(kind="bar", ax=ax[0], color=GREEN, width=0.75)
    ax[0].set_xlabel("Wave (one speed-dating event)")
    ax[0].set_ylabel("Participants")
    ax[0].tick_params(axis="x", rotation=0)
    ax[0].set_xticks(range(0, 21, 2))
    ax[0].set_xticklabels(range(1, 22, 2))
    ax[0].text(0.97, 0.93, f"n = {len(people_per_wave)} folds\n10–44 people",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=8)

    # Bins centred on the integers, so a bar sits over the count it represents.
    ax[1].hist(partners_per_person, bins=np.arange(4.5, 23.5, 1), color=LIGHT, edgecolor="white")
    ax[1].axvline(partners_per_person.median(), color=RED, linestyle="--", linewidth=1.2,
                  label=f"median {int(partners_per_person.median())}")
    ax[1].set_xlabel("Partners met (candidate-set size)")
    ax[1].set_ylabel("Participants")
    ax[1].legend(frameon=False, loc="upper left")

    save(fig, "fig1_structure.png")


def fig2_leakage(raw: pd.DataFrame, features: pd.DataFrame) -> None:
    """What the pre-date constraint costs: in-date ratings against admissible pair features."""
    in_date = ["like", "attr", "fun", "shar", "prob", "intel", "sinc", "amb"]
    pre_date = sf.FEATURES_DYADIC + sf.FEATURES_COMPAT

    rows = [{"feature": c, "r": abs(raw[c].corr(raw["dec"])), "family": "in-date (excluded)"}
            for c in in_date]
    rows += [{"feature": c, "r": abs(features[c].corr(features["dec"])),
              "family": "pre-date (admissible)"} for c in pre_date]
    sig = pd.DataFrame(rows).sort_values("r")

    # Print space is scarce: show the whole in-date family and the eight strongest pre-date
    # features. The ones dropped are weaker still, so the comparison is if anything understated.
    keep_pre = sig[sig["family"].str.startswith("pre-date")].tail(8)["feature"]
    sig = sig[sig["family"].str.startswith("in-date") | sig["feature"].isin(keep_pre)]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
    colours = [RED if f.startswith("in-date") else GREEN for f in sig["family"]]
    ax.barh(np.arange(len(sig)), sig["r"], color=colours, height=0.72)
    ax.set_yticks(np.arange(len(sig)))
    ax.set_yticklabels(sig["feature"])
    ax.set_xlabel("|correlation with the decision (dec)|")
    ax.set_xlim(0, 0.56)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RED, GREEN)]
    ax.legend(handles, ["recorded during the date — excluded",
                        "known before the date — admissible"],
              loc="lower right", frameon=False)
    save(fig, "fig2_leakage.png")


def fig3_stated_revealed(raw: pd.DataFrame) -> None:
    """Stated preference weights against the weights implied by actual decisions, per gender."""
    attrs = sf.ATTR6
    person = raw.drop_duplicates("iid").set_index("iid")
    total = person[sf.PREF1_RAW].sum(axis=1, min_count=len(sf.PREF1_RAW))
    stated = person[sf.PREF1_RAW].div(total, axis=0).set_axis(attrs, axis=1)
    stated = stated.join(person["gender"]).groupby("gender").mean()

    revealed = {}
    for code in (0, 1):
        sub = raw[raw["gender"] == code]
        c = sub[attrs].corrwith(sub["dec"]).clip(lower=0)
        revealed[code] = c / c.sum()
    revealed = pd.DataFrame(revealed).T

    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.1), sharey=True)
    for i, (code, label) in enumerate(sf.GENDER_LABELS.items() if hasattr(sf, "GENDER_LABELS")
                                      else {0: "female", 1: "male"}.items()):
        side = pd.DataFrame({"stated": stated.loc[code], "revealed": revealed.loc[code]})
        side.plot(kind="bar", ax=ax[i], color=[LIGHT, RED], width=0.78, legend=(i == 1))
        ax[i].set_xlabel(f"({'ab'[i]}) {label} participants")
        ax[i].tick_params(axis="x", rotation=0)
        if i == 0:
            ax[i].set_ylabel("Share of total weight")
        else:
            ax[i].legend(frameon=False)
    save(fig, "fig3_stated_revealed.png")


def main() -> None:
    raw = sf.load_raw()
    features = sf.build_features(raw)
    print(f"raw {raw.shape}, features {features.shape}")
    fig1_structure(raw)
    fig2_leakage(raw, features)
    fig3_stated_revealed(raw)


if __name__ == "__main__":
    main()
