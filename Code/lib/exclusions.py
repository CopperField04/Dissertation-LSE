"""
lib/exclusions.py   (v4 -- staged)
==================================
The single authoritative feature-exclusion list, shared by the aggregate and
panel pipelines, plus an independent raw-value scanner that audits it.

WHAT CHANGED IN v4, AND WHY
---------------------------
v3 merged three unrelated decisions into one drop list:

    structural       this is not a feature at all (calendar dummies)
    correctness      this is a bug (ISO, discontinued OAP)
    scale pathology  this feature's z-scores will misbehave

Only the first two have to be settled before normalisation. The third should
not be settled here at all, because of a flaw in the evidence v3 relied on:

    var_share_of_max is measured on RAW values with NO CAP. The +/-10
    into-estimator cap exists precisely to stop one observation owning the
    variance. So flagging open_vs_mid at std/sigma_f = 13,429 and calling it
    dead uses a measurement of exactly the quantity the normaliser is designed
    to eliminate. Post-cap that observation contributes at most 10 sigma to the
    moments, the standard deviation stays sane, and the feature may be entirely
    usable.

v4 therefore stages every entry:

    stage="now"     dropped in notebook 01. The justification does not depend on
                    z-scores: structural, correctness, zero-information, or
                    redundancy.
    stage="review"  NOT dropped. Recorded as a candidate for notebook 05, which
                    re-measures scale behaviour AFTER capping and z-scoring and
                    decides with evidence. Each carries a review_check saying
                    what to verify.

Net effect: notebook 01 is conservative and reversible, and no feature is
discarded on the strength of a statistic the pipeline was built to neutralise.

TAXONOMY CORRECTION
-------------------
v3 grouped n30_pos, n5_pos and n_obs alongside the calendar dummies because they
surfaced together in the sigma_f rung-2/3 listing. They are not calendar
features. They are TAQ interval counts -- intervals with positive 30-second
returns, positive 5-minute returns, and intervals with valid quotes. They are
dropped for a different and more informative reason: n30_pos is identical on
99.4% of stock-days, and a count over roughly 780 intervals per day cannot
legitimately be that constant. That points at a broken TAQ field rather than a
bounded temporal indicator.

EVIDENCE
--------
    std / sigma_f       ordinary standard deviation over the robust scale.
    implied_typical_z   sigma_f / std -- the z a one-robust-sigma move receives.
    var_share_of_max    share of total squared deviation owned by one point.
    modal_share         fraction of observations equal to the modal value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


MOMENT_SUFFIXES = ("_cwmean", "_cwstd", "_cwskew", "_cwkurt", "_spread")
MONTHLY_PREFIX = "monthly_"

# ── Configuration ───────────────────────────────────────────────────────────
# The TAQ spread/impact families come in three weightings: _ave (equal), _dw
# (dollar) and _sw (share). Redundancy between them is scale-invariant -- it
# will not change after z-scoring -- so this is one of the few scale-adjacent
# judgements that can safely be made now. Set either flag to True to retain.
KEEP_DOLLAR_WEIGHTED = False
KEEP_SHARE_WEIGHTED = False

_TAQ_WEIGHT_FAMILIES = (
    "effectivespread_dollar", "effectivespread_percent",
    "dollarpriceimpact_lr", "dollarrealizedspread_lr",
    "percentpriceimpact_lr", "percentrealizedspread_lr",
)
_DROPPED_WEIGHTINGS = tuple(
    w for w, keep in (("dw", KEEP_DOLLAR_WEIGHTED), ("sw", KEEP_SHARE_WEIGHTED))
    if not keep
)

BINARY_FEATURES = (
    "vix_above_20", "vix_above_30",
    "curve_inverted_2y10y", "curve_inverted_3m10y", "credit_stress",
)

# ── Detection thresholds ────────────────────────────────────────────────────
CONSTANT_STD = 1e-12
HIGH_NAN_RATE = 0.30
SCALE_INFLATED_IMPLIED_Z = 0.05
SINGLE_POINT_DOMINANCE = 0.50
# Above this modal share a feature carries essentially nothing: 99% of
# observations identical leaves too little variation for a spline to learn from,
# and no amount of robust scaling changes that.
ZERO_INFORMATION_MODAL = 0.95

KNOWN_LEAKS = {
    "iso": ("iso_dollar_to_cap", "iso_vol_to_shrout", "n_iso_trade_pct"),
    "discontinued_oap": ("OptionVolume1", "OptionVolume2", "PriceDelayRsq",
                         "PriceDelaySlope", "PriceDelayTstat"),
}

DROP_FLAGS = frozenset({"constant", "dead_single_outlier"})
KEEP_FLAGS = frozenset({"heavy_tailed"})
INVESTIGATE_FLAGS = frozenset({"high_nan", "insufficient_data"})


# ═══════════════════════════════════════════════════════════════════════════════
# THE EXCLUSION LIST
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Exclusion:
    factor: str
    category: str
    reason: str
    pipelines: str = "both"      # both | aggregate | panel
    stage: str = "now"           # now | review
    evidence: str = ""
    review_check: str = ""       # for stage="review": what to verify later


EXCLUSIONS: tuple[Exclusion, ...] = (

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE "NOW" -- justification does not depend on z-scores
    # ═══════════════════════════════════════════════════════════════════════

    # ── Structural: not features in this design ─────────────────────────────
    *[
        Exclusion(
            c, "calendar",
            "Calendar indicator on a bounded scale, never z-scored. Formerly "
            "theme 14.1, removed in the old Stage 4.",
            evidence="Stage 4 REMOVING_CALENDAR_THEME",
        )
        for c in (
            "day_of_week", "is_monday", "is_friday", "month_of_year",
            "is_quarter_end", "trading_days_to_month_end",
            "is_turn_of_month", "is_opex_week",
        )
    ],

    # ── Correctness: bugs, not judgements ───────────────────────────────────
    *[
        Exclusion(
            c, "leak_iso",
            "Intermarket sweep order feature. Reg NMS created ISOs in 2007, so "
            "the series starts 162 days late. Dropped in Stage 2 for the "
            "aggregate; the panel loads Panel A from Stage 1.5 and therefore "
            "still carries it. Removing it makes the two pipelines agree.",
            pipelines="panel",
            evidence="Stage 2 notebook 01 iso_drop",
        )
        for c in KNOWN_LEAKS["iso"]
    ],
    *[
        Exclusion(
            f"{MONTHLY_PREFIX}{c}", "leak_discontinued",
            "OAP factor that stops publishing in late 2023/2024. Dropped in "
            "Stage 2 monthly for the aggregate. In the panel it survives to "
            "fillna(0.0) and reads as 'exactly the pooled mean' across the whole "
            "of 2024 -- the entire test period of Split_D. Fabricated signal in "
            "a test set.",
            pipelines="panel",
            evidence="Stage 2 notebook 03 oap_drop; affects 2024 = Split_D test",
        )
        for c in KNOWN_LEAKS["discontinued_oap"]
    ],

    # ── Zero information: too little variation for any scaling to help ──────
    # NOT calendar features (v3 grouped them there by mistake). TAQ interval
    # counts that are near-constant to a degree no genuine count could be.
    Exclusion(
        "n30_pos", "broken_taq_count",
        "Count of 30-second intervals with positive midpoint returns. Identical "
        "on 99.4% of stock-days and 97.2% of aggregate days. A count over "
        "roughly 780 intervals per day cannot legitimately be that constant; "
        "this is a broken TAQ field rather than a low-variance feature.",
        evidence="panel modal 0.994, aggregate modal 0.972",
    ),
    Exclusion(
        "n5_pos", "broken_taq_count",
        "Count of 5-minute intervals with positive midpoint returns. Identical "
        "on 98.4% of stock-days. Same failure as n30_pos.",
        evidence="panel modal 0.984, aggregate modal 0.788",
    ),
    Exclusion(
        "n_obs", "broken_taq_count",
        "Count of observation intervals with valid quotes. Identical on 98.3% of "
        "stock-days. A data-coverage diagnostic rather than a predictor.",
        evidence="panel modal 0.983, aggregate modal 0.756",
    ),
    Exclusion(
        "vxd_overnight_gap", "zero_information",
        "VXD overnight gap. Exactly zero on 64.2% of days, meaning the VXD open "
        "equals the previous close -- impossible for a genuine index and "
        "indicative of a stale open price in the CBOE VXD series.",
        evidence="modal_share = zero_share = 0.642",
    ),
    Exclusion(
        "rf", "zero_information",
        "Daily risk-free rate. Takes only three distinct values across the "
        "sample and is exactly zero on 63.3% of days (ZIRP). Retained in Stage 2 "
        "for any excess-return computation but carries nothing as a feature.",
        evidence="modal_share = 0.633; three distinct values; wall mass 0.00%",
    ),

    # ── Zero information at stock level only ────────────────────────────────
    # These are corporate-event indicators. At the aggregate they are
    # cap-weighted means over ~100 stocks and retain 53-95% modal share, which
    # the sigma_f ladder handles -- so they are deferred there, not dropped.
    # At stock level they exceed 99%, which no scaling can rescue.
    *[
        Exclusion(
            f"{MONTHLY_PREFIX}{f}", "zero_information",
            f"Corporate-event indicator, identical on {m:.1%} of stock-months. "
            f"Retained for the aggregate, where cap-weighting across ~100 "
            f"constituents leaves usable variation.",
            pipelines="panel",
            evidence=f"panel modal {m:.3f}",
        )
        for f, m in (("ExchSwitch", 1.000), ("IndIPO", 0.998),
                     ("DivOmit", 0.998), ("Spinoff", 0.994), ("DivInit", 0.993))
    ],
    Exclusion(
        "dlyreti", "zero_information",
        "Dividend-only daily return. Exactly zero on 98.7% of stock-days -- "
        "dividends are quarterly. Retained for the aggregate, where "
        "cap-weighting makes it a meaningful market dividend-yield series "
        "(modal 0.383).",
        pipelines="panel",
        evidence="panel modal 0.987 vs aggregate 0.383",
    ),

    # ── Redundancy: scale-invariant, so safe to settle now ──────────────────
    # Correlation between _ave, _dw and _sw does not change after z-scoring, so
    # deferring this decision would gain nothing. Two of the twelve are also
    # demonstrably broken; the rest are parsimony. 2,199 features against 4,299
    # observations is already very wide for a spline model.
    *[
        Exclusion(
            f"{fam}_{w}", "redundant_weighting",
            f"{'Dollar' if w == 'dw' else 'Share'}-weighted variant of {fam}. "
            f"The _ave variant measures the same quantity through a stable "
            f"denominator and is retained. Redundancy is scale-invariant, so "
            f"this does not need to wait for normalisation.",
            evidence="_dw: std/sigma_f 126-257 with max|z| in the thousands. "
                     "_sw: mostly healthy, dropped for parsimony. Toggle with "
                     "KEEP_DOLLAR_WEIGHTED / KEEP_SHARE_WEIGHTED.",
        )
        for fam in _TAQ_WEIGHT_FAMILIES
        for w in _DROPPED_WEIGHTINGS
    ],
    Exclusion(
        "venue_range_a", "redundant_venue",
        "ARCA intraday range. CRSP intraday_range measures the same quantity "
        "across all venues and is healthy.",
        evidence="std/sigma_f = 18; max|z| = 357 on 2008-09-29 (TARP rejection)",
    ),
    Exclusion(
        "venue_range_b", "redundant_venue",
        "BATS intraday range. Redundant with CRSP intraday_range; BATS coverage "
        "also starts late (2005).",
    ),
    Exclusion(
        "venue_range_m", "redundant_venue",
        "All-exchange intraday range. Directly redundant with CRSP "
        "intraday_range.",
        evidence="std/sigma_f = 271; max|z| = 0.3",
    ),

# ═══════════════════════════════════════════════════════════════════════
    # STAGE "REVIEW" -- deferred to notebook 05, after capping and z-scoring
    # ═══════════════════════════════════════════════════════════════════════
    # Every entry below was a v3 drop. The +/-10 cap may well rescue them, and
    # the evidence that condemned them was measured without it.

    Exclusion(
        "ivol_q", "scale_pathology",
        "Intraday quote-midpoint variance, order 1e-8. One crossed quote moves "
        "it six orders of magnitude and the inflated scale then flattens every "
        "other observation.",
        stage="review",
        evidence="std/sigma_f = 8,367; max|z| = 2,790 on 2010-05-07",
        review_check="After capping: is implied_typical_z above 0.05, and does "
                     "wall mass fall below 1%? If yes the cap has fixed it. If "
                     "the extremes cluster on crossed-quote dates, the values "
                     "are wrong rather than merely large -- drop.",
    ),
    Exclusion(
        "ivol_t", "scale_pathology",
        "Intraday trade-price variance. Same failure mode as ivol_q.",
        stage="review",
        evidence="std/sigma_f = 416; max|z| = 101 on 2008-09-19",
        review_check="As ivol_q.",
    ),
    Exclusion(
        "open_vs_mid", "scale_pathology",
        "Opening trade versus NBBO midpoint. Scale destroyed by a single "
        "outlier, so a normal move currently receives z = 0.00007.",
        stage="review",
        evidence="std/sigma_f = 13,429; max|z| = 0.1; var_share_of_max high",
        review_check="This is the clearest test of whether the cap works. One "
                     "point owns the variance, which is exactly what the cap "
                     "prevents. If post-cap implied_typical_z is near 1, the "
                     "feature is fine and v3 would have discarded it wrongly.",
    ),
    Exclusion(
        "monthly_ChInvIA", "scale_pathology",
        "Industry-adjusted inventory change. Contains a value of order -8.7e12.",
        stage="review",
        evidence="std/sigma_f = 4.13e11; max|z| = 0.4",
        review_check="Find the offending stock-month first. If -8.7e12 is a "
                     "units or sign error, fix or drop at source. If it is a "
                     "real (if extreme) inventory swing, the cap handles it.",
    ),
    Exclusion(
        "monthly_BPEBM", "scale_pathology",
        "Book price-to-earnings x book-to-market. A product of two ratios, so "
        "denominators compound.",
        stage="review",
        evidence="std/sigma_f = 121; max|z| = 0.3",
        review_check="Check post-cap implied_typical_z. Note EP and BMdec, the "
                     "constituents, are retained and healthy, so little is lost "
                     "if this goes.",
    ),
    Exclusion(
        "monthly_EBM", "scale_pathology",
        "Earnings x book-to-market. Same compounding as BPEBM.",
        stage="review",
        evidence="std/sigma_f = 42; max|z| = 0.3",
        review_check="As BPEBM.",
    ),
    Exclusion(
        "monthly_HerfBE", "scale_pathology",
        "Herfindahl index on book equity. Worst feature in the dataset by "
        "boundary mass. Divides by book equity, which approaches zero for some "
        "firms. Herf (sales) and HerfAsset (assets) measure the same concept on "
        "stable denominators and are retained.",
        stage="review",
        evidence="wall mass 8.8% (cwmean), 11.7% (cwstd); std/sigma_f "
                 "5,050-9,546; max|z| = 4,933 on 2020-06-30",
        review_check="Highest-priority review item. If post-cap wall mass stays "
                     "above 5%, drop -- the redundancy with Herf and HerfAsset "
                     "means the cost is near zero.",
    ),
    *[
        Exclusion(
            f"{MONTHLY_PREFIX}{f}", "sparse_indicator",
            f"Corporate-event indicator, modal share {m:.3f} at the aggregate. "
            f"The sigma_f ladder gives it a usable scale via rung 2, so it is "
            f"not structurally dead here as it is at stock level.",
            pipelines="aggregate",
            stage="review",
            evidence=f"aggregate modal {m:.3f}; sigma_f from ladder rung 2",
            review_check="Post-z-score, does it show any relationship with the "
                         "target, or is it a flat line with occasional spikes? "
                         "Judge on the plotted spline.",
        )
        for f, m in (("DivInit", 0.560), ("Spinoff", 0.532), ("DivOmit", 0.794),
                     ("IndIPO", 0.817), ("ExchSwitch", 0.945),
                     ("delbreadth_chg_1m", 0.670))
    ],
)


# ── Moment-level surgery ────────────────────────────────────────────────────
# Matched canonically, so an entry written with or without the monthly_ prefix
# fires against either the Stage 2 monthly table (unprefixed) or a combined
# table (prefixed).
MOMENT_DROPS: dict[str, tuple[str, str]] = {
    # key: (stage, reason)
    "monthly_DivSeason_spread": (
        "now",
        "p90-p10 identical in every month of the sample (modal 1.000). Standard "
        "deviation is exactly zero: no scaling can create information."),
    "monthly_rec_median_spread": (
        "now",
        "p90-p10 identical in every month. rec_median is on a 1-5 scale, so its "
        "cross-sectional spread is structurally constant."),
    "monthly_analyst_alignment_spread": (
        "now",
        "p90-p10 identical in every month. The variable takes values in "
        "{-1, 0, +1}, so the spread is fixed by construction."),
    "dlyreti_spread": (
        "review",
        "p90-p10 of dividend-only return. Exactly zero for long stretches, so "
        "the expanding std stays near zero until roughly 2013. The sigma_f floor "
        "may make this usable; check whether the early period is still degenerate "
        "once the warm-up is extended."),
    "monthly_earnings_surprise_chg_1m_spread": (
        "review",
        "p90-p10 of the one-month change in earnings surprise. Exactly zero in "
        "63.8% of months. The ladder gives it a scale; check the plotted spline "
        "for a spike of mass at zero."),
}


WATCHLIST: dict[str, str] = {
    "PC_Ratio":
        "std/sigma_f = 115 at stock level against no aggregate flag. Put-call "
        "ratios are genuinely extreme for some stocks on some days. Stage 2 "
        "recorded a maximum of 291,812 against a 99.9th percentile of 13.8, "
        "which is the classic signature of a ratio needing a log transform "
        "rather than a drop. Handle in notebook 02.",
    "monthly_VarCF":
        "std/sigma_f = 31, wall mass 1.95%. Cash-flow volatility is genuinely "
        "heavy-tailed rather than broken; the cap is the right treatment.",
    "vxd":
        "vxd_overnight_gap is exactly zero on 64% of days and vix_vxd_ratio "
        "reaches 74 sigma on 2021-07-13. vxd and vxd_intraday_range are retained "
        "because DJIA volatility is distinct information, but if VXD proves "
        "unreliable the family should follow vxd_overnight_gap out.",
    "vxd_intraday_range":
        "Wall mass 2.37%, max|z| = 35 on 2021-07-13, the same date on which "
        "vix_vxd_ratio spikes. Symptom of the VXD problem above.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def build_column_map(columns: Iterable[str]) -> dict[str, tuple[str, str]]:
    """
    Map each column to (base_factor, moment_type).

    A moment suffix is only honoured when the same base also appears with
    _cwmean. Genuine moment columns come in families of five; a macro series
    that merely ends in the word "spread" does not. Without this guard
    other_spread, lev_spread, bull_bear_spread, bbb_aaa_spread, vix_term_spread,
    ff_2y_spread and the rest are misread as spread moments.
    """
    cols = list(columns)
    colset = set(cols)
    out: dict[str, tuple[str, str]] = {}
    for c in cols:
        base, moment = c, "level"
        for s in MOMENT_SUFFIXES:
            if c.endswith(s):
                cand = c[: -len(s)]
                if f"{cand}_cwmean" in colset:
                    base, moment = cand, s[1:]
                break
        out[c] = (base, moment)
    return out


def canonical(name: str) -> str:
    """Strip an optional monthly_ prefix so one list matches both namings."""
    return name[len(MONTHLY_PREFIX):] if name.startswith(MONTHLY_PREFIX) else name


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST SCALE -- THE FULL LADDER
# ═══════════════════════════════════════════════════════════════════════════════

def robust_scale_ladder(v: np.ndarray) -> tuple[float, int]:
    """
    Robust scale with the full four-rung ladder. Returns (sigma_f, rung).

    Every rung is a quantile, so no rung can be moved by an extreme value: a
    -8.7e12 datum is merely "the smallest value", and half the data would have to
    be corrupted to shift a median. Neither an ordinary standard deviation nor a
    mean absolute deviation can be used, since both are destroyed by the very
    values sigma_f exists to bound.

      rung 1  Scaled MAD, 1.4826 * median(|x - median|). The normal case.
      rung 2  Fires when the MAD is exactly zero -- more than half the values
              identical, so the median absolute deviation vanishes. Substitutes
              P95(|dev|)/1.9600, where 1.9600 is the 95th percentile of |N(0,1)|
              so the result still reads as a standard deviation.
      rung 3  Fires when P95(|dev|) is also zero (>95% ties). Takes the MAD of
              the strictly positive deviations, measuring spread using only the
              observations that moved. Fails SAFE: a 99%-constant feature with a
              few enormous movers gets a LARGE sigma_f, so the floor dominates
              and the feature contributes z ~ 0 -- correct for something that
              barely moves. Falling through to epsilon would fail DANGEROUS,
              giving a near-zero denominator and exploding z-scores.
      rung 4  Genuinely constant. Epsilon, a numerical catch only.

    Rungs 2 and 3 are load-bearing: without them ConvDebt (89.1% ties),
    ShareRepurchase (83.8%) and every binary indicator receive sigma_f = 1e-8,
    which makes the denominator floor meaningless and made all of them look dead
    in an earlier version of this scanner.

    For rung-2 and rung-3 features sigma_f is a rough proxy scale rather than a
    calibrated standard deviation -- a zero-inflated distribution is nowhere near
    Gaussian, so the 1.9600 equivalence does not really hold. Acceptable, because
    sigma_f only feeds the denominator floor and the warm-up trim.
    """
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, 0

    abs_dev = np.abs(v - np.median(v))

    scale = 1.4826 * float(np.median(abs_dev))
    if scale > 1e-12:
        return max(scale, 1e-8), 1

    scale = float(np.percentile(abs_dev, 95)) / 1.9600
    if scale > 1e-12:
        return max(scale, 1e-8), 2

    pos = abs_dev[abs_dev > 1e-12]
    if len(pos):
        scale = 1.4826 * float(np.median(pos))
        if scale > 1e-12:
            return max(scale, 1e-8), 3

    return 1e-8, 4


def robust_scale(v: np.ndarray) -> float:
    return robust_scale_ladder(v)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExclusionResult:
    to_drop: list[str] = field(default_factory=list)
    by_category: dict[str, list[str]] = field(default_factory=dict)
    matched_factors: list[str] = field(default_factory=list)
    unmatched_factors: list[str] = field(default_factory=list)
    unmatched_moment_drops: list[str] = field(default_factory=list)
    deferred: dict[str, list[str]] = field(default_factory=dict)
    watchlist_present: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    canonical_collisions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_drop(self) -> int:
        return len(self.to_drop)

    @property
    def n_deferred_cols(self) -> int:
        return sum(len(v) for v in self.deferred.values())


def resolve_exclusions(
    columns: Iterable[str],
    colmap: dict[str, tuple[str, str]] | None = None,
    for_pipeline: str = "aggregate",
    stage: str = "now",
) -> ExclusionResult:
    """
    Expand the exclusion list into concrete column names present in `columns`.

    stage="now" drops only entries whose justification is independent of
    z-scores. Entries marked stage="review" are NOT dropped; they are collected
    in `result.deferred` for notebook 05, which decides after the +/-10 cap and
    the expanding z-score have been applied.
    """
    if for_pipeline not in ("aggregate", "panel"):
        raise ValueError(f"for_pipeline must be 'aggregate' or 'panel', "
                         f"got {for_pipeline!r}")

    cols = list(columns)
    if colmap is None:
        colmap = build_column_map(cols)

    res = ExclusionResult()

    raw_bases: dict[str, set[str]] = {}
    for c in cols:
        raw_bases.setdefault(canonical(colmap[c][0]), set()).add(colmap[c][0])
    res.canonical_collisions = {k: sorted(v) for k, v in raw_bases.items()
                                if len(v) > 1}

    fam: dict[str, list[str]] = {}
    for c in cols:
        fam.setdefault(canonical(colmap[c][0]), []).append(c)

    seen: set[str] = set()

    for ex in EXCLUSIONS:
        if ex.pipelines not in ("both", for_pipeline):
            continue
        hits = sorted(fam.get(canonical(ex.factor), []))
        if not hits:
            if ex.stage == stage:
                res.unmatched_factors.append(ex.factor)
            continue

        if ex.stage != stage:
            if ex.stage == "review":
                res.deferred[ex.factor] = hits
            continue

        res.matched_factors.append(ex.factor)
        res.by_category.setdefault(ex.category, [])
        for h in hits:
            if h not in seen:
                seen.add(h)
                res.to_drop.append(h)
                res.by_category[ex.category].append(h)
                res.reasons[h] = f"[{ex.category}] {ex.reason}"

    canon_to_col: dict[str, list[str]] = {}
    for c in cols:
        canon_to_col.setdefault(canonical(c), []).append(c)

    for key, (md_stage, reason) in MOMENT_DROPS.items():
        hits = sorted(canon_to_col.get(canonical(key), []))
        if not hits:
            if md_stage == stage:
                res.unmatched_moment_drops.append(key)
            continue
        if md_stage != stage:
            if md_stage == "review":
                res.deferred[key] = hits
            continue
        for h in hits:
            if h not in seen:
                seen.add(h)
                res.to_drop.append(h)
                res.by_category.setdefault("degenerate_moment", []).append(h)
                res.reasons[h] = f"[degenerate_moment] {reason}"

    for w in WATCHLIST:
        if canonical(w) in fam:
            res.watchlist_present.append(w)

    res.to_drop = sorted(res.to_drop)
    res.matched_factors = sorted(set(res.matched_factors))
    res.unmatched_factors = sorted(set(res.unmatched_factors))
    res.unmatched_moment_drops = sorted(set(res.unmatched_moment_drops))
    res.watchlist_present = sorted(set(res.watchlist_present))
    return res


def verify_leaks_closed(surviving_columns: Iterable[str],
                        colmap: dict[str, tuple[str, str]] | None = None
                        ) -> dict[str, list[str]]:
    cols = list(surviving_columns)
    if colmap is None:
        colmap = build_column_map(cols)
    present = {canonical(colmap[c][0]) for c in cols}
    return {g: sorted(canonical(f) for f in factors if canonical(f) in present)
            for g, factors in KNOWN_LEAKS.items()}


def review_register() -> pd.DataFrame:
    """
    Every deferred candidate with the check to perform in notebook 05.

    Written out as a CSV so the deferral is an actionable artefact rather than a
    promise. Notebook 05 reads it, re-measures each entry post-cap, and records
    the decision alongside the evidence.
    """
    rows = [{"factor": e.factor, "category": e.category,
             "pipelines": e.pipelines, "reason": e.reason,
             "evidence_pre_cap": e.evidence, "review_check": e.review_check}
            for e in EXCLUSIONS if e.stage == "review"]
    rows += [{"factor": k, "category": "degenerate_moment", "pipelines": "both",
              "reason": v[1], "evidence_pre_cap": "", "review_check": v[1]}
             for k, v in MOMENT_DROPS.items() if v[0] == "review"]
    return pd.DataFrame(rows).sort_values(["category", "factor"]) \
                             .reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_candidates(df: pd.DataFrame, feature_cols: Iterable[str],
                      skip: Iterable[str] = BINARY_FEATURES) -> pd.DataFrame:
    """
    Scan raw (pre-z-score) values. Audits the list; does not replace it.

    IMPORTANT INTERPRETIVE CAVEAT
    -----------------------------
    Everything here is measured WITHOUT the +/-10 into-estimator cap, which the
    normaliser applies. var_share_of_max in particular measures precisely the
    condition the cap is designed to remove. A high value therefore means "this
    feature NEEDS the cap", not "this feature is beyond saving". That is why
    scale-pathology entries are staged for review rather than dropped here.

    Columns:
        sigma_f, sigma_f_rung  robust scale from the full ladder, and which rung.
        implied_typical_z      sigma_f / std, the z a one-robust-sigma move gets.
        var_share_of_max       share of total squared deviation owned by one point.
        max_dev_sigmas         how many standard deviations the largest deviation
                               sits from the mean. More interpretable than
                               var_share: 0.87 at n = 525,957 means 677 sigma,
                               which is unmistakably a data error rather than a
                               market event.
    """
    skip = set(skip)
    rows = []

    for c in feature_cols:
        if c in skip:
            rows.append({"column": c, "n_valid": np.nan, "nan_rate": np.nan,
                         "std": np.nan, "sigma_f": np.nan,
                         "sigma_f_rung": np.nan, "implied_typical_z": np.nan,
                         "var_share_of_max": np.nan, "max_dev_sigmas": np.nan,
                         "modal_share": np.nan, "zero_share": np.nan,
                         "flag": "binary_excluded"})
            continue

        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        finite = v[np.isfinite(v)]
        nan_rate = 1.0 - len(finite) / len(v) if len(v) else 1.0

        if len(finite) < 10:
            rows.append({"column": c, "n_valid": len(finite),
                         "nan_rate": nan_rate, "std": np.nan, "sigma_f": np.nan,
                         "sigma_f_rung": np.nan, "implied_typical_z": np.nan,
                         "var_share_of_max": np.nan, "max_dev_sigmas": np.nan,
                         "modal_share": np.nan, "zero_share": np.nan,
                         "flag": "insufficient_data"})
            continue

        sd = float(np.std(finite, ddof=1))
        sf, rung = robust_scale_ladder(finite)
        implied = (sf / sd) if sd > CONSTANT_STD else np.nan

        dev = finite - float(np.mean(finite))
        dev2 = dev ** 2
        tot = float(dev2.sum())
        var_share = float(dev2.max() / tot) if tot > 0 else np.nan
        max_sig = float(np.abs(dev).max() / sd) if sd > CONSTANT_STD else np.nan

        _, counts = np.unique(np.round(finite, 12), return_counts=True)
        modal = float(counts.max() / len(finite))
        zero = float(np.mean(np.abs(finite) < 1e-12))

        if sd <= CONSTANT_STD:
            flag = "constant"
        elif nan_rate > HIGH_NAN_RATE:
            flag = "high_nan"
        elif (not np.isnan(implied)) and implied < SCALE_INFLATED_IMPLIED_Z:
            flag = ("dead_single_outlier"
                    if (not np.isnan(var_share)
                        and var_share >= SINGLE_POINT_DOMINANCE)
                    else "heavy_tailed")
        else:
            flag = ""

        rows.append({"column": c, "n_valid": len(finite), "nan_rate": nan_rate,
                     "std": sd, "sigma_f": sf, "sigma_f_rung": rung,
                     "implied_typical_z": implied,
                     "var_share_of_max": var_share, "max_dev_sigmas": max_sig,
                     "modal_share": modal, "zero_share": zero, "flag": flag})

    return pd.DataFrame(rows)


def audit(detected: pd.DataFrame, result: ExclusionResult,
          colmap: dict[str, tuple[str, str]]) -> dict[str, pd.DataFrame]:
    """
    Reconcile the list against the detected flags.

    needs_cap     scale pathology, not dropped now and not already deferred.
                  These rely on the +/-10 cap; notebook 05 confirms it worked.
    already_known flagged, and already on the deferred register. No action.
    unsupported   dropped without a scale flag -- structural, correctness,
                  zero-information or redundancy. Expected.
    """
    dropped = set(result.to_drop)
    deferred_cols = {c for v in result.deferred.values() for c in v}
    flagged = detected[detected["flag"] != ""].copy()

    scale_flagged = flagged[flagged["flag"].isin(DROP_FLAGS | KEEP_FLAGS)]
    outstanding = scale_flagged[
        ~scale_flagged["column"].isin(dropped | deferred_cols)].copy()
    if len(outstanding):
        outstanding["base_factor"] = outstanding["column"].map(
            lambda c: canonical(colmap[c][0]) if c in colmap else c)
        outstanding = outstanding.sort_values("var_share_of_max",
                                              ascending=False)

    already = scale_flagged[scale_flagged["column"].isin(deferred_cols)].copy()

    investigate = flagged[flagged["flag"].isin(INVESTIGATE_FLAGS)
                          & ~flagged["column"].isin(dropped)].copy()

    unsupported = pd.DataFrame({
        "column": sorted(dropped - set(scale_flagged["column"]))})
    if len(unsupported):
        unsupported["reason"] = unsupported["column"].map(result.reasons)

    return {"needs_cap": outstanding, "already_known": already,
            "investigate": investigate, "unsupported": unsupported,
            "flagged": flagged}


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def exclusion_table() -> pd.DataFrame:
    rows = [{"factor": e.factor, "category": e.category, "stage": e.stage,
             "pipelines": e.pipelines, "reason": e.reason,
             "evidence": e.evidence, "review_check": e.review_check}
            for e in EXCLUSIONS]
    rows += [{"factor": k, "category": "degenerate_moment", "stage": v[0],
              "pipelines": "both", "reason": v[1], "evidence": "",
              "review_check": ""}
             for k, v in MOMENT_DROPS.items()]
    return pd.DataFrame(rows).sort_values(["stage", "category", "factor"]) \
                             .reset_index(drop=True)


def print_summary(result: ExclusionResult, n_cols_before: int) -> None:
    print(f"\n  Columns: {n_cols_before:,} -> "
          f"{n_cols_before - result.n_drop:,} ({result.n_drop} dropped now)")

    if result.canonical_collisions:
        print(f"\n  ** CANONICALISATION COLLISIONS **")
        for k, v in sorted(result.canonical_collisions.items()):
            print(f"    {k:<32} <- {v}")

    print(f"\n  Dropped now, by category:")
    for cat in sorted(result.by_category):
        print(f"    {cat:<24} {len(result.by_category[cat]):>4} columns")

    if result.deferred:
        print(f"\n  DEFERRED to notebook 05 ({len(result.deferred)} factors, "
              f"{result.n_deferred_cols} columns) -- retained for now, decided "
              f"after capping:")
        for f in sorted(result.deferred):
            n = len(result.deferred[f])
            print(f"    {f:<40} {n:>3} column{'s' if n != 1 else ''}")

    if result.unmatched_moment_drops:
        print(f"\n  Stage-{'now'} moment drops not matched here: "
              f"{result.unmatched_moment_drops}")

    if result.watchlist_present:
        print(f"\n  Watchlist present: {', '.join(result.watchlist_present)}")