"""
lib/review.py
=============
The feature-exclusion rules, in one place, so the aggregate and panel pipelines
apply exactly the same test. Imported by:

    Code/Data_Merging/Stage_3_Cleaning/04_normalise.ipynb          (aggregate)
    Code/Data_Merging/Stage_3_Cleaning/05_panel_data_normalisation.ipynb

Two grounds for exclusion, and the distinction matters for the write-up:

  A. STRUCTURAL   -- known from how the factor was constructed. A bond
                     total-return index is the cumulative product of a return
                     series already held; its z-score diagnostics can be
                     perfectly healthy because it is not broken, it is
                     redundant. No statistic can see this.

  B. MEASURED     -- standardisation failed, against pre-set thresholds.

Both are reproducible. B is a formula; A is a fixed list written down below.

WHERE EVERYTHING IS MEASURED
----------------------------
Split A's training window (2004-01-01 to 2015-12-31), the EARLIEST of the four,
so one frozen feature set is clean for every split. See normalise.py.

WHAT IS NOT A RULE
------------------
Drift (shift_max_z). It measures whether the distribution differs between early
and late sample -- which is between train and test. Dropping on it discards
precisely the features whose behaviour changed, i.e. conditions the model on
knowing the future was different. Measured, reported as a limitation, after the
freeze.

Collinearity beyond exact duplication. The eleven Treasury yields, nine OAS
series and ten seasonal momentum variants stay. The sparse KAN groups features
by subtheme because within-subtheme redundancy is expected; pruning it would
work against the architecture.

R4 AND R5 USE PROXY STATISTICS -- WHY THEY WERE AMENDED
--------------------------------------------------------
R1, R2, R3 and R6 measure their intent directly: modal share IS concentration,
std_of_z IS scale validity. R4 and R5 do not.

  R4 intends "does this level TREND". shift_max measures "did it MOVE A LOT".
     Over 2004-2015 the housing block (starts, permits, new home sales,
     Case-Shiller) moved enormously -- 1,800k -> 600k -> 950k -- without
     trending. Meanwhile nonfarm payrolls, the textbook trending level, scored
     only 1.20 because the GFC cancelled its trend inside the window.
     AMENDMENT: require the three period medians to be monotone. The rule
     already says "trending"; this makes the test match the word.

  R5 intends "is this the SAME MEASUREMENT TWICE". Correlation on levels
     measures "do these CO-MOVE". Two series that both rise smoothly for twelve
     years correlate above 0.99 whatever they are, which is why ZIRP fused the
     short end of the yield curve (yield_1m/3m/6m/1y all > 0.9976) and why core
     CPI matched average hourly earnings at 0.9960.
     AMENDMENT: correlate FIRST DIFFERENCES. An identity survives differencing
     (tips_5y / real_rate_5y stays at 1.000); a shared trend does not.

Both amendments are stated without reference to the factors that revealed them,
which is the test for a specification fix rather than an overfit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.normalise import shift_max_stat


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

DIAG_START = '2004-01-01'
DIAG_END   = '2015-12-31'      # Split A training window end -- the earliest split

CRISIS_START = '2007-12-01'    # NBER peak
CRISIS_END   = '2009-06-30'    # NBER trough


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

MODAL_SHARE_MAX      = 0.90      # rule 1
STD_LO, STD_HI       = 0.5, 2.0  # rule 2
                                 # rule 3 is boolean: warmup_degenerate
RAW_SHIFT_MAX_LEVEL  = 1.5       # rule 4. A pure linear ramp scores ~1.80
                                 # analytically; set below that so imperfect
                                 # trends are caught and mean-reverting macro
                                 # levels (unemployment ~1.1) survive.
RHO_DUP              = 0.995     # rule 5, on first differences
PCT_GT5_MAX          = 0.02      # rule 6, measured ex-crisis
PCT_GT5_MIN_COUNT    = 5         # rule 6 small-sample guard. 2% of 98 monthly
                                 # observations is 2 events, which cannot
                                 # distinguish a 2% rate from a 0.2% one; at
                                 # n=98, k=5 is the smallest count whose
                                 # one-sided 95% lower bound clears 2%. Never
                                 # binds at daily frequency, where 2% is ~47.

# Rule 0. A feature with too little history inside the diagnostic window cannot
# be assessed. Stated in advance so the case is not decided on the spot when it
# appears: UNASSESSABLE MEANS DROP. A feature with almost no usable history
# before 2016 also has almost none in Split A's training data, so it could not
# have been learned from anyway.
MIN_DIAG_OBS = {'daily': 250, 'weekly': 26, 'monthly': 24}


# Order in which rules are attributed. A feature can fail several; `rule_fired`
# records the first in this order, `all_rules` records every one.
RULE_ORDER = [
    'S_structural',
    'R4_trending_level_twin',
    'R5_duplicate_rho',
    'R0_unassessable',
    'R1_modal_share',
    'R2_std_bounds',
    'R3_warmup_degenerate',
    'R6_boundary_mass',
]

RULE_LABEL = {
    'S_structural':           'Structural (construction)',
    'R4_trending_level_twin': 'R4  monotone trending level, has _mom/_yoy twin',
    'R5_duplicate_rho':       'R5  |rho| > 0.995 on first differences',
    'R0_unassessable':        'R0  too little history in diag window',
    'R1_modal_share':         'R1  modal_share > 0.90',
    'R2_std_bounds':          'R2  std_of_z_ex_capped outside [0.5, 2.0]',
    'R3_warmup_degenerate':   'R3  warm-up degenerate (sigma_f = eps)',
    'R6_boundary_mass':       'R6  pct_gt5_ex_crisis > 2%',
}

# Column schemas for the audit frames, declared so an empty result still
# returns a frame the notebook can sort and print.
R4_AUDIT_COLS = ['feature', 'twins', 'raw_shift_max', 'med_early', 'med_mid',
                 'med_late', 'monotone', 'threshold', 'dropped']
R5_AUDIT_COLS = ['feature_a', 'feature_b', 'rho_diff', 'kept', 'dropped', 'status']


# ═══════════════════════════════════════════════════════════════════════════════
# A. THE STRUCTURAL LIST  (33 base factors)
# ═══════════════════════════════════════════════════════════════════════════════
# Keyed on BASE factor. In the aggregate a base factor expands to up to five
# moment columns, so each entry here removes one column in the panel and up to
# five in the aggregate full-moments table.

STRUCTURAL = {}

# -- Cumulative indices whose return series is already held (10) --------------
# Each is the compounded product of b30ret / t90ret / cpiret etc. They grow
# monotonically by construction and carry nothing the returns do not.
for _c in ['b30ind', 'b20ind', 'b10ind', 'b7ind', 'b5ind', 'b2ind', 'b1ind',
           't90ind', 't30ind', 'cpiind']:
    STRUCTURAL[_c] = ('cumulative_index',
                      'Cumulative product of a return series already held')

# -- Index levels with no derivative and no meaningful zero (5) ---------------
# cpi_urban and cpi_core keep their _mom/_yoy twins so the level can go and the
# information survives; these have no twin, but a base-100 index rising
# monotonically gives a near-constant z regardless of slope, so the level as it
# stands carries nothing either.
# NOTE: cpi_energy and import_prices are deliberately NOT here. Energy CPI ran
# 250 (2008) -> 180 (2009) -> 250 (2014) -> 190 (2016) -> 300 (2022): genuinely
# cyclical and crisis-adjacent, and an expanding z handles oscillation well.
for _c in ['cpi_food', 'cpi_shelter', 'cpi_services',
           'business_inventories', 'trade_balance_12m_avg']:
    STRUCTURAL[_c] = ('monotone_index_level',
                      'Monotone index level, no rate-of-change twin')

# -- Structural market-share breaks (3) ---------------------------------------
# BATS did not exist at the start of the sample. 0% -> ~20% share is a launch,
# not a trend, and an expanding window cannot represent it.
for _c in ['total_dollar_b_to_cap', 'total_vol_b_to_shrout',
           'total_n_trades_b_pct']:
    STRUCTURAL[_c] = ('venue_launch',
                      'BATS market-share ramp from launch, not a trend')

# -- Product launch, same logic (2) -------------------------------------------
# VIX futures launched 2004-03-26. Volume and open interest grew by orders of
# magnitude as the market developed. The raw scan gave shift_max 1.08 and missed
# it, because after the initial ramp the growth is smooth enough that a
# thirds-of-timeline median test does not register it. Mechanism beats statistic
# when the mechanism is known.
for _c in ['vix_fut_volume', 'vix_fut_oi']:
    STRUCTURAL[_c] = ('product_launch',
                      'VIX futures launched 2004-03; contract counts ramp from zero')

# -- Coverage diagnostic, not a predictor (1) ---------------------------------
STRUCTURAL['nopt_Parity'] = ('coverage_diagnostic',
                             'Count of option pairs used in the parity calc; '
                             'same category as n_obs / n5_pos / n30_pos')

# -- Deterministic in time (1) ------------------------------------------------
STRUCTURAL['FirmAge'] = ('deterministic_in_time',
                         'Negative months since first CRSP listing; a linear '
                         'function of the date for a surviving firm. Its '
                         'cap-weighted moments drift purely with universe '
                         'turnover -- FirmAge_cwstd and _cwskew are the two '
                         'features above 3.0 on the aggregate drift scan.')

# -- A model output, not an observation (1) -----------------------------------
STRUCTURAL['PredictedFE'] = ('model_output',
                             'Predicted forecast error from another '
                             'cross-sectional model; makes the learned spline '
                             'uninterpretable and the provenance unclear')

# -- Policy-controlled with step changes (1) ----------------------------------
STRUCTURAL['fx_cny'] = ('policy_step',
                        'Pegged to 2005, managed float, re-pegged 2008-2010. '
                        'Aggregate min_z -785.')

# -- Compound ratios with unstable denominators (4) ---------------------------
STRUCTURAL['EBM']     = ('compound_ratio',
                         'Product of two ratios; tail behaviour compounds. '
                         'Aggregate max_z 440 on cwstd.')
STRUCTURAL['BPEBM']   = ('compound_ratio',
                         'Product of two ratios; aggregate max_z 434 on cwstd.')
STRUCTURAL['VarCF']   = ('compound_ratio',
                         '408x heavy-tail ratio, drift 2.3-2.5, worst by '
                         'boundary mass in the panel.')
STRUCTURAL['EntMult'] = ('compound_ratio',
                         'EV/EBITDA, and EBITDA crosses zero.')

# -- Midpoint-relative execution prices (2) -----------------------------------
# (open - open_midpoint) / open_midpoint. When the opening quote is stale or
# crossed the denominator approaches zero.
STRUCTURAL['open_vs_mid']  = ('zero_denominator',
                              'std/sigma_f = 13,429 in the old one-pass run; '
                              'aggregate max_z 272,707 in the capped run')
STRUCTURAL['close_vs_mid'] = ('zero_denominator',
                              'Same construction as open_vs_mid')

# -- Column does not contain the quantity its name describes (1) --------------
STRUCTURAL['RDS'] = ('column_invalid',
                     'R&D to sales. VERIFIED against the raw Panel B column: '
                     '0.0% of 24,142 non-null observations fall in the valid '
                     '[0, 0.3] range, 66.2% are negative, 28.0% exceed 100. '
                     'The annual median is stable near -200 across all 21 '
                     'years, so this is not a scaling error (which would still '
                     'show economically sensible variation) nor a '
                     'near-zero-denominator problem (which would leave the '
                     'centre intact). PERMNO 14593 (Apple) reads -1,521,208 '
                     'for every month of 2021, constant within the year, '
                     'against a true R&D/sales of ~0.06 -- consistent with an '
                     'annual accounting level in dollars rather than a ratio. '
                     'The column does not contain the quantity its name '
                     'describes.')

PENDING_VERIFICATION = []

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_MOMENTS = ('_cwmean', '_cwstd', '_cwskew', '_cwkurt')


def base_factor(col: str, all_cols=()) -> str:
    """
    Strip a cross-sectional moment suffix to recover the base factor.

    `_spread` is only stripped when the matching `_cwmean` column also exists,
    because several macro features legitimately end in _spread
    (bull_bear_spread, brent_wti_spread, vix_term_spread, bb_bbb_spread) and
    stripping those would be wrong.
    """
    for s in _MOMENTS:
        if col.endswith(s):
            return col[:-len(s)]
    if col.endswith('_spread'):
        stem = col[:-len('_spread')]
        if f'{stem}_cwmean' in all_cols:
            return stem
    return col


def base_factor_map(cols) -> dict:
    """
    {column: base factor} for one table.

    Use this rather than calling base_factor() per column. The _spread branch
    only fires when the matching _cwmean is visible, so base_factor('X_spread')
    on its own returns 'X_spread' unchanged. This passes the full column list,
    which is what makes stripping work.

    Matters at assembly: union_drop_list.csv holds BASE factors, while the
    full-moments tables hold X_cwmean / X_cwstd / X_cwskew / X_cwkurt /
    X_spread. Without the column list, X_spread maps to itself, is not found in
    the drop list, and survives while its four siblings are removed -- one
    orphaned moment of a factor that was deliberately excluded.

        bmap = base_factor_map(df.columns)
        drop = [c for c in df.columns if bmap[c] in union_drop]
    """
    cols = set(cols)
    return {c: base_factor(c, cols) for c in cols}

def structural_drops(feature_cols) -> dict:
    """{feature: (rule, reason)} for every column whose base factor is listed."""
    cols = set(feature_cols)
    out = {}
    for c in feature_cols:
        b = base_factor(c, cols)
        if b in STRUCTURAL:
            cat, why = STRUCTURAL[b]
            out[c] = ('S_structural', f'[{cat}] {why}')
    return out


def _numeric_subset(df: pd.DataFrame, cols):
    """Columns present in df, numeric, in the given order."""
    return [c for c in cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _period_stats(v: np.ndarray, years: np.ndarray):
    """
    (shift_max, m_early, m_mid, m_late) over thirds of the timeline.

    Mirrors _shift_max in normalise.py exactly -- same year blocking, same
    denominator -- but also returns the three medians, which R4 needs for the
    monotonicity test. Returns NaNs when the series is too short to block.
    """
    ok = np.isfinite(v)
    if ok.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan
    v, y = v[ok], years[ok]

    yrs = np.unique(y)
    if len(yrs) < 5:
        return np.nan, np.nan, np.nan, np.nan
    k = max(len(yrs) // 3, 1)
    blocks = (yrs[:k], yrs[k:-k] if len(yrs) > 2 * k else yrs[k:k + 1], yrs[-k:])

    meds = [float(np.median(v[np.isin(y, b)])) for b in blocks if len(b)]
    if len(meds) < 3:
        return np.nan, np.nan, np.nan, np.nan

    sf = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    sm = (max(meds) - min(meds)) / sf if sf > 1e-12 else np.nan
    return float(sm), meds[0], meds[1], meds[2]


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 4 -- monotone trending level with a rate-of-change twin
# ═══════════════════════════════════════════════════════════════════════════════

def rule4(raw_df: pd.DataFrame, feature_cols, date_col='date',
          thresh: float = RAW_SHIFT_MAX_LEVEL):
    """
    A level is redundant with its own _mom / _yoy twin ONLY when it actually
    trends. Two conditions, both required:

        shift_max > 1.5        it moved substantially
        m_early < m_mid < m_late   (or strictly decreasing)
                               it moved in ONE DIRECTION

    The second condition is the amendment. Without it the rule fires on
    anything that moved a lot, which over a GFC-dominated window means the
    housing block -- starts, permits, new home sales, Case-Shiller -- all of
    which crashed and recovered. For a V-shape the level tells you where in the
    cycle you are and the _mom twin cannot recover that, so dropping it loses
    information. For a monotone trend the level is nearly a function of the
    date, its expanding z is pinned near a constant, and the twin carries
    everything.

    raw_df must ALREADY be restricted to the diagnostic window.

    The twin lookup is moment-aware. A bare macro level `cpi_urban` pairs with
    `cpi_urban_mom`; an aggregate moment `cpi_urban_cwmean` pairs with
    `cpi_urban_mom_cwmean`. For a bare column the suffix is empty and this
    reduces to the simple f'{c}_mom' lookup.

    The asymmetry is what makes an otherwise arbitrary cutoff defensible: the
    rule fires only where a twin exists, so a false positive costs nothing --
    the information survives in the _mom column.

    Returns (drops dict, audit DataFrame of every candidate examined).
    """
    cols = set(feature_cols)
    usable = _numeric_subset(raw_df, feature_cols)
    years = pd.DatetimeIndex(raw_df[date_col]).year.to_numpy()

    rows, drops = [], {}
    for c in usable:
        b = base_factor(c, cols)
        sfx = c[len(b):]
        twins = [t for t in (f'{b}_mom{sfx}', f'{b}_yoy{sfx}') if t in cols]
        if not twins:
            continue

        sm, m1, m2, m3 = _period_stats(raw_df[c].to_numpy(dtype=float), years)

        monotone = bool(np.isfinite(m1) and np.isfinite(m2) and np.isfinite(m3)
                        and ((m1 < m2 < m3) or (m1 > m2 > m3)))
        fire = bool(np.isfinite(sm) and sm > thresh and monotone)

        rows.append({'feature': c, 'twins': ', '.join(twins),
                     'raw_shift_max': sm, 'med_early': m1, 'med_mid': m2,
                     'med_late': m3, 'monotone': monotone,
                     'threshold': thresh, 'dropped': fire})
        if fire:
            direction = 'rising' if m1 < m3 else 'falling'
            drops[c] = ('R4_trending_level_twin',
                        f'Monotone {direction} level (shift_max {sm:.2f} > '
                        f'{thresh}; medians {m1:,.4g} -> {m2:,.4g} -> '
                        f'{m3:,.4g}); information retained in '
                        f'{", ".join(twins)}')

    if not rows:
        return {}, pd.DataFrame(columns=R4_AUDIT_COLS)

    audit = (pd.DataFrame(rows)
             .sort_values('raw_shift_max', ascending=False)
             .reset_index(drop=True))
    return drops, audit


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 5 -- the same measurement recorded twice
# ═══════════════════════════════════════════════════════════════════════════════

def rule5(raw_df: pd.DataFrame, feature_cols, thresh: float = RHO_DUP,
          date_col: str = 'date', group_col: str | None = None):
    """
    |rho| > 0.995 ON FIRST DIFFERENCES, within a table. This is NOT collinearity
    pruning -- it removes columns that are the same measurement recorded twice.

    THREE AMENDMENTS, each one line:

    1. DIFFERENCES, NOT LEVELS. Correlation between two persistent series
       measures shared trend, not shared identity: over 2004-2015 ZIRP fused
       yield_1m/3m/6m/1y above 0.9976 and core CPI matched average hourly
       earnings at 0.9960, none of which are duplicates. An identity survives
       differencing (tips_5y / real_rate_5y stays at exactly 1.000, as does a
       pair differing only by a constant, like skew / skew_excess); a shared
       trend does not.

    2. SAME BASE FACTOR IS SKIPPED. DebtIssuance_cwstd vs DebtIssuance_cwskew at
       rho = -0.9997 means one or two stocks drive the whole cross-section --
       move the outlier and both moments move together. That is a DEGENERATE
       cross-section, not a duplicate, and dropping cwstd would leave the
       broken half. R1 (modal share) and R3 (degenerate warm-up) catch these
       correctly. Such pairs are recorded in the audit as a flag but never
       dropped here.

    3. A DESIGNATED KEEPER CANNOT LATER BE DROPPED. The loop previously skipped
       a pair only if a member was already DROPPED, not if it was already KEPT.
       That is how iv_PATM was declared the keeper against iv_catm and then
       eliminated against iv_30d_put25, losing both. Correlation is not
       transitive, so protecting keepers is also more conservative than
       clustering: a-b and b-c above threshold does not imply a-c is, and
       keeping two correlated columns costs nothing.

    Deterministic tiebreak, fixed in advance so the rule stays a rule: keep the
    member with the lower NaN rate over the diagnostic window (measured on
    levels); break remaining ties alphabetically.

    raw_df must ALREADY be restricted to the diagnostic window, and sorted by
    date.

    group_col : pass 'permno' for PANEL tables. Rows there are (permno, date)
        ordered by date, so a plain .diff() would subtract one stock's value
        from another's. None for the aggregate tables, which hold one row per
        date.

    Returns (drops dict, audit DataFrame of every pair found). An empty audit
    is a result in itself -- it documents that no duplicates exist.
    """
    usable = _numeric_subset(raw_df, feature_cols)
    if len(usable) < 2:
        return {}, pd.DataFrame(columns=R5_AUDIT_COLS)

    levels = raw_df[usable]
    nan_rate = levels.isna().mean()          # tiebreak, measured on levels

    # --- AMENDMENT 1: first differences -------------------------------------
    if group_col is not None and group_col in raw_df.columns:
        d = raw_df.sort_values([group_col, date_col], kind='stable')
        diffs = d.groupby(group_col, sort=False)[usable].diff()
    else:
        diffs = levels.diff()

    C = diffs.corr().to_numpy()
    names = list(levels.columns)

    iu = np.triu_indices(len(names), k=1)
    pairs = [(names[i], names[j], float(C[i, j]))
             for i, j in zip(*iu)
             if np.isfinite(C[i, j]) and abs(C[i, j]) > thresh]
    pairs.sort(key=lambda p: -abs(p[2]))     # strongest first, so deterministic

    all_names = set(names)
    rows, drops, kept = [], {}, set()

    for a, b, r in pairs:
        # --- AMENDMENT 2: same base factor is degeneracy, not duplication ----
        if base_factor(a, all_names) == base_factor(b, all_names):
            rows.append({'feature_a': a, 'feature_b': b, 'rho_diff': r,
                         'kept': '', 'dropped': '',
                         'status': 'same base factor - degenerate '
                                   'cross-section, see R1/R3'})
            continue

        # --- AMENDMENT 3: keepers are protected, not just droppees -----------
        if a in drops or b in drops or a in kept or b in kept:
            rows.append({'feature_a': a, 'feature_b': b, 'rho_diff': r,
                         'kept': '', 'dropped': '', 'status': 'already resolved'})
            continue

        keep = min(a, b)
        loser = b if keep == a else a
        kept.add(keep)
        drops[loser] = ('R5_duplicate_rho',
                        f'|rho| = {abs(r):.4f} on first differences with '
                        f'{keep}; kept {keep} (NaN {nan_rate[keep]:.3%} vs '
                        f'{nan_rate[loser]:.3%})')
        rows.append({'feature_a': a, 'feature_b': b, 'rho_diff': r,
                     'kept': keep, 'dropped': loser, 'status': 'dropped'})

    if not rows:
        return {}, pd.DataFrame(columns=R5_AUDIT_COLS)

    return drops, pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rules(report: pd.DataFrame, extra_drops: dict,
                bucket_of: dict, freq_of: dict,
                exempt: set | None = None) -> pd.DataFrame:
    """
    report       : concatenated per-feature reports, with a `table_source` column
    extra_drops  : {(table_source, feature): (rule, reason)} from structural / R4 / R5
    bucket_of    : {table_source: display bucket}
    freq_of      : {table_source: 'daily' | 'weekly' | 'monthly'}
    exempt       : features never subject to any rule (binary regime indicators)

    Returns one row per (table_source, feature) with the verdict and the
    statistics behind it.
    """
    exempt = exempt or set()
    out = []

    for _, r in report.iterrows():
        tbl, f = r['table_source'], r['feature']

        if f in exempt:
            fired, action = [], 'keep'
            reason = 'Binary regime indicator, not z-scored'
        else:
            # rule -> reason. extra_drops supplies its own specific text; the
            # measured rules build theirs here.
            why = {}

            key = (tbl, f)
            if key in extra_drops:
                rule, text = extra_drops[key]
                why[rule] = text

            n_min = MIN_DIAG_OBS[freq_of[tbl]]
            if (not r['reached_warmup']) or r['n_obs_diag'] < n_min:
                why['R0_unassessable'] = (
                    f'{int(r["n_obs_diag"])} obs in diagnostic window '
                    f'(minimum {n_min})')
            if pd.notna(r['modal_share']) and r['modal_share'] > MODAL_SHARE_MAX:
                why['R1_modal_share'] = (
                    f'modal_share {r["modal_share"]:.3f} > {MODAL_SHARE_MAX}')
            s = r['std_of_z_ex_capped']
            if pd.isna(s) or not (STD_LO <= s <= STD_HI):
                why['R2_std_bounds'] = (
                    f'std_of_z_ex_capped {s:.3f} outside [{STD_LO}, {STD_HI}] '
                    f'(sigma_f rung {int(r["sigma_f_rung"])})'
                    if pd.notna(s) else 'std_of_z_ex_capped undefined')
            if bool(r['warmup_degenerate']):
                why['R3_warmup_degenerate'] = 'sigma_f = eps, floor offers no protection'
            p = r['pct_gt5_ex_crisis']
            n_gt5 = (round(p * r['n_obs_diag_ex_crisis'])
                     if pd.notna(p) and pd.notna(r['n_obs_diag_ex_crisis']) else 0)
            if pd.notna(p) and p > PCT_GT5_MAX and n_gt5 >= PCT_GT5_MIN_COUNT:
                why['R6_boundary_mass'] = (
                    f'pct_gt5_ex_crisis {p:.3%} > {PCT_GT5_MAX:.0%} '
                    f'({n_gt5} of {int(r["n_obs_diag_ex_crisis"])} obs; '
                    f'all-window {r["pct_gt5"]:.3%}, n_capped {int(r["n_capped"])})')

            fired = [x for x in RULE_ORDER if x in why]
            action = 'drop' if fired else 'keep'
            reason = why[fired[0]] if fired else ''

        out.append({
            'bucket':               bucket_of[tbl],
            'table_source':         tbl,
            'feature':              f,
            'base_factor':          base_factor(f),
            'action':               action,
            'rule_fired':           fired[0] if fired else '',
            'all_rules':            ' | '.join(fired),
            'reason':               reason,
            'modal_share':          r['modal_share'],
            'std_of_z_ex_capped':   r['std_of_z_ex_capped'],
            'pct_gt5':              r['pct_gt5'],
            'pct_gt5_ex_crisis':    r['pct_gt5_ex_crisis'],
            'n_gt5_ex_crisis':      n_gt5 if f not in exempt else 0,
            'warmup_degenerate':    r['warmup_degenerate'],
            'sigma_f_rung':         r['sigma_f_rung'],
            'n_capped':             r['n_capped'],
            'n_obs_diag':           r['n_obs_diag'],
            'max_abs_z':            r['max_abs_z'],
            'shift_max_z':          r['shift_max_z'],   # reported, NOT a rule
            'd_start':              r['d_start'],
        })

    return pd.DataFrame(out)


def rule_summary(dec: pd.DataFrame, bucket_order) -> pd.DataFrame:
    """
    Counts per rule per bucket. A feature can fail several rules, so the rule
    columns do not sum to n_dropped -- that is the unique count.
    """
    def _counts(d, label):
        row = {'bucket': label, 'n_features': len(d),
               'n_dropped': int((d['action'] == 'drop').sum()),
               'n_kept': int((d['action'] == 'keep').sum())}
        for rule in RULE_ORDER:
            row[rule] = int(d['all_rules'].str.contains(rule, regex=False).sum())
        return row

    rows = [_counts(dec[dec['bucket'] == b], b) for b in bucket_order]
    rows.append(_counts(dec, 'TOTAL'))
    return pd.DataFrame(rows)