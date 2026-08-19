"""
lib/normalise.py
================
One robust expanding z-score, shared by the aggregate and panel pipelines.

Called eight times with identical constants -- aggregate daily/weekly/monthly
for both feature sets, and panel daily/monthly. That identity is what makes the
panel-vs-aggregate spline comparison legitimate: both terminate on the same
bounded grid, reached the same way.

The pooling dimension is just "however many rows share this date" -- 1 for the
aggregate and macro, ~100 for the panel. The Welford batch merge treats
batch_n = 1 no differently from batch_n = 100, so no special-casing is needed.

TWO REGIMES
-----------
WARM-UP   batch, two-pass, per feature, over its own leading block.
          Emits NO z-scores. Exists only to build a clean ruler.

RUNNING   streaming, one Welford merge per date.
          R2  denom = max(expanding_std, floor, eps)     read BEFORE updating
          R3  EMIT   z = (x - mean) / denom              UNCAPPED
          R4  CAP    x_cap = clip(x, mean +/- 10*denom)
          R5  UPDATE Welford with x_cap

R3 before R5 is the causality guarantee: the moments normalising date t contain
nothing from date t.

R4 is the two-step "trim z to +/-10, convert back to raw" written as one
operation, because mean +/- 10*denom IS the raw value at z = +/-10. When
|z| <= 10 the clip is a no-op, so the "only cap what breaks 10" case falls out
automatically.

THE EMITTED Z IS UNCAPPED
-------------------------
The cap touches only what feeds the moments. Every in-between z-score is
returned for every feature on every date, so |z| > 10 in the output IS the
record of which contributions were capped. No separate flag is kept.

This function does NOT clip to +/-5. That happens once, in the assembly step,
on a copy -- so the saved z-score files stay uncapped and remain diagnosable.

THE DIAGNOSTIC WINDOW  (added)
------------------------------
The z-scores are computed over EVERY row. The per-feature REPORT, however, is
measured only over rows inside [diag_start, diag_end].

Reason: the report is what decides which features enter the models, and the
models are evaluated on 2020-2024. If the report is measured over the full
sample then test-period data has selected the feature set. It is label-free so
the leak is mild, but it is indefensible once someone names it and it costs
nothing to prevent.

The window to pass is the EARLIEST split's training window (Split A, ending
2015-12-31), because a feature set frozen there is clean for all four splits.
Split B trains to 2019-12-31, which sits inside Split A's test period, so
Split B's window would not be.

`crisis_start` / `crisis_end` mark the NBER recession (2007-12 to 2009-06).
They affect ONE statistic, `pct_gt5_ex_crisis`, so that a feature which
produced |z| > 5 legitimately during the crisis is not mistaken for a feature
with an unstable denominator. They deliberately do NOT affect
`std_of_z_ex_capped`: removing high-|z| observations lowers the measured std
and would push extra features below the 0.5 bound -- the opposite of intended.

WHY CAPPED-CONTRIBUTION RATHER THAN ONE-PASS
--------------------------------------------
Both were run on this project, accidentally:

  one-pass, no cap   old aggregate Stage 3   12 zombie features
                                             open_vs_mid  std/sigma_f = 13,429
                                             ChInvIA      std/sigma_f = 4.1e11
  capped + floor     panel notebook          0 features crushed

Worked example. sigma_f = 1, and at date 100 one observation arrives at 10,000.

  One-pass: that point enters the moments. With n ~ 100 the expanding std
  becomes ~10,000/sqrt(100) = 1,000, so for the next twenty years an ordinary
  move of 1 receives z = 0.001. SILENT DEATH -- no NaN, no extreme, no clipping,
  and no threshold rule can detect it.

  Capped: at date 100 the denominator is still ~1, so z = 10,000 -- loud and
  visible. Its CONTRIBUTION is capped, the std barely moves, and date 101 onward
  behaves normally.

The cap does not make bad features good. open_vs_mid came back from the panel at
max|z| = 946,447. It converts a dead feature into an obviously broken one, which
a threshold rule then catches.

Why 10 and not 5: the two numbers answer different questions. +/-10 is how much
a single observation is allowed to influence the RULER; +/-5 (applied later, in
assembly) is how far the MODEL is allowed to see. The gap is deliberate --
letting genuine turmoil widen the denominator is what makes a resulting z of 5
mean "among the most extreme readings this feature has ever produced" rather
than "somewhat unusual". Cap at 5 and the std learns nothing from real crises,
stays too tight, and z = 5 fires on ordinary days. Above ~10, a crisis WEEK -- a
cluster, and variance weights the square -- can still reset the std and
re-poison subsequent years.

Accepted consequence: because the std responds to realised turmoil, the raw->z
mapping is not perfectly time-stationary. A deliberate trade of responsiveness
for stationarity, and one for the limitations section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST SCALE -- THE FOUR-RUNG LADDER
# ═══════════════════════════════════════════════════════════════════════════════

def robust_scale_ladder(v: np.ndarray, eps: float = 1e-8) -> tuple[float, int]:
    """
    Robust scale sigma_f, with the rung that produced it.

    Every rung is a quantile, so no rung can be shifted by an extreme value: a
    -8.7e12 datum is merely "the smallest value", and half the data would have
    to be corrupted to move a median. Neither an ordinary std nor a mean
    absolute deviation can be used -- both are destroyed by the very values
    sigma_f exists to bound. This is the ONLY reason the MAD appears: in the
    warm-up there are no prior moments to cap against, so a monster would poison
    the seed directly.

      rung 1  1.4826 * median(|x - med|).  The normal case.
      rung 2  Fires when the MAD is exactly zero -- more than half the values
              identical, so the median absolute deviation vanishes. Substitutes
              P95(|dev|)/1.9600, where 1.9600 is P95 of |N(0,1)| so the result
              still reads as a standard deviation. ~15 features in panel monthly
              need this: ConvDebt (89.1% ties), ShareRepurchase (83.8%).
      rung 3  Fires when P95 is also zero (>95% ties). MAD of the strictly
              positive deviations -- spread measured using only the observations
              that moved. Fails SAFE: a 99%-constant feature with a few enormous
              movers gets a LARGE sigma_f, so the floor dominates and it
              contributes z ~ 0, which is correct for something that barely
              moves. Falling through to eps would fail DANGEROUS.
      rung 4  Genuinely constant. eps, a numerical catch only.

    For rung-2 and rung-3 features sigma_f is a rough proxy scale rather than a
    calibrated standard deviation -- a zero-inflated distribution is nowhere near
    Gaussian, so the 1.9600 equivalence does not really hold. Acceptable, since
    sigma_f only feeds the denominator floor and the warm-up trim, both loose
    safety rails.
    """
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return eps, 4

    dev = np.abs(v - np.median(v))

    s = 1.4826 * float(np.median(dev))
    if s > 1e-12:
        return max(s, eps), 1

    s = float(np.percentile(dev, 95)) / 1.9600
    if s > 1e-12:
        return max(s, eps), 2

    pos = dev[dev > 1e-12]
    if len(pos):
        s = 1.4826 * float(np.median(pos))
        if s > 1e-12:
            return max(s, eps), 3

    return eps, 4


def _shift_max(v: np.ndarray, years: np.ndarray) -> float:
    """
    (max - min of period medians) / sigma_f, over thirds of the timeline.

    Max-minus-min rather than last-minus-first because tga went 100 -> 1,800
    -> 20, which a monotone test misses entirely.

    Used in two places:
      - on Z-SCORES in the report, where near-zero means the expanding mean kept
        up with whatever the raw level did;
      - on RAW values in the review notebook (exposed as shift_max_stat), where
        it is the rule-4 test for a trending level.
    """
    ok = np.isfinite(v)
    if ok.sum() < 30:
        return np.nan
    v, y = v[ok], years[ok]

    yrs = np.unique(y)
    if len(yrs) < 5:
        return np.nan
    k = max(len(yrs) // 3, 1)
    blocks = (yrs[:k], yrs[k:-k] if len(yrs) > 2 * k else yrs[k:k + 1], yrs[-k:])

    meds = [np.median(v[np.isin(y, b)]) for b in blocks if len(b)]
    sf = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float((max(meds) - min(meds)) / sf) if sf > 1e-12 else np.nan


def _shift_max_safe(v: np.ndarray, years: np.ndarray) -> float:
    """
    (max - min of period medians) / max(sf, 1.0), over thirds of the timeline.

    Floors the robust scale at 1.0 so that crushed/collapsed variance does not
    artificially inflate the drift score. Only meaningful on z-scores, where a
    healthy scale is ~1.0.
    """
    ok = np.isfinite(v)
    if ok.sum() < 30:
        return np.nan
    v, y = v[ok], years[ok]

    yrs = np.unique(y)
    if len(yrs) < 5:
        return np.nan
    k = max(len(yrs) // 3, 1)
    blocks = (yrs[:k], yrs[k:-k] if len(yrs) > 2 * k else yrs[k:k + 1], yrs[-k:])

    meds = [np.median(v[np.isin(y, b)]) for b in blocks if len(b)]
    if not meds:
        return np.nan

    sf = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    sf_safe = max(sf, 1.0)
    return float((max(meds) - min(meds)) / sf_safe)


# Public alias. The review notebook uses this on RAW columns for rule 4, and on
# full-sample z-scores for the post-freeze drift limitation.
shift_max_stat = _shift_max


# ═══════════════════════════════════════════════════════════════════════════════
# THE Z-SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def robust_expanding_zscore(
    df: pd.DataFrame,
    feature_cols: list,
    date_col: str = 'date',
    min_dates: int = 252,
    sigma_refresh_every: int | None = None,
    cap: float = 10.0,
    floor_mult: float = 0.1,
    eps: float = 1e-8,
    diag_start: str | None = None,
    diag_end: str | None = None,
    crisis_start: str | None = None,
    crisis_end: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df with features replaced by UNCAPPED z-scores, per-feature report).

    Z-scores are computed over ALL rows. The report is measured only over rows
    in [diag_start, diag_end] -- see the module docstring. If those are None the
    report covers everything, which is the old behaviour.
    """
    if sigma_refresh_every is None:
        sigma_refresh_every = min_dates

    # kind='stable' so rows within a date keep input order, making row order
    # deterministic. Load-bearing in three places: raw_matrix, the date slices,
    # and the r_start indexing in the sigma_f refresh -- all derive from this
    # one sorted frame.
    df = df.sort_values(date_col, kind='stable').reset_index(drop=True).copy()
    for c in feature_cols:
        if hasattr(df[c].dtype, 'numpy_dtype'):
            df[c] = df[c].astype('float64')

    raw = df[feature_cols].to_numpy(dtype=np.float64)
    n_rows, n_feat = raw.shape
    z_out = np.full_like(raw, np.nan)

    dates = df[date_col].to_numpy()

    # ── DIAGNOSTIC AND CRISIS MASKS ───────────────────────────────────────────
    dser = pd.to_datetime(df[date_col])
    diag = np.ones(n_rows, dtype=bool)
    if diag_start is not None:
        diag &= (dser >= pd.Timestamp(diag_start)).to_numpy()
    if diag_end is not None:
        diag &= (dser <= pd.Timestamp(diag_end)).to_numpy()

    crisis = np.zeros(n_rows, dtype=bool)
    if crisis_start is not None and crisis_end is not None:
        crisis = ((dser >= pd.Timestamp(crisis_start)) &
                  (dser <= pd.Timestamp(crisis_end))).to_numpy()

    # modal_share on the RAW values, restricted to the diagnostic window so it
    # is measured on the same rows as everything else in the report.
    modal_share = np.full(n_feat, np.nan)
    for j in range(n_feat):
        col = raw[diag, j]
        col = col[np.isfinite(col)]
        if len(col):
            modal_share[j] = pd.Series(col).value_counts().iloc[0] / len(col)

    codes, uniq = pd.factorize(df[date_col], sort=True)
    n_dates = len(uniq)

    # Per-date row slices. Contiguity is asserted rather than assumed -- a
    # non-contiguous date would make the [start, end) slice swallow other dates.
    slices = []
    for d in range(n_dates):
        r = np.flatnonzero(codes == d)
        assert len(r) == r[-1] - r[0] + 1, \
            f"date index {d} occupies non-contiguous rows"
        slices.append((r[0], r[-1] + 1))

    if verbose:
        print(f"    {n_feat} features, {n_dates:,} dates, {n_rows:,} rows "
              f"({n_rows / n_dates:.1f} rows/date)")
        print(f"    z-scored over ALL rows; report measured over "
              f"{diag.sum():,} rows "
              f"[{diag_start or 'start'} .. {diag_end or 'end'}]")
        if crisis.any():
            print(f"    crisis rows excluded from pct_gt5_ex_crisis: "
                  f"{int((diag & crisis).sum()):,} "
                  f"[{crisis_start} .. {crisis_end}]")

    # ══════════════════════════════════════════════════════════════════════════
    # PER-FEATURE WARM-UP START
    # ══════════════════════════════════════════════════════════════════════════
    # Counted in DATES WITH DATA, not rows and not calendar dates. This is what
    # handles late-starting features: retail flow is all-NaN through 2004-05, so
    # it starts counting in 2006. And in the weekly union table each feature
    # updates once a week while the table holds ~3 rows/week, so 52 of ITS OWN
    # observations takes ~208 rows.
    #
    # NaN scattered INSIDE the warm-up is fine. Every warm-up statistic skips
    # NaN, so a gap simply means one fewer observation contributing. Nothing
    # requires the NaN to be contiguous.
    has_data = np.zeros((n_dates, n_feat), dtype=bool)
    for d, (a, b) in enumerate(slices):
        has_data[d] = np.isfinite(raw[a:b]).any(axis=0)

    cum = np.cumsum(has_data, axis=0)
    reached = cum[-1] >= min_dates
    warm_end = np.where(reached, np.argmax(cum >= min_dates, axis=0), n_dates)

    n_never = int((~reached).sum())
    if verbose and n_never:
        print(f"    {n_never} feature(s) never reach {min_dates} dates with "
              f"data -> all-NaN")

    # ══════════════════════════════════════════════════════════════════════════
    # WARM-UP: batch, two-pass, per feature over its own block
    # ══════════════════════════════════════════════════════════════════════════
    w_n = np.zeros(n_feat)
    w_mean = np.zeros(n_feat)
    w_M2 = np.zeros(n_feat)
    sigma_f = np.full(n_feat, eps)
    rung = np.full(n_feat, -1, dtype=int)
    n_trim = np.zeros(n_feat, dtype=int)
    max_z_warm = np.full(n_feat, np.nan)

    for j in range(n_feat):
        if not reached[j]:
            continue
        end = slices[warm_end[j]][1]          # inclusive of the reaching date
        block = raw[:end, j]
        v = block[np.isfinite(block)]
        if len(v) < 2:
            continue

        # W1 -- robust centre and scale, both quantile-based
        med = float(np.median(v))
        sf, rg = robust_scale_ladder(v, eps)
        sigma_f[j], rung[j] = sf, rg

        dev = np.abs(v - med) / sf
        max_z_warm[j] = float(dev.max())

        # W2 -- trim. Equivalently: clip z to +/-cap and convert back to raw.
        lo, hi = med - cap * sf, med + cap * sf
        clean = np.clip(block, lo, hi)
        clean = np.where(np.isfinite(block), clean, np.nan)
        n_trim[j] = int(((block < lo) | (block > hi)).sum())

        # W3 -- seed Welford. SEEDED, not restarted empty, so the first
        # post-warm-up observation is z-scored against a fully built ruler.
        cv = clean[np.isfinite(clean)]
        w_n[j] = len(cv)
        w_mean[j] = float(cv.mean())
        w_M2[j] = float(cv.var(ddof=1) * (len(cv) - 1)) if len(cv) > 1 else 0.0

    # W4 -- floor
    floor = floor_mult * sigma_f

    if verbose:
        for r in (2, 3, 4):
            k = int((rung[reached] == r).sum())
            if k:
                lab = {2: 'P95/1.96, >50% ties', 3: 'MAD of movers, >95% ties',
                       4: 'eps, constant'}[r]
                print(f"    sigma_f rung {r}: {k:>4} feature(s)  ({lab})")
        print(f"    warm-up trimmed at +/-{cap:.0f} robust sigma: "
              f"{int(n_trim.sum()):,} values")

    # ══════════════════════════════════════════════════════════════════════════
    # RUNNING
    # ══════════════════════════════════════════════════════════════════════════
    n_capped = np.zeros(n_feat, dtype=int)
    n_refresh = 0
    last_refresh = 0
    start = int(warm_end[reached].min()) + 1 if reached.any() else n_dates

    for d in range(start, n_dates):
        r0, r1 = slices[d]
        x = raw[r0:r1]
        nan = ~np.isfinite(x)
        active = (warm_end < d) & reached              # (n_feat,)
        if not active.any():
            continue

        # R1 -- annual sigma_f refresh from RAW rows strictly before this date.
        #
        # From raw values, not z-scores: sigma_f answers "what is this feature's
        # natural spread in its own units", the only question for which
        # 0.1*sigma_f is a sensible floor. From z-scores it would return ~0.674
        # (the MAD of a standard normal) for every feature regardless of scale --
        # a meaningless near-constant, and circular.
        #
        # Annual rather than per-date because a MAD has no O(1) online update:
        # it needs the sorted history. Per-date expanding MAD is O(N^2) and
        # infeasible; ~20 recomputes is near-linear. The floor is a
        # slowly-varying safety anchor, not a signal-bearing statistic, so
        # annual resolution is ample.
        if d - last_refresh >= sigma_refresh_every:
            for j in np.flatnonzero(active):
                v = raw[:r0, j]
                v = v[np.isfinite(v)]
                if len(v) >= 2:
                    sigma_f[j], _ = robust_scale_ladder(v, eps)
            floor = floor_mult * sigma_f
            last_refresh = d
            n_refresh += 1

        # R2 -- denominator, read BEFORE this date's update. Three layers:
        #   expanding std  the real ruler
        #   floor          catches the std collapsing on a temporarily-flat
        #                  feature -- signal in the numerator, tiny denominator,
        #                  the genuine amplification danger
        #   eps            catches sigma_f ITSELF collapsing on a permanently
        #                  flat feature, which the floor cannot since it is
        #                  built from sigma_f
        # eps cannot reintroduce the blow-up: it only ever divides when the
        # numerator is also ~0, since sigma_f ~ 0 means essentially constant, so
        # x - mean ~ 0 and ~0/1e-8 ~ 0 -- correct for something that never varies.
        var = np.where(w_n > 1, w_M2 / np.maximum(w_n - 1.0, 1.0), 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        denom = np.maximum(np.maximum(std, floor), eps)

        # R3 -- EMIT, uncapped
        z = (x - w_mean) / denom
        z[nan] = np.nan
        z[:, ~active] = np.nan
        z_out[r0:r1] = z

        # R4 -- CAP the contribution. clip(x, mean +/- cap*denom) is exactly
        # "clip z to +/-cap then convert back to raw", and is a no-op when
        # |z| <= cap.
        dev = x - w_mean
        beyond = (np.abs(dev) > cap * denom) & (~nan) & active
        n_capped += beyond.sum(axis=0)
        x_cap = np.where(beyond, w_mean + np.sign(dev) * cap * denom, x)
        x_cap = np.where(nan, np.nan, x_cap)

        # R5 -- UPDATE. Welford parallel merge, one per date.
        #
        # batch_n is PER FEATURE, never a shared row count: feature NaN patterns
        # differ, so a shared scalar would weight the merge wrongly.
        upd = (~nan) & active
        batch_n = upd.sum(axis=0).astype(np.float64)
        live = batch_n > 0

        bsum = np.nansum(np.where(upd, x_cap, np.nan), axis=0)
        bmean = np.where(live, bsum / np.maximum(batch_n, 1.0), 0.0)
        bdev = np.where(upd, x_cap - bmean, 0.0)
        bM2 = np.nansum(bdev ** 2, axis=0)

        comb = w_n + batch_n
        safe = np.where(comb > 0, comb, 1.0)
        delta = np.where(live, bmean - w_mean, 0.0)

        w_mean = np.where(live, w_mean + delta * batch_n / safe, w_mean)
        w_M2 = np.where(live,
                        w_M2 + bM2 + delta ** 2 * w_n * batch_n / safe,
                        w_M2)
        w_n = np.where(live, comb, w_n)

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE BACK
    # ══════════════════════════════════════════════════════════════════════════
    # Positional assignment is valid ONLY because z_out was built in the row
    # order of this same stably-sorted frame. Assert the key is untouched so the
    # invariant is checked rather than reasoned about.
    keys = df[date_col].copy()
    df[feature_cols] = z_out
    assert df[date_col].equals(keys), "row order changed during write-back"

    # ══════════════════════════════════════════════════════════════════════════
    # REPORT  -- measured on the diagnostic window only
    # ══════════════════════════════════════════════════════════════════════════
    years = pd.DatetimeIndex(dates).year.to_numpy()
    rep = []
    for j, c in enumerate(feature_cols):
        v = z_out[:, j]
        finite = np.isfinite(v)

        vv = v[finite & diag]                      # diagnostic-window z-scores
        vnc = v[finite & diag & ~crisis]           # ... excluding NBER recession
        ordinary = vv[np.abs(vv) <= cap]           # the un-capped observations

        v_diag = np.where(diag, v, np.nan)         # for the drift statistics

        rep.append({
            'feature': c,
            'reached_warmup': bool(reached[j]),
            'modal_share': float(modal_share[j]),
            # sigma_f = eps means the warm-up block was entirely constant (or
            # had <2 valid values), so the floor is 0.1*eps and offers no
            # protection. The first genuine move then divides by eps and
            # explodes -- loudly, which is the intent, and the cap recovers the
            # ruler immediately after. Flagged so it is explicit rather than
            # inferred from a large max_abs_z.
            'warmup_degenerate': bool(reached[j] and rung[j] == 4),
            'sigma_f': float(sigma_f[j]),
            'sigma_f_rung': int(rung[j]),
            'd_start': (str(pd.Timestamp(uniq[warm_end[j]]).date())
                        if reached[j] else ''),
            'n_obs_all': int(finite.sum()),
            'n_obs_diag': int(len(vv)),
            'n_obs_diag_ex_crisis': int(len(vnc)),
            'n_warmup_trimmed': int(n_trim[j]),
            'max_abs_z_warmup': float(max_z_warm[j]),
            'n_capped': int(n_capped[j]),
            # std_of_z over ALL emitted z. Dominated by any single extreme:
            # one z of 1e6 among 500 ordinary values gives std ~ 45,000
            # regardless of how well the cap worked. Kept because a large value
            # is itself a flag, but NOT the measure of whether the cap worked.
            'std_of_z': float(vv.std(ddof=1)) if len(vv) > 1 else np.nan,
            # THIS is the "did the cap work" number. std over the observations
            # that were NOT capped, i.e. the ordinary ones. A correct expanding
            # z-score gives ~1. Far below means the denominator is still
            # inflated and the feature is being crushed; far above means the
            # scale is unstable. Verified on a synthetic series with one 1e6
            # spike: std_of_z = 45,154 but std_of_z_ex_capped = 0.90.
            # NOTE: crisis rows are deliberately INCLUDED here.
            'std_of_z_ex_capped': (float(ordinary.std(ddof=1))
                                   if len(ordinary) > 1 else np.nan),
            'mean_of_z': float(vv.mean()) if len(vv) else np.nan,
            'max_z': float(vv.max()) if len(vv) else np.nan,
            'min_z': float(vv.min()) if len(vv) else np.nan,
            'max_abs_z': float(np.abs(vv).max()) if len(vv) else np.nan,
            'pct_gt3': float((np.abs(vv) > 3).mean()) if len(vv) else np.nan,
            'pct_gt5': float((np.abs(vv) > 5).mean()) if len(vv) else np.nan,
            # The rule-6 statistic. Crisis rows removed so that a feature which
            # responded correctly to 2008-09 is not mistaken for one with an
            # unstable denominator.
            'pct_gt5_ex_crisis': (float((np.abs(vnc) > 5).mean())
                                  if len(vnc) else np.nan),
            'pct_gt10': float((np.abs(vv) > 10).mean()) if len(vv) else np.nan,
            'pct_gt20': float((np.abs(vv) > 20).mean()) if len(vv) else np.nan,
            'pct_gt50': float((np.abs(vv) > 50).mean()) if len(vv) else np.nan,
            'pct_gt100': float((np.abs(vv) > 100).mean()) if len(vv) else np.nan,
            # Measured, REPORTED, never acted on. Drift between early and late
            # sample is drift between train and test; dropping on it is
            # selection on test-period distribution. See 05_review cell 7 for
            # the full-sample version, computed after the feature set is frozen.
            'shift_max_z': _shift_max(v_diag, years),
            'shift_max_z_safe': _shift_max_safe(v_diag, years),
        })
    rep = pd.DataFrame(rep)

    if verbose:
        ok = rep[rep['reached_warmup']]
        print(f"    contributions capped at +/-{cap:.0f}: "
              f"{int(rep['n_capped'].sum()):,}   "
              f"sigma_f refreshes: {n_refresh}")
        n_deg = int(rep['warmup_degenerate'].sum())
        if n_deg:
            print(f"    ** {n_deg} feature(s) constant through warm-up "
                  f"(sigma_f = eps, no floor protection) **")
            for f in rep.loc[rep['warmup_degenerate'], 'feature'].head(8):
                print(f"       {f}")
        if len(ok):
            s = ok['std_of_z_ex_capped']
            print(f"    std_of_z_ex_capped: median {s.median():.3f}   "
                  f"<0.5 {int((s < 0.5).sum())}   >2 {int((s > 2).sum())}")
            print(f"    max|z|: median {ok['max_abs_z'].median():.1f}   "
                  f"worst {ok['max_abs_z'].max():.1f} "
                  f"({ok.loc[ok['max_abs_z'].idxmax(), 'feature']})")
            print(f"    pct_gt5 median {ok['pct_gt5'].median():.4%}   "
                  f"ex-crisis {ok['pct_gt5_ex_crisis'].median():.4%}")

    return df, rep