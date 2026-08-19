"""
Evaluation Utilities
====================
Metrics, backtesting, prediction I/O, calibration, and comparison tables.
Handles both binary and continuous target types.

Usage:
    from evaluation import evaluate_model, save_predictions, print_comparison_table

Design decisions:

    CONTINUOUS TARGET IS Z-SCORED (minret_5d_z):
        The continuous target is an expanding z-score of minret_5d, NOT the
        raw decimal return. This has two consequences:

        1. derive_binary_from_continuous must NOT threshold predictions at
           -0.02. That threshold is only meaningful in raw return space.
           Instead, derived AUC uses -y_pred as a ranking score against
           pre-computed y_binary labels (which are still computed from raw
           returns in 01_prepare_datasets.py and stored in the parquet).
           AUC is rank-based, so ranking by -z_pred is identical to ranking
           by -raw_pred since z-scoring is a monotone transform.

        2. save_predictions must NOT compute y_true_binary by thresholding
           y_true at -0.02. y_true is now a z-score. The binary labels are
           loaded separately from the parquet (y_binary column).

    DERIVED AUC PROTOCOL:
        For continuous models, derived AUC is computed as:
            roc_auc_score(y_binary, -y_pred_z)
        where y_binary comes from the DataLoader (the target when evaluate_model
        is called without y_true_binary) or is passed explicitly.
        The defensive check in derive_binary_from_continuous detects already-
        binary inputs {0, 1} and uses them directly, avoiding re-thresholding.

    R² IS THE PRIMARY CONTINUOUS METRIC:
        training.py early-stops on R². evaluation.py reports R² as the headline
        metric for continuous models. MSE is also reported for completeness and
        is directly comparable to Ridge/polymodel (which use MSELoss).

    BINARY METRICS NEVER CHANGE:
        y_binary is still (minret_5d < -0.02) computed in prepare_datasets.py.
        Binary models, binary labels, and AUC for binary models are unaffected
        by the z-scoring of the continuous target.

    BACKTESTING PHILOSOPHY (see the BACKTEST section for detail):
        minret_5d is a 5-day-forward left-tail exceedance probability. Two
        structural facts drive the strategy design:

        1. Overlapping windows → consecutive predictions are highly
           autocorrelated → the raw signal whipsaws around any threshold.
           Signal smoothing (EMA) filters transient spikes.

        2. It is a RISK measure, not a return forecast. The natural use is
           risk-scaled exposure (reduce when tail risk is high) rather than
           binary in/out timing.

        Crashes arrive suddenly; recoveries are gradual. The asymmetric
        strategy de-risks fast and re-risks slowly, which also cuts turnover
        on the costly re-entry side.

        ALL backtests model transaction costs as turnover-proportional.
        A strategy that only survives at 0 bps is not credible; the
        cost-sensitivity table exposes this directly.

        Sharpe AND Sortino are both reported. Sortino only penalises downside
        deviation, which matches a tail-risk-avoidance overlay whose explicit
        goal is to cut downside while preserving upside.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.calibration import calibration_curve
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Classification metrics for binary crash prediction.

    y_true : 0/1 labels  (minret_5d < -0.02, computed in prepare_datasets.py)
    y_prob : predicted probabilities in [0, 1]  (sigmoid of model logit)
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {
        "auc":            auc,
        "brier":          brier_score_loss(y_true, y_prob),
        "log_loss":       log_loss(y_true, y_prob_clipped),
        "n_samples":      len(y_true),
        "n_positive":     int(y_true.sum()),
        "base_rate":      float(y_true.mean()),
        "mean_pred_prob": float(y_prob.mean()),
    }


def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Regression metrics for continuous z-scored minret prediction.

    y_true : actual minret_5d_z values  (expanding z-score, NOT raw decimal)
    y_pred : predicted minret_5d_z values

    R² and MSE are computed in z-score space. MSE is directly comparable
    across neural network models (all trained on the same z-scored target).
    Ridge and polymodel also report MSE but computed against raw decimal
    targets — those numbers are on a different scale and must not be
    directly compared. R² is scale-invariant and IS directly comparable.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse    = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / max(ss_tot, 1e-10)
    mae    = float(mean_absolute_error(y_true, y_pred))

    return {
        "mse":            mse,
        "r2":             r2,
        "mae":            mae,
        "n_samples":      len(y_true),
        "mean_actual":    float(y_true.mean()),
        "mean_predicted": float(y_pred.mean()),
        "pred_std":       float(y_pred.std()),
    }


def derive_binary_from_continuous(
    y_true: np.ndarray,
    y_pred_z: np.ndarray,
) -> dict:
    """
    Derive binary classification metrics from continuous z-scored predictions.

    IMPORTANT: y_pred_z contains z-scores, NOT raw decimal returns.
    Thresholding at -0.02 in z-score space is NOT done here — it is
    meaningless since z-scores are not in return units.

    Instead:
        - y_true_binary is obtained either from the passed y_true directly
          (if already binary {0, 1}) or derived by the caller passing actual
          binary labels. The defensive check handles both cases.
        - AUC is computed using -y_pred_z as the ranking score. More negative
          z-score = more negative predicted return = higher crash risk.
          AUC is rank-based, so this gives identical results to ranking by
          -y_pred_raw (since z-scoring is a monotone increasing transform).

    Hard-threshold accuracy is NOT computed here because there is no
    meaningful z-score threshold equivalent to -0.02 in return space
    without knowing the expanding mean and std (which vary by date).

    Parameters
    ----------
    y_true    : either binary 0/1 labels OR z-scored true values.
                Defensive check detects which case applies.
    y_pred_z  : z-scored predictions from a continuous model.
    """
    y_true    = np.asarray(y_true).ravel()
    y_pred_z  = np.asarray(y_pred_z).ravel()

    # ── Defensive check: detect already-binary input ──
    # If y_true is already {0, 1} binary labels, use them directly.
    # If y_true contains z-scores or raw returns, this should not happen —
    # callers should always pass binary labels (from data["y_{part}"]) when
    # calling evaluate_model for continuous targets. The check prevents the
    # old bug where binary labels were re-thresholded at -0.02.
    unique_vals = set(np.unique(y_true))
    if unique_vals <= {0.0, 1.0}:
        # Already binary — use directly
        y_true_binary = y_true
    else:
        # Unexpected: continuous values passed as y_true.
        # This should not happen with the current pipeline.
        # Fall back to treating as z-scores (cannot threshold meaningfully)
        # and raise a warning rather than silently producing wrong results.
        import warnings
        warnings.warn(
            "derive_binary_from_continuous received non-binary y_true values "
            f"(range [{y_true.min():.2f}, {y_true.max():.2f}]). "
            "Expected binary 0/1 labels from data['y_{part}']. "
            "Derived AUC may be incorrect.",
            stacklevel=2,
        )
        y_true_binary = y_true  # proceed with whatever was passed

    # ── Ranking score: more negative z-prediction = higher crash risk ──
    crash_score = -y_pred_z

    try:
        auc = (roc_auc_score(y_true_binary, crash_score)
               if len(np.unique(y_true_binary)) > 1 else 0.5)
    except ValueError:
        auc = 0.5

    return {
        "derived_auc":       auc,
        "derived_base_rate": float(y_true_binary.mean()),
        "derived_pred_mean": float(y_pred_z.mean()),
        "derived_pred_std":  float(y_pred_z.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_type: str = "binary",
    y_true_binary: np.ndarray = None,
) -> dict:
    """
    Run model on a DataLoader and compute all metrics.

    Parameters
    ----------
    target_type : str
        "binary"     → targets in loader are 0/1 labels
        "continuous" → targets in loader are z-scored minret_5d_z values

    y_true_binary : np.ndarray, optional
        For continuous target: the binary 0/1 crash labels for derived AUC.
        Pass data["y_{part}"] explicitly, OR leave as None.

        If None: targets from the loader are used. Since loader targets are
        z-scores, the defensive check in derive_binary_from_continuous will
        detect they are NOT binary and warn. To avoid this, always pass
        y_true_binary explicitly for continuous models:

            metrics = evaluate_model(
                model, loaders["test"], device, "continuous",
                y_true_binary=data["y_test"],
            )

    Returns
    -------
    dict with:
        Binary:     y_prob, auc, brier, log_loss, n_samples, ...
        Continuous: y_pred, mse, r2, mae, derived_auc, pred_std, ...
    """
    model.eval()
    all_outputs = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_outputs.append(outputs.cpu())
        all_targets.append(y_batch)

    outputs = torch.cat(all_outputs).numpy().ravel()
    targets = torch.cat(all_targets).numpy().ravel()

    if target_type == "binary":
        y_prob  = 1.0 / (1.0 + np.exp(-outputs))  # sigmoid
        metrics = compute_binary_metrics(targets, y_prob)
        metrics["y_true"] = targets
        metrics["y_prob"]  = y_prob

    else:
        # Continuous: outputs and targets are both z-scored minret_5d_z
        y_pred  = outputs
        metrics = compute_continuous_metrics(targets, y_pred)

        # Derived binary metrics require actual binary labels.
        # Use y_true_binary if provided; otherwise pass targets (z-scores)
        # and let the defensive check handle it (with a warning).
        true_for_binary = y_true_binary if y_true_binary is not None else targets
        derived = derive_binary_from_continuous(true_for_binary, y_pred)
        metrics.update(derived)

        metrics["y_true"] = targets   # z-scored
        metrics["y_pred"] = y_pred    # z-scored

    return metrics


def evaluate_sklearn_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    target_type: str = "binary",
    y_true_binary: np.ndarray = None,
) -> dict:
    """
    Evaluate a sklearn model (Ridge, LogisticRegression, etc.).

    Binary:     uses predict_proba → AUC, Brier, log_loss
    Continuous: uses predict → MSE, R², derived AUC
                NOTE: sklearn continuous models predict in z-score space
                (since they are trained on minret_5d_z targets).
                Pass y_true_binary for correct derived AUC.
    """
    if target_type == "binary":
        y_prob  = model.predict_proba(X)[:, 1]
        metrics = compute_binary_metrics(y, y_prob)
        metrics["y_true"] = y
        metrics["y_prob"]  = y_prob
    else:
        y_pred  = model.predict(X)
        metrics = compute_continuous_metrics(y, y_pred)
        true_for_binary = y_true_binary if y_true_binary is not None else y
        derived = derive_binary_from_continuous(true_for_binary, y_pred)
        metrics.update(derived)
        metrics["y_true"] = y
        metrics["y_pred"] = y_pred

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data for reliability diagrams.

    Only meaningful for binary targets (y_prob ∈ [0, 1]).
    Do not call for continuous model outputs (z-scores are not probabilities).

    Returns dict with bin_means, bin_true, bin_counts, ece.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    try:
        bin_true, bin_means = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        return {"bin_means": [], "bin_true": [], "bin_counts": [], "ece": np.nan}

    bin_edges          = np.linspace(0, 1, n_bins + 1)
    bin_counts         = np.histogram(y_prob, bins=bin_edges)[0]
    nonempty           = bin_counts > 0
    bin_counts_matched = bin_counts[nonempty][:len(bin_means)]

    weights = bin_counts_matched / bin_counts_matched.sum()
    ece     = float(np.sum(weights * np.abs(bin_true - bin_means)))

    return {
        "bin_means":  bin_means.tolist(),
        "bin_true":   bin_true.tolist(),
        "bin_counts": bin_counts_matched.tolist(),
        "ece":        ece,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST: SHARED HELPERS (risk metrics, smoothing, costs)
# ═══════════════════════════════════════════════════════════════════════════════

def _annualised_risk_metrics(
    strategy_returns: np.ndarray,
    target_return: float = 0.0,
    ann_factor: float = np.sqrt(252),
    periods_per_year: int = 252,
) -> dict:
    """
    Compute Sharpe, Sortino, annualised return, and drawdown from a
    series of realised (net-of-cost) strategy returns.

    Sharpe  : mean / std of ALL returns  (penalises up and down alike)
    Sortino : mean / downside deviation  (penalises only returns below
              target_return — the metric that matches a tail-risk overlay,
              whose goal is to cut downside while preserving upside).

    Downside deviation uses the standard "square, sum over ALL periods,
    divide by total N" convention (not the count of down days only). This
    is the conventional definition and is stable when down days are rare.

    Guards:
        std == 0            → Sharpe  = 0.0
        downside_dev == 0   → Sortino = nan  (undefined; no down days at all —
                              nan keeps JSON valid and is skipped by the
                              param search's np.isfinite check)
    """
    strategy_returns = np.asarray(strategy_returns).ravel()
    n = len(strategy_returns)

    mean_r = strategy_returns.mean()
    std_r  = strategy_returns.std()

    # ── Sharpe ──
    sharpe = (mean_r / std_r * ann_factor) if std_r > 0 else 0.0

    # ── Sortino (downside deviation over all periods) ──
    downside = np.minimum(strategy_returns - target_return, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev > 1e-12:
        sortino = (mean_r - target_return) / downside_dev * ann_factor
    else:
        # No downside deviation at all — Sortino is undefined. Return nan
        # rather than inf: inf serialises to invalid JSON ("Infinity") and
        # poisons any mean over splits. nan is handled by pd.isna in the
        # comparison tables and by the np.isfinite guard in the param search.
        # (For a daily strategy over years this branch is essentially never
        # hit — there is almost always at least one down day.)
        sortino = float("nan")

    # ── Drawdown ──
    cum    = np.cumprod(1 + strategy_returns)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(((cum - peak) / peak).min()) if n > 0 else 0.0

    return {
        "sharpe":            float(sharpe),
        "sortino":           float(sortino),
        "annual_return":     float(mean_r * periods_per_year),
        "max_drawdown":      max_dd,
        "cumulative_return": float(cum[-1] - 1) if n > 0 else 0.0,
        "volatility_annual": float(std_r * ann_factor),
    }


def _turnover_metrics(
    exposure: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    """
    Turnover diagnostics from a sequence of daily exposures.

    total_turnover     : sum of |Δexposure| over the whole period
    annual_turnover    : turnover scaled to per-year (how many times you
                         effectively churn your full position each year)
    n_trades           : number of days on which exposure changed at all
    avg_holding_days   : n_days / n_trades — mean number of days between
                         position changes. This is a clean holding-period
                         proxy for the BINARY strategy (piecewise-constant
                         exposure). For the CONTINUOUS risk-scaled strategy
                         exposure changes almost daily, so n_trades ≈ n and
                         this value collapses toward 1 — read annual_turnover
                         instead as the churn measure there.
    """
    exposure = np.asarray(exposure).ravel()
    n = len(exposure)
    if n < 2:
        return {"total_turnover": 0.0, "annual_turnover": 0.0,
                "n_trades": 0, "avg_holding_days": float(n)}

    changes        = np.abs(np.diff(exposure))
    total_turnover = float(changes.sum())
    annual_turnover = total_turnover / n * periods_per_year
    n_trades       = int((changes > 1e-9).sum())
    avg_holding    = n / max(n_trades, 1)

    return {
        "total_turnover":   total_turnover,
        "annual_turnover":  float(annual_turnover),
        "n_trades":         n_trades,
        "avg_holding_days": float(avg_holding),
    }


def smooth_signal(signal: np.ndarray, span: int = 1, method: str = "ema") -> np.ndarray:
    """
    Smooth an autocorrelated signal before it drives position decisions.

    Because minret_5d uses OVERLAPPING 5-day windows, consecutive
    predictions share 4 days of horizon and are highly autocorrelated.
    A single-day spike is often noise; a sustained elevation is a real
    regime shift. Smoothing filters transient spikes and cuts whipsaw.

    Parameters
    ----------
    span : int
        EMA span (or rolling window length). span=1 is a no-op (raw signal).
    method : str
        "ema"  → exponential moving average (pandas .ewm(span=...))
        "sma"  → simple rolling mean
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if span <= 1:
        return signal

    s = pd.Series(signal)
    if method == "sma":
        out = s.rolling(window=span, min_periods=1).mean()
    else:
        out = s.ewm(span=span, adjust=False).mean()
    return out.to_numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST 1: BINARY MARKET TIMING (with transaction costs)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_timing(
    returns: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    go_cash_when: str = "above",
    cost_bps: float = 3.0,
    smooth_span: int = 5,
    smooth_method: str = "ema",
    target_return: float = 0.0,
) -> dict:
    """
    Simple binary market-timing backtest: fully invested or fully in cash,
    with transaction costs on each switch.

    This is the straightforward "go to cash when risk is high" strategy —
    the buy-and-hold-with-an-off-switch baseline. It is the honest simple
    comparator for the asymmetric risk-scaled strategy below.

    Parameters
    ----------
    returns : daily returns (target_daily_return: return on day t+1)
    signal  : model output at each row
        Binary:     y_prob (probability of crash) — use go_cash_when="above"
        Continuous: y_pred_z (z-scored prediction) — use go_cash_when="below"
    threshold : decision boundary (applied to the SMOOTHED signal)
        Binary:     e.g. 0.3 (go cash if P(crash) > 0.3)
        Continuous: e.g. -1.5 (go cash if predicted z < -1.5)
    go_cash_when : "above" → cash when signal > threshold (binary: high prob)
                   "below" → cash when signal < threshold (continuous: low z)
    cost_bps : transaction cost in basis points per unit turnover.
               Cost on day t = cost_bps/1e4 * |position[t] - position[t-1]|.
               A full 0→1 or 1→0 switch costs cost_bps/1e4.
    smooth_span : EMA/SMA span applied to the signal before thresholding.
                  Filters whipsaw from the overlapping-window autocorrelation.

    Returns
    -------
    dict with Sharpe, Sortino, annual return, drawdown, turnover, pct invested,
    plus the same risk metrics for buy-and-hold as benchmark.
    """
    returns = np.asarray(returns).ravel()
    signal  = np.asarray(signal).ravel()

    sig = smooth_signal(signal, span=smooth_span, method=smooth_method)

    # ── Binary position: invested (1) or cash (0) ──
    if go_cash_when == "above":
        invested = (sig <= threshold).astype(float)   # cash when signal high
    else:
        invested = (sig >= threshold).astype(float)   # cash when signal low

    # ── Transaction costs on position changes ──
    # prepend=0.0 charges the strategy for establishing its initial position
    # on day one, consistent with the buy-and-hold benchmark below (which is
    # also charged a one-off entry cost). Using prepend=invested[0] instead
    # would make day-one entry free and unfairly favour the strategy.
    position_change      = np.abs(np.diff(invested, prepend=0.0))
    cost                 = cost_bps / 1e4 * position_change
    strategy_returns     = invested * returns - cost

    # ── Metrics ──
    risk     = _annualised_risk_metrics(strategy_returns, target_return=target_return)
    turnover = _turnover_metrics(invested)

    # ── Buy-and-hold benchmark (also cost-charged for the initial entry) ──
    bh_returns = returns.copy()
    bh_returns[0] -= cost_bps / 1e4          # one-off cost to establish position
    bh_risk = _annualised_risk_metrics(bh_returns, target_return=target_return)

    return {
        "strategy":           "binary_timing",
        "sharpe":             risk["sharpe"],
        "sortino":            risk["sortino"],
        "annual_return":      risk["annual_return"],
        "max_drawdown":       risk["max_drawdown"],
        "cumulative_return":  risk["cumulative_return"],
        "volatility_annual":  risk["volatility_annual"],
        "pct_invested":       float(invested.mean()),
        "avg_exposure":       float(invested.mean()),
        "annual_turnover":    turnover["annual_turnover"],
        "n_trades":           turnover["n_trades"],
        "avg_holding_days":   turnover["avg_holding_days"],
        "cost_bps":           cost_bps,
        "smooth_span":        smooth_span,
        # Benchmark
        "buy_hold_sharpe":       bh_risk["sharpe"],
        "buy_hold_sortino":      bh_risk["sortino"],
        "buy_hold_drawdown":     bh_risk["max_drawdown"],
        "buy_hold_cumulative":   bh_risk["cumulative_return"],
        "buy_hold_annual":       bh_risk["annual_return"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST 2: ASYMMETRIC RISK-SCALED EXPOSURE (with transaction costs)
# ═══════════════════════════════════════════════════════════════════════════════

def signal_to_target_exposure(
    sig: np.ndarray,
    calm_threshold: float,
    stress_threshold: float,
    go_cash_when: str = "above",
    floor: float = 0.0,
    cap: float = 1.0,
) -> np.ndarray:
    """
    Piecewise-linear map from a (smoothed) signal to a TARGET exposure.

    Full exposure (cap) when risk is calm, floor exposure when risk is
    stressed, linearly interpolated in between. This is the continuous
    risk-scaling that fits minret_5d's nature as a risk estimate rather
    than a binary event flag.

    go_cash_when="above" (binary prob, higher = riskier):
        sig <= calm_threshold      → exposure = cap
        sig >= stress_threshold    → exposure = floor
        between                    → linear ramp down

    go_cash_when="below" (continuous z, lower = riskier):
        sig >= calm_threshold      → exposure = cap
        sig <= stress_threshold    → exposure = floor
        between                    → linear ramp down
    """
    sig = np.asarray(sig, dtype=float).ravel()

    if go_cash_when == "above":
        # Higher signal = more risk. calm < stress.
        lo, hi = calm_threshold, stress_threshold
        # fraction of risk in [0,1]: 0 at/below calm, 1 at/above stress
        frac = (sig - lo) / max(hi - lo, 1e-9)
    else:
        # Lower signal = more risk. calm > stress.
        hi, lo = calm_threshold, stress_threshold
        # fraction of risk in [0,1]: 0 at/above calm, 1 at/below stress
        frac = (hi - sig) / max(hi - lo, 1e-9)

    frac      = np.clip(frac, 0.0, 1.0)
    exposure  = cap - frac * (cap - floor)
    return np.clip(exposure, floor, cap)


def backtest_risk_scaled(
    returns: np.ndarray,
    signal: np.ndarray,
    calm_threshold: float,
    stress_threshold: float,
    go_cash_when: str = "below",
    alpha_down: float = 1.0,
    alpha_up: float = 0.2,
    floor: float = 0.0,
    cap: float = 1.0,
    cost_bps: float = 3.0,
    smooth_span: int = 5,
    smooth_method: str = "ema",
    target_return: float = 0.0,
) -> dict:
    """
    Asymmetric risk-scaled exposure strategy.

    The strategy that actually fits minret_5d: continuous exposure that
    shrinks as predicted tail risk rises, with ASYMMETRIC adjustment speed.

    Layer 1: smooth the signal (EMA) to filter overlapping-window whipsaw.
    Layer 2: map smoothed signal → target exposure (piecewise linear).
    Layer 3: move ACTUAL exposure toward target asymmetrically —
             fast when de-risking (alpha_down ≈ 1), slow when re-risking
             (alpha_up small). Crashes arrive suddenly, recoveries are
             gradual; this also cuts turnover on the costly re-entry side.
    Layer 4: realise returns with turnover-proportional transaction costs.

    Parameters
    ----------
    calm_threshold, stress_threshold : bounds of the exposure ramp, in
        signal units (probability for binary, z-score for continuous).
    alpha_down : de-risking speed in (0, 1]. 1.0 = jump straight to a lower
        target the moment risk rises.
    alpha_up : re-risking speed in (0, 1]. Small = require the low-risk
        signal to persist before re-entering.
    floor, cap : min / max exposure (cap can exceed 1.0 for leverage).
    cost_bps : transaction cost per unit turnover (bps).
    smooth_span : EMA/SMA span for the signal.

    Returns
    -------
    dict with Sharpe, Sortino, annual return, drawdown, avg exposure,
    turnover, and buy-and-hold benchmark metrics.
    """
    returns = np.asarray(returns).ravel()
    signal  = np.asarray(signal).ravel()
    n       = len(returns)

    sig    = smooth_signal(signal, span=smooth_span, method=smooth_method)
    target = signal_to_target_exposure(
        sig, calm_threshold, stress_threshold,
        go_cash_when=go_cash_when, floor=floor, cap=cap,
    )

    # ── Asymmetric adjustment of actual exposure toward target ──
    actual = np.empty(n, dtype=float)
    prev   = cap  # start fully invested (conservative: costs the first de-risk)
    for t in range(n):
        tgt = target[t]
        if tgt < prev:
            # de-risking: move fast
            cur = prev + alpha_down * (tgt - prev)
        else:
            # re-risking: move slow
            cur = prev + alpha_up * (tgt - prev)
        actual[t] = cur
        prev = cur

    # ── Transaction costs on exposure changes ──
    # prepend=0.0 charges for establishing the initial exposure on day one,
    # consistent with the buy-and-hold benchmark (also charged a one-off
    # entry cost). Note the strategy starts at prev=cap internally for the
    # asymmetric adjustment, but the COST is measured from a flat book so the
    # day-one entry is not free.
    position_change  = np.abs(np.diff(actual, prepend=0.0))
    cost             = cost_bps / 1e4 * position_change
    strategy_returns = actual * returns - cost

    # ── Metrics ──
    risk     = _annualised_risk_metrics(strategy_returns, target_return=target_return)
    turnover = _turnover_metrics(actual)

    bh_returns = returns.copy()
    bh_returns[0] -= cost_bps / 1e4
    bh_risk = _annualised_risk_metrics(bh_returns, target_return=target_return)

    return {
        "strategy":           "risk_scaled_asymmetric",
        "sharpe":             risk["sharpe"],
        "sortino":            risk["sortino"],
        "annual_return":      risk["annual_return"],
        "max_drawdown":       risk["max_drawdown"],
        "cumulative_return":  risk["cumulative_return"],
        "volatility_annual":  risk["volatility_annual"],
        "avg_exposure":       float(actual.mean()),
        "min_exposure":       float(actual.min()),
        "max_exposure":       float(actual.max()),
        "annual_turnover":    turnover["annual_turnover"],
        "n_trades":           turnover["n_trades"],
        "avg_holding_days":   turnover["avg_holding_days"],
        "cost_bps":           cost_bps,
        "smooth_span":        smooth_span,
        "alpha_down":         alpha_down,
        "alpha_up":           alpha_up,
        # Benchmark
        "buy_hold_sharpe":     bh_risk["sharpe"],
        "buy_hold_sortino":    bh_risk["sortino"],
        "buy_hold_drawdown":   bh_risk["max_drawdown"],
        "buy_hold_cumulative": bh_risk["cumulative_return"],
        "buy_hold_annual":     bh_risk["annual_return"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COST SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def cost_sensitivity_table(
    returns: np.ndarray,
    signal: np.ndarray,
    backtest_fn,
    backtest_kwargs: dict,
    cost_bps_grid=(0.0, 1.0, 3.0, 5.0, 10.0),
) -> pd.DataFrame:
    """
    Run a backtest across a grid of transaction-cost assumptions.

    A strategy that only works at 0 bps is not credible; one that survives
    5-10 bps is. This table is a first-class robustness result.

    Parameters
    ----------
    backtest_fn : backtest_timing OR backtest_risk_scaled
    backtest_kwargs : all kwargs for that function EXCEPT cost_bps
    cost_bps_grid : cost levels to sweep

    Returns
    -------
    DataFrame indexed by cost_bps with net Sharpe, Sortino, annual return,
    drawdown, turnover, and avg exposure.
    """
    rows = []
    for c in cost_bps_grid:
        res = backtest_fn(returns, signal, cost_bps=c, **backtest_kwargs)
        rows.append({
            "cost_bps":        c,
            "sharpe":          res["sharpe"],
            "sortino":         res["sortino"],
            "annual_return":   res["annual_return"],
            "max_drawdown":    res["max_drawdown"],
            "annual_turnover": res["annual_turnover"],
            "avg_exposure":    res["avg_exposure"],
        })
    return pd.DataFrame(rows).set_index("cost_bps")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD / PARAMETER SEARCH (net-of-cost, on validation)
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(
    returns: np.ndarray,
    signal: np.ndarray,
    go_cash_when: str = "above",
    thresholds: np.ndarray = None,
    cost_bps: float = 3.0,
    smooth_span: int = 5,
    objective: str = "sortino",
    target_return: float = 0.0,
) -> dict:
    """
    Grid search for the binary-timing threshold that maximises NET-OF-COST
    Sortino (or Sharpe) on the validation set.

    Run on VALIDATION only. Apply the chosen threshold to test.

    The objective defaults to Sortino because the strategy is a tail-risk
    overlay whose goal is to cut downside, not total variance. Costs are
    charged so the search does not reward frictionless whipsaw.
    """
    if thresholds is None:
        if go_cash_when == "above":
            thresholds = np.arange(0.05, 0.95, 0.05)   # probability grid
        else:
            # z-score grid — predictions sit in roughly [-5, +2], so the
            # "go to cash" threshold is searched across the negative range
            # where crash risk lives, from mild to severe.
            # Base grid: -4.5 to -0.25 in steps of 0.25.
            # Extra values added just above -0.25 for compressed signals
            # (e.g. polymodel AVA) whose test signal barely dips below 0.
            base = np.arange(-4.5, 0.0, 0.25)          # [..., -0.50, -0.25]
            extra = np.array([-0.125, 0.0, 0.1])        # targeted additions
            thresholds = np.unique(np.concatenate([base, extra]))

    # Seed with a full backtest at the middle threshold so the returned dict
    # always contains every metric key, even if no candidate improves on it
    # (e.g. all scores nan). Prevents callers from KeyError-ing on "sharpe".
    mid_t = float(thresholds[len(thresholds) // 2])
    best = {"threshold": mid_t, **backtest_timing(
        returns, signal, threshold=mid_t, go_cash_when=go_cash_when,
        cost_bps=cost_bps, smooth_span=smooth_span, target_return=target_return,
    )}
    best_score = best[objective] if np.isfinite(best[objective]) else -np.inf

    for t in thresholds:
        result = backtest_timing(
            returns, signal, threshold=float(t), go_cash_when=go_cash_when,
            cost_bps=cost_bps, smooth_span=smooth_span, target_return=target_return,
        )
        score = result[objective]
        # nan Sortino (no downside days) cannot be ranked — skip via isfinite.
        if np.isfinite(score) and score > best_score:
            best = {"threshold": float(t), **result}
            best_score = score

    return best


def find_best_risk_scaled_params(
    returns: np.ndarray,
    signal: np.ndarray,
    go_cash_when: str = "below",
    calm_grid=None,
    stress_grid=None,
    smooth_grid=(1, 3, 5, 10),
    alpha_down: float = 1.0,
    alpha_up: float = 0.2,
    cost_bps: float = 3.0,   # default matches global standard
    objective: str = "sortino",
    target_return: float = 0.0,
) -> dict:
    """
    Coarse grid search for the risk-scaled strategy on VALIDATION.

    Searches (calm_threshold, stress_threshold, smooth_span) jointly with
    fixed asymmetric speeds (alpha_down, alpha_up) by default — five free
    parameters on a single market-level series risks overfitting the
    backtest, so keep the speeds fixed at sensible defaults and search
    only the ramp + smoothing. Optimises NET-OF-COST Sortino.

    calm/stress grids default to sensible ranges per signal type.
    """
    # Default grids per signal type.
    # Continuous predictions sit in roughly [-5, +2]. The exposure ramp runs
    # from calm (full exposure, signal high/positive) down to stress (floor
    # exposure, signal very negative). Grids span that actual range:
    #   calm  ∈ [ 0.0, +2.0]  — start cutting exposure once signal dips here
    #   stress∈ [-4.5, -1.0]  — reach floor exposure by here
    if go_cash_when == "below":   # continuous z-score
        if calm_grid   is None: calm_grid   = np.arange(0.0, 2.01, 0.5)
        if stress_grid is None: stress_grid = np.arange(-4.5, -0.99, 0.5)
    else:                         # binary probability
        if calm_grid   is None: calm_grid   = np.arange(0.10, 0.41, 0.10)
        if stress_grid is None: stress_grid = np.arange(0.40, 0.81, 0.10)

    best = None
    best_score = -np.inf

    for span in smooth_grid:
        for calm in calm_grid:
            for stress in stress_grid:
                # Validity: for "below", calm must be > stress; for "above", calm < stress
                if go_cash_when == "below" and not (calm > stress):
                    continue
                if go_cash_when == "above" and not (calm < stress):
                    continue

                result = backtest_risk_scaled(
                    returns, signal,
                    calm_threshold=float(calm),
                    stress_threshold=float(stress),
                    go_cash_when=go_cash_when,
                    alpha_down=alpha_down, alpha_up=alpha_up,
                    cost_bps=cost_bps, smooth_span=int(span),
                    target_return=target_return,
                )
                candidate = {
                    "calm_threshold":   float(calm),
                    "stress_threshold": float(stress),
                    "smooth_span":      int(span),
                    **result,
                }
                # Seed best with the first valid config so the returned dict
                # always carries every metric key, then improve on it. nan
                # Sortino (no downside days) can't be ranked — skip via isfinite.
                score = result[objective]
                if best is None:
                    best = candidate
                if np.isfinite(score) and score > best_score:
                    best = candidate
                    best_score = score

    if best is None:
        raise ValueError(
            "No valid (calm, stress) combination found — check that the grids "
            "contain at least one pair satisfying the calm/stress ordering."
        )
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_predictions(
    model_name: str,
    split_name: str,
    target_type: str,
    part: str,
    dates: pd.Series,
    returns: np.ndarray,
    metrics: dict,
    hyperparameters: dict = None,
    results_dir: Path = None,
    y_true_binary: np.ndarray = None,
):
    """
    Save predictions and metrics for one model × split × part.

    File naming: {model_name}_{target_type}_{split_name}_{part}.parquet
    E.g.: sparse_kan_binary_Split_A_test.parquet

    For continuous models, y_true_binary must be passed explicitly so
    that the saved parquet contains the correct binary crash labels
    (NOT derived by thresholding z-scores at -0.02, which is wrong).

    Parameters
    ----------
    y_true_binary : np.ndarray, optional
        Required for continuous target_type. Pass data["y_{part}"].
        If None for continuous, y_true_binary column is omitted from parquet.
    """
    if results_dir is None:
        raise ValueError("results_dir must be specified")

    results_dir = Path(results_dir)
    prefix      = f"{model_name}_{target_type}_{split_name}_{part}"

    # ── Build prediction DataFrame ──
    pred_data = {
        "date":         pd.Series(dates).values,
        "daily_return": np.asarray(returns).ravel(),
    }

    if target_type == "binary":
        pred_data["y_true"]  = metrics["y_true"]
        pred_data["y_prob"]  = metrics["y_prob"]

    else:
        # y_true and y_pred are both z-scored
        pred_data["y_true_z"] = metrics["y_true"]   # z-scored actual
        pred_data["y_pred"]   = metrics["y_pred"]   # z-scored prediction

        # Binary labels stored separately for derived AUC evaluation
        # Must come from data["y_{part}"], NOT from thresholding z-scores
        if y_true_binary is not None:
            pred_data["y_true_binary"] = np.asarray(y_true_binary).ravel()
        # If y_true_binary is None, column is omitted — derived AUC won't
        # be recomputable from the parquet alone, but metrics JSON has it.

    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_data).to_parquet(
        pred_dir / f"{prefix}.parquet", index=False
    )

    # ── Save metrics JSON ──
    metrics_clean = {
        k: v for k, v in metrics.items()
        if not isinstance(v, np.ndarray)
    }
    if hyperparameters is not None:
        metrics_clean["hyperparameters"] = hyperparameters

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{prefix}.json", "w") as f:
        json.dump(metrics_clean, f, indent=2, default=str)


def load_predictions(
    model_name: str,
    split_name: str,
    target_type: str,
    part: str = "test",
    results_dir: Path = None,
) -> dict:
    """Load saved predictions and metrics."""
    results_dir = Path(results_dir)
    prefix      = f"{model_name}_{target_type}_{split_name}_{part}"

    pred_df = pd.read_parquet(
        results_dir / "predictions" / f"{prefix}.parquet"
    )
    with open(results_dir / "metrics" / f"{prefix}.json") as f:
        metrics = json.load(f)

    return {"predictions": pred_df, "metrics": metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_comparison_data(
    model_names: list,
    split_names: list,
    target_type: str,
    results_dir: Path,
    part: str = "test",
    metric_key: str = "auc",
) -> pd.DataFrame:
    """
    Load metrics across all models and splits into a DataFrame.

    For continuous models with metric_key="auc", automatically maps to
    "derived_auc" so cross-target-type comparison tables work cleanly.
    """
    results_dir = Path(results_dir)
    rows = []

    for model in model_names:
        for split in split_names:
            prefix       = f"{model}_{target_type}_{split}_{part}"
            metrics_path = results_dir / "metrics" / f"{prefix}.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)

                if target_type == "continuous" and metric_key == "auc":
                    value = m.get("derived_auc")
                else:
                    value = m.get(metric_key)

                rows.append({"model": model, "split": split, "value": value})
            else:
                rows.append({"model": model, "split": split, "value": None})

    return pd.DataFrame(rows)


def make_comparison_table(
    model_names: list,
    split_names: list,
    target_type: str,
    results_dir: Path,
    metric_key: str = "auc",
) -> pd.DataFrame:
    """Pivot table: models (rows) × splits (columns) + mean column."""
    df    = load_comparison_data(
        model_names, split_names, target_type, results_dir, metric_key=metric_key
    )
    pivot = df.pivot(index="model", columns="split", values="value")
    pivot = pivot.reindex(model_names)
    pivot["Mean"] = pivot[split_names].mean(axis=1)
    return pivot


def print_comparison_table(
    model_names: list,
    split_names: list,
    target_type: str,
    results_dir: Path,
    metric_key: str = "auc",
):
    """Print a formatted comparison table."""
    table = make_comparison_table(
        model_names, split_names, target_type, results_dir, metric_key
    )
    print(f"\n  {metric_key.upper()} ({target_type}) — Test Set")
    print(f"  {'Model':<28}", end="")
    for s in split_names:
        print(f"  {s:>10}", end="")
    print(f"  {'Mean':>10}")
    print("  " + "-" * (28 + 12 * (len(split_names) + 1)))

    for model in model_names:
        print(f"  {model:<28}", end="")
        for s in split_names:
            val = table.loc[model, s] if model in table.index else None
            if val is not None and not pd.isna(val):
                print(f"  {val:>10.4f}", end="")
            else:
                print(f"  {'—':>10}", end="")
        mean_val = table.loc[model, "Mean"] if model in table.index else None
        if mean_val is not None and not pd.isna(mean_val):
            print(f"  {mean_val:>10.4f}")
        else:
            print(f"  {'—':>10}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN_FULL_BACKTEST — single call produces everything
# ═══════════════════════════════════════════════════════════════════════════════

# ── Global backtest defaults ──
# Change these once here to update every call in every notebook.
DEFAULT_COST_BPS   = 3.0          # transaction cost per unit turnover (bps)
DEFAULT_SMOOTH     = 5            # EMA span applied to signal before decisions
DEFAULT_COST_GRID  = (0.0, 1.0, 3.0, 5.0, 10.0)   # cost sensitivity sweep
DEFAULT_ALPHA_DOWN = 1.0          # de-risk speed  (1 = instant)
DEFAULT_ALPHA_UP   = 0.2          # re-risk speed  (0.2 = slow, ~7 days to re-enter)


def run_full_backtest(
    val_returns:   np.ndarray,
    val_signal:    np.ndarray,
    test_returns:  np.ndarray,
    test_signal:   np.ndarray,
    go_cash_when:  str   = "above",
    model_name:    str   = "",
    split_name:    str   = "",
    cost_bps:      float = DEFAULT_COST_BPS,
    smooth_span:   int   = DEFAULT_SMOOTH,
    cost_grid:     tuple = DEFAULT_COST_GRID,
    alpha_down:    float = DEFAULT_ALPHA_DOWN,
    alpha_up:      float = DEFAULT_ALPHA_UP,
    verbose:       bool  = True,
) -> dict:
    """
    Single call that produces the complete backtest output for one model ×
    split combination.  Called identically from every notebook.

    Parameters
    ----------
    val_returns, val_signal   : validation-set daily returns and model signal.
        Signal is y_prob for binary models (go_cash_when="above") or
        y_pred_z for continuous models (go_cash_when="below").
    test_returns, test_signal : same for the test set.
    go_cash_when : "above" for binary (high prob → cash),
                   "below" for continuous (low z-score → cash).
    model_name, split_name   : used in printed headers only.
    cost_bps   : central transaction-cost assumption (default 3.0 bps).
    smooth_span: EMA span applied to the signal (default 5 days).
    cost_grid  : tuple of cost levels for the sensitivity sweep.
    alpha_down : de-risk speed for risk-scaled strategy (default 1.0 = instant).
    alpha_up   : re-risk speed for risk-scaled strategy (default 0.2 = slow).

    What is printed
    ---------------
    1. Signal diagnostics  — raw and smoothed signal range, so you can see
       whether chosen thresholds sit sensibly within the signal distribution
       and are not pinned to grid edges.

    2. Simple binary timing (headline at default cost_bps / smooth_span)
       — threshold chosen on val (net-of-cost Sortino), applied to test.
       Columns: Threshold | Sharpe | Sortino | AnnRet | MaxDD | %Inv |
                Turn/yr | BH Sharpe | BH Sortino
       Special cases flagged: all-cash strategy (nan Sortino), threshold at
       grid edge (⚠edge).

    3. Simple timing cost sensitivity sweep
       — threshold re-chosen on val at each cost level (honest end-to-end).
       Columns: Cost(bps) | Threshold | Sharpe | Sortino | AnnRet |
                MaxDD | Turn/yr

    4. Asymmetric risk-scaled (headline at default cost_bps)
       — calm/stress thresholds and smoothing span found by grid search on val.
       Columns: Calm | Stress | Span | Sharpe | Sortino | AnnRet | MaxDD |
                AvgExp | Turn/yr | BH Sortino

    5. Risk-scaled cost sensitivity sweep
       — parameters re-found at each cost level.
       Columns: Cost(bps) | Calm | Stress | Span | Sharpe | Sortino |
                AnnRet | MaxDD | AvgExp | Turn/yr

    6. Side-by-side comparison
       — simple timing vs risk-scaled vs buy-and-hold at default cost_bps.
       Makes the relative performance immediately readable.

    Returns
    -------
    dict with keys:
        "simple"     : full backtest_timing result at default params
        "risk_scaled": full backtest_risk_scaled result at default params
        "buy_hold"   : buy-and-hold risk metrics at default cost_bps
        "simple_threshold"    : chosen threshold for simple strategy
        "simple_cost_table"   : DataFrame, cost sensitivity for simple
        "risk_scaled_params"  : dict, chosen params for risk-scaled strategy
        "risk_scaled_cost_table": DataFrame, cost sensitivity for risk-scaled
    """
    val_returns  = np.asarray(val_returns).ravel()
    val_signal   = np.asarray(val_signal).ravel()
    test_returns = np.asarray(test_returns).ravel()
    test_signal  = np.asarray(test_signal).ravel()

    label = f"{model_name}  {split_name}".strip()
    W = 72   # print width

    def _fmt(v, pct=False, dp=2):
        """Format a float, handling nan gracefully."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  nan  " if not pct else "  nan  "
        return f"{v:{'.0%' if pct else f'.{dp}f'}}"

    def _sortino_flag(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return " ← strategy sat in cash (no down days)"
        return ""

    # ── 1. SIGNAL DIAGNOSTICS ────────────────────────────────────────────────
    if verbose:
        print("\n" + "═" * W)
        print(f"  BACKTEST  {label}")
        print("═" * W)

        sm_val  = smooth_signal(val_signal,  span=smooth_span)
        sm_test = smooth_signal(test_signal, span=smooth_span)

        print(f"\n  Signal diagnostics (EMA span={smooth_span}):")
        print(f"    Val  raw : [{val_signal.min():+.3f}, {val_signal.max():+.3f}]   "
              f"smoothed: [{sm_val.min():+.3f}, {sm_val.max():+.3f}]")
        print(f"    Test raw : [{test_signal.min():+.3f}, {test_signal.max():+.3f}]   "
              f"smoothed: [{sm_test.min():+.3f}, {sm_test.max():+.3f}]")

    # ── 2. SIMPLE TIMING — headline ──────────────────────────────────────────
    best_simple = find_best_threshold(
        val_returns, val_signal,
        go_cash_when=go_cash_when,
        cost_bps=cost_bps,
        smooth_span=smooth_span,
        objective="sortino",
    )
    thr = best_simple["threshold"]

    bt_simple = backtest_timing(
        test_returns, test_signal,
        threshold=thr,
        go_cash_when=go_cash_when,
        cost_bps=cost_bps,
        smooth_span=smooth_span,
    )

    # Flag if threshold is at the grid edge
    if go_cash_when == "above":
        edge = thr >= 0.85 or thr <= 0.10
    else:
        edge = thr <= -4.25 or thr >= 0.05
    edge_flag = "  ⚠ threshold at grid edge — consider widening grid" if edge else ""

    if verbose:
        print(f"\n  ── Simple Binary Timing  (cost={cost_bps}bps, EMA={smooth_span}) ──")
        print(f"  Threshold chosen on val (net-of-cost Sortino): {thr:.2f}{edge_flag}")
        if np.isnan(bt_simple["sortino"]):
            print("  ⚠ Strategy sat entirely in cash on test set — "
                  "threshold never triggered.")
        print()
        hdr = (f"  {'Sharpe':>8} {'Sortino':>8} {'AnnRet':>8} {'MaxDD':>8} "
               f"{'%Inv':>7} {'Turn/yr':>8}  {'BH Shrp':>8} {'BH Srt':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        srt_flag = _sortino_flag(bt_simple["sortino"])
        print(f"  {bt_simple['sharpe']:>8.2f} "
              f"{bt_simple['sortino']:>8.2f} "
              f"{bt_simple['annual_return']:>8.1%} "
              f"{bt_simple['max_drawdown']:>8.1%} "
              f"{bt_simple['avg_exposure']:>6.1%} "
              f"{bt_simple['annual_turnover']:>8.2f}  "
              f"{bt_simple['buy_hold_sharpe']:>8.2f} "
              f"{bt_simple['buy_hold_sortino']:>8.2f}"
              f"{srt_flag}")

    # ── 3. SIMPLE TIMING — cost sensitivity ──────────────────────────────────
    simple_cost_rows = []
    for c in cost_grid:
        best_c = find_best_threshold(
            val_returns, val_signal,
            go_cash_when=go_cash_when,
            cost_bps=c,
            smooth_span=smooth_span,
            objective="sortino",
        )
        bt_c = backtest_timing(
            test_returns, test_signal,
            threshold=best_c["threshold"],
            go_cash_when=go_cash_when,
            cost_bps=c,
            smooth_span=smooth_span,
        )
        simple_cost_rows.append({
            "cost_bps":      c,
            "threshold":     best_c["threshold"],
            "sharpe":        bt_c["sharpe"],
            "sortino":       bt_c["sortino"],
            "annual_return": bt_c["annual_return"],
            "max_drawdown":  bt_c["max_drawdown"],
            "annual_turnover": bt_c["annual_turnover"],
            "avg_exposure":  bt_c["avg_exposure"],
        })
    simple_cost_df = pd.DataFrame(simple_cost_rows).set_index("cost_bps")

    if verbose:
        print(f"\n  Cost sensitivity — Simple Timing (threshold re-chosen at each cost):")
        print(f"  {'Cost(bps)':>10} {'Thr':>6} {'Sharpe':>8} {'Sortino':>8} "
              f"{'AnnRet':>8} {'MaxDD':>8} {'Turn/yr':>8}")
        print("  " + "-" * 64)
        for c, row in simple_cost_df.iterrows():
            srt = f"{row['sortino']:>8.2f}" if np.isfinite(row["sortino"]) else "     nan"
            print(f"  {c:>10.1f} {row['threshold']:>6.2f} {row['sharpe']:>8.2f} "
                  f"{srt} {row['annual_return']:>8.1%} "
                  f"{row['max_drawdown']:>8.1%} {row['annual_turnover']:>8.2f}")

    # ── 4. RISK-SCALED — headline ─────────────────────────────────────────────
    best_rs = find_best_risk_scaled_params(
        val_returns, val_signal,
        go_cash_when=go_cash_when,
        cost_bps=cost_bps,
        alpha_down=alpha_down,
        alpha_up=alpha_up,
        objective="sortino",
    )

    bt_rs = backtest_risk_scaled(
        test_returns, test_signal,
        calm_threshold=best_rs["calm_threshold"],
        stress_threshold=best_rs["stress_threshold"],
        go_cash_when=go_cash_when,
        alpha_down=alpha_down,
        alpha_up=alpha_up,
        cost_bps=cost_bps,
        smooth_span=best_rs["smooth_span"],
    )

    if verbose:
        print(f"\n  ── Asymmetric Risk-Scaled  "
              f"(cost={cost_bps}bps, αdown={alpha_down}, αup={alpha_up}) ──")
        print(f"  Params from val search: "
              f"calm={best_rs['calm_threshold']:.2f}  "
              f"stress={best_rs['stress_threshold']:.2f}  "
              f"span={best_rs['smooth_span']}")
        print()
        hdr2 = (f"  {'Sharpe':>8} {'Sortino':>8} {'AnnRet':>8} {'MaxDD':>8} "
                f"{'AvgExp':>7} {'Turn/yr':>8}  {'BH Shrp':>8} {'BH Srt':>8}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        print(f"  {bt_rs['sharpe']:>8.2f} "
              f"{bt_rs['sortino']:>8.2f} "
              f"{bt_rs['annual_return']:>8.1%} "
              f"{bt_rs['max_drawdown']:>8.1%} "
              f"{bt_rs['avg_exposure']:>6.1%} "
              f"{bt_rs['annual_turnover']:>8.2f}  "
              f"{bt_rs['buy_hold_sharpe']:>8.2f} "
              f"{bt_rs['buy_hold_sortino']:>8.2f}")

    # ── 5. RISK-SCALED — cost sensitivity ────────────────────────────────────
    rs_cost_rows = []
    for c in cost_grid:
        best_rs_c = find_best_risk_scaled_params(
            val_returns, val_signal,
            go_cash_when=go_cash_when,
            cost_bps=c,
            alpha_down=alpha_down,
            alpha_up=alpha_up,
            objective="sortino",
        )
        bt_rs_c = backtest_risk_scaled(
            test_returns, test_signal,
            calm_threshold=best_rs_c["calm_threshold"],
            stress_threshold=best_rs_c["stress_threshold"],
            go_cash_when=go_cash_when,
            alpha_down=alpha_down,
            alpha_up=alpha_up,
            cost_bps=c,
            smooth_span=best_rs_c["smooth_span"],
        )
        rs_cost_rows.append({
            "cost_bps":       c,
            "calm":           best_rs_c["calm_threshold"],
            "stress":         best_rs_c["stress_threshold"],
            "span":           best_rs_c["smooth_span"],
            "sharpe":         bt_rs_c["sharpe"],
            "sortino":        bt_rs_c["sortino"],
            "annual_return":  bt_rs_c["annual_return"],
            "max_drawdown":   bt_rs_c["max_drawdown"],
            "annual_turnover":bt_rs_c["annual_turnover"],
            "avg_exposure":   bt_rs_c["avg_exposure"],
        })
    rs_cost_df = pd.DataFrame(rs_cost_rows).set_index("cost_bps")

    if verbose:
        print(f"\n  Cost sensitivity — Risk-Scaled (params re-chosen at each cost):")
        print(f"  {'Cost(bps)':>10} {'Calm':>6} {'Stress':>7} {'Span':>5} "
              f"{'Sharpe':>8} {'Sortino':>8} {'AnnRet':>8} {'MaxDD':>8} "
              f"{'AvgExp':>7} {'Turn/yr':>8}")
        print("  " + "-" * 80)
        for c, row in rs_cost_df.iterrows():
            srt = f"{row['sortino']:>8.2f}" if np.isfinite(row["sortino"]) else "     nan"
            print(f"  {c:>10.1f} {row['calm']:>6.2f} {row['stress']:>7.2f} "
                  f"{int(row['span']):>5d} {row['sharpe']:>8.2f} "
                  f"{srt} {row['annual_return']:>8.1%} "
                  f"{row['max_drawdown']:>8.1%} "
                  f"{row['avg_exposure']:>6.1%} "
                  f"{row['annual_turnover']:>8.2f}")

    # ── 6. SIDE-BY-SIDE COMPARISON ───────────────────────────────────────────
    # Buy-and-hold at the default cost_bps
    bh_returns = test_returns.copy()
    bh_returns[0] -= cost_bps / 1e4
    bh = _annualised_risk_metrics(bh_returns)

    if verbose:
        print(f"\n  ── Side-by-Side Comparison  (cost={cost_bps}bps) ──")
        print()
        print(f"  {'Strategy':<22} {'Sharpe':>8} {'Sortino':>8} {'AnnRet':>8} "
              f"{'MaxDD':>8} {'AvgExp':>7} {'Turn/yr':>8}")
        print("  " + "-" * 72)

        def _row(name, d, exp, turn):
            srt = f"{d['sortino']:>8.2f}" if np.isfinite(d["sortino"]) else "     nan"
            print(f"  {name:<22} {d['sharpe']:>8.2f} {srt} "
                  f"{d['annual_return']:>8.1%} {d['max_drawdown']:>8.1%} "
                  f"{exp:>6.1%} {turn:>8.2f}")

        _row("Simple timing",
             bt_simple, bt_simple["avg_exposure"], bt_simple["annual_turnover"])
        _row("Risk-scaled asymmetric",
             bt_rs, bt_rs["avg_exposure"], bt_rs["annual_turnover"])
        _row("Buy & Hold",
             bh, 1.0, cost_bps / 1e4)   # b&h always 100% invested, trivial turn

        print()

    return {
        "simple":                bt_simple,
        "simple_threshold":      thr,
        "simple_cost_table":     simple_cost_df,
        "risk_scaled":           bt_rs,
        "risk_scaled_params":    {
            "calm_threshold":  best_rs["calm_threshold"],
            "stress_threshold": best_rs["stress_threshold"],
            "smooth_span":     best_rs["smooth_span"],
        },
        "risk_scaled_cost_table": rs_cost_df,
        "buy_hold":              bh,
    }