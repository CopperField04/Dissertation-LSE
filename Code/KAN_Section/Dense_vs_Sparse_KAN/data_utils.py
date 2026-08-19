"""
Data Utilities
==============
Loads prepared splits and creates PyTorch DataLoaders.
Foundation module -- every experiment notebook imports from here.

Usage:
    from data_utils import get_dataloaders, get_device, load_theme_assignment

    device = get_device()
    loaders = get_dataloaders(
        split_name="Split_A",
        dataset="agg_full_moments",   # or "agg_means", "panel"
        target_type="binary",         # or "continuous"
        batch_size=128,
        splits_dir=SPLITS_DIR,
    )

    for X_batch, y_batch in loaders["train"]:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        ...

Design decisions:

    NO DEFAULT PATHS. Every call must specify splits_dir (and, for the
    taxonomy loader, themes_dir) explicitly. This avoids silent path
    resolution bugs when moving between local and Colab environments --
    unchanged from the original design.

    THREE DATASETS, ONE FUNCTION. The old pipeline distinguished only
    "full_moments" vs "means_only", both drawn from a single combined
    aggregate table. The current pipeline produces three genuinely
    different tables -- agg_means, agg_full_moments, panel -- with
    different meta columns (panel carries permno/dlyret/dlycap, the
    aggregate tables don't) and different file locations. `dataset` is
    now the single argument that selects both, matching the file stem
    written by Stage_4_Assembly/05_splits.ipynb exactly
    ({dataset}_{part}.parquet), so there is no separate name-mapping
    layer to keep in sync with the pipeline.

    PANEL META. panel_{part}.parquet carries permno, date, dlyret,
    dlycap, minret_5d_pct, y_binary, then features. The extra ID/weight
    columns are why META_COLS could not stay a single flat set -- it is
    now DATASETS[dataset]['meta'], defined per dataset.

    CONTINUOUS TARGET IS NOW A RAW PERCENTAGE, NOT A Z-SCORE. The old
    target minret_5d_z was an expanding z-score (mean ~0, std ~1). The
    current target minret_5d_pct is 100 * min(next 5 trading-day returns),
    e.g. -2.3 meaning -2.3%. This is NOT a rescale of the old target --
    the z-score divided out the crisis level via a time-varying mean/std,
    the percentage keeps it. Old and new continuous-target R^2 values are
    NOT directly comparable across pipelines. Downstream code that assumed
    predictions live on a roughly [-5, +2] scale (e.g. fixed backtest
    threshold grids) needs revisiting -- see evaluation.py, deferred.

    minret_raw_{part} is REMOVED. It existed only because the old pipeline
    kept both a z-scored and a raw-decimal copy of the same target. There
    is now exactly one continuous target column (minret_5d_pct), so
    keeping a second "raw" copy under a different name would just be the
    same array twice.

    RETURN COLUMN NAME DIFFERS BY DATASET. Aggregate tables carry
    target_daily_return (next trading day's cap-weighted market return,
    already forward-shifted in Stage 4). Panel carries dlyret (day-t's
    OWN realised return for that stock, not forward-shifted). Both are
    exposed under the single key returns_{part} so calling code does not
    need to branch on dataset -- but be aware these are NOT the same kind
    of quantity, and panel's dlyret is not a valid drop-in for a backtest
    built assuming a forward return (see evaluation.py's backtest
    functions, which assume returns[t] is realised on day t+1 relative to
    the signal at t -- true for target_daily_return, not true for dlyret).

    .to_numpy(dtype=np.float32) is required because the parquets contain
    mixed column dtypes (some int, some float64) which cause pandas
    .values to produce an object array that PyTorch rejects.

    Feature clipping (+-5, see lib.config.CLIP) is handled upstream in the
    pipeline (Stage_4_Assembly/02 and 03). Data arrives here already
    clipped. No clipping in this module.

    pos_weight for class-imbalanced BCE is always computed and returned.
    Whether to actually use it is decided in the notebook, not here.

Column name contract -- must match what Stage_4_Assembly/05_splits.ipynb
actually writes:
    agg_means, agg_full_moments : date, target_daily_return,
                                   minret_5d_pct, y_binary  (+ features)
    panel                        : permno, date, dlyret, dlycap,
                                   minret_5d_pct, y_binary  (+ features)
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


# ── Per-dataset meta columns and return-column name ─────────────────────────
# Single source of truth for "what is NOT a feature" and "which column is
# the realised return", per dataset. Everything in a dataset's parquet that
# is not listed in its 'meta' set is a feature.
DATASETS = {
    "agg_means": {
        "meta": {"date", "target_daily_return", "minret_5d_pct", "y_binary"},
        "return_col": "target_daily_return",
    },
    "agg_full_moments": {
        "meta": {"date", "target_daily_return", "minret_5d_pct", "y_binary"},
        "return_col": "target_daily_return",
    },
    "panel": {
        "meta": {"permno", "date", "dlyret", "dlycap",
                 "minret_5d_pct", "y_binary"},
        "return_col": "dlyret",
    },
}

# Which taxonomy file backs each dataset. agg_full_moments carries moment
# suffixes (Tax_cwmean) and needs the full long file; agg_means and panel
# both use bare base-factor names (Tax) and share the means-only file --
# this is the sharing verified by Stage_4_Assembly/08_preflight.ipynb
# Check 2 (agg_means and panel feature sets are identical).
TAXONOMY_FILE = {
    "agg_means": "numbered_classified_moment_inventory_means_only.csv",
    "agg_full_moments": "numbered_classified_moment_inventory_long.csv",
    "panel": "numbered_classified_moment_inventory_means_only.csv",
}


def _validate_dataset(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """
    Return the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)} (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(
    split_name: str,
    dataset: str,
    splits_dir: Path,
) -> dict:
    """
    Load a prepared split from parquet files.

    Parameters
    ----------
    split_name : str
        e.g. "Split_A".
    dataset : str
        One of "agg_means", "agg_full_moments", "panel". Matches the file
        stem written by 05_splits.ipynb exactly:
            {splits_dir}/{split_name}/{dataset}_{part}.parquet

    Continuous target is 'minret_5d_pct': 100 * min(next 5 trading-day
    returns), a raw percentage -- NOT a z-score (see module docstring).

    Returns
    -------
    dict with keys:
        X_{train,val,test}      : np.ndarray float32 -- feature matrices
        y_{train,val,test}      : np.ndarray float32 -- binary target (0/1)
        minret_{train,val,test} : np.ndarray float32 -- minret_5d_pct
        dates_{train,val,test}  : pd.Series -- dates for backtesting
        returns_{train,val,test}: np.ndarray float32 -- realised return
                                   (target_daily_return for agg datasets,
                                   dlyret for panel -- see module docstring
                                   for why these are not interchangeable)
        permno_{train,val,test} : np.ndarray, ONLY present if dataset == "panel"
        feature_cols            : list of str
        n_features              : int
        dataset                 : str (echoed back, for convenience)
        metadata                : dict
    """
    _validate_dataset(dataset)
    meta_cols = DATASETS[dataset]["meta"]
    return_col = DATASETS[dataset]["return_col"]

    splits_dir = Path(splits_dir)
    split_dir = splits_dir / split_name

    if not split_dir.exists():
        available = [d.name for d in splits_dir.iterdir()
                     if d.is_dir() and d.name.startswith("Split")]
        raise FileNotFoundError(
            f"Split not found: {split_dir}\nAvailable: {available}"
        )

    result = {}

    for part in ["train", "val", "test"]:
        path = split_dir / f"{dataset}_{part}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        df = pd.read_parquet(path)

        feature_cols = [c for c in df.columns if c not in meta_cols]

        # KEY: .to_numpy(dtype=float32) required -- mixed dtypes in parquet
        result[f"X_{part}"]       = df[feature_cols].to_numpy(dtype=np.float32)
        result[f"y_{part}"]       = df["y_binary"].to_numpy(dtype=np.float32)
        result[f"minret_{part}"]  = df["minret_5d_pct"].to_numpy(dtype=np.float32)
        result[f"dates_{part}"]   = df["date"].reset_index(drop=True)
        result[f"returns_{part}"] = df[return_col].to_numpy(dtype=np.float32)

        if dataset == "panel":
            result[f"permno_{part}"] = df["permno"].to_numpy(dtype=np.int64)

    result["feature_cols"] = feature_cols
    result["n_features"]   = len(feature_cols)
    result["dataset"]      = dataset

    # Load metadata (Stage_4_Assembly/05_splits.ipynb writes metadata.json
    # into the splits_dir root, keyed by split/part with per-dataset rows
    # inside 'summary' -- kept best-effort here since callers mostly want
    # the arrays above, not this).
    meta_path = splits_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["metadata"] = json.load(f)
    else:
        result["metadata"] = {}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# THEME ASSIGNMENT (for Sparse KAN / Sparse MLP connectivity)
# ═══════════════════════════════════════════════════════════════════════════════

def load_theme_assignment(dataset: str, themes_dir: Path) -> pd.DataFrame:
    """
    Load the numbered theme/subtheme taxonomy for a given dataset.

    Parameters
    ----------
    dataset : str
        One of "agg_means", "agg_full_moments", "panel". Determines which
        taxonomy file is read (see TAXONOMY_FILE above) -- agg_means and
        panel share the means-only file since they carry identical bare
        base-factor names (verified in 08_preflight.ipynb Check 2).
    themes_dir : Path
        The Stage_5_Model_Ready/05_themes/ directory. Deliberately a
        SEPARATE argument from splits_dir, not derived from it -- the
        taxonomy lives in a different Stage-5 subfolder than the splits
        do, and no default path is assumed (see module docstring).

    Returns
    -------
    DataFrame with (at minimum) columns:
        column, subtheme_id, subtheme_name, theme_id, theme_name
    read with dtype=str on the ID columns. Subtheme/theme IDs are
    zero-padded strings like "07_03" -- this dtype is a defensive habit
    against a future re-export accidentally letting pandas infer numeric
    types, even though the underscore format (unlike the old dot format)
    cannot silently mis-parse as a float.
    """
    _validate_dataset(dataset)
    path = Path(themes_dir) / TAXONOMY_FILE[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    return pd.read_csv(path, dtype={"theme_id": str, "subtheme_id": str})


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    split_name: str,
    dataset: str,
    splits_dir: Path,
    target_type: str = "binary",
    batch_size: int = 128,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> dict:
    """
    Load a split and create PyTorch DataLoaders.

    Features arrive already clipped (see lib.config.CLIP upstream). No
    additional clipping is applied here.

    Parameters
    ----------
    dataset : str
        One of "agg_means", "agg_full_moments", "panel".
    target_type : str
        "binary"     -> y is 0/1 crash indicator, use BCEWithLogitsLoss
        "continuous" -> y is minret_5d_pct (raw percentage, NOT a
                        z-score -- see module docstring), use HuberLoss

    Returns
    -------
    dict with keys:
        train, val, test : DataLoader
        data              : raw numpy dict (both targets always accessible)
        pos_weight        : torch.Tensor for BCEWithLogitsLoss
        n_features        : int
        dataset           : str
        target_type       : str
    """
    if target_type not in ("binary", "continuous"):
        raise ValueError(
            f"target_type must be 'binary' or 'continuous', got '{target_type}'"
        )

    data = load_split(split_name, dataset, splits_dir)

    # ── TARGET SELECTION ──
    # binary     -> data["y_train"]      (0/1 labels)
    # continuous -> data["minret_train"] (minret_5d_pct, use HuberLoss)
    target_key = "y" if target_type == "binary" else "minret"

    loaders = {}
    for part in ["train", "val", "test"]:
        X = torch.tensor(data[f"X_{part}"])
        y = torch.tensor(data[f"{target_key}_{part}"]).unsqueeze(1)

        loaders[part] = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=(shuffle_train and part == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    # ── CLASS WEIGHT (binary only, but always computed for convenience) ──
    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    loaders["data"]        = data
    loaders["pos_weight"]  = pos_weight
    loaders["n_features"]  = data["n_features"]
    loaders["dataset"]     = dataset
    loaders["target_type"] = target_type

    return loaders


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_data_summary(loaders: dict):
    """Print a concise summary of loaded data."""
    data = loaders["data"]
    print(f"Dataset:     {loaders['dataset']}")
    print(f"Features:    {data['n_features']}")
    print(f"Target:      {loaders['target_type']}")
    print(f"pos_weight:  {loaders['pos_weight'].item():.2f}")
    print()
    print(f"  {'Part':<7} {'Rows':>6}  {'Crash%':>7}  {'Date range'}")
    print(f"  {'-'*7} {'-'*6}  {'-'*7}  {'-'*23}")
    for part in ["train", "val", "test"]:
        n    = len(data[f"y_{part}"])
        rate = data[f"y_{part}"].mean()
        d0   = str(data[f"dates_{part}"].iloc[0])[:10]
        d1   = str(data[f"dates_{part}"].iloc[-1])[:10]
        print(f"  {part:<7} {n:>6}  {rate:>6.1%}  {d0} → {d1}")