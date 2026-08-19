"""
Training Utilities
==================
Model-agnostic training loop, early stopping, LR scheduling, and Optuna.
Works with any nn.Module that maps (batch, n_features) -> (batch, 1).

Usage:
    from training import train_model, run_optuna, save_checkpoint

    result = train_model(
        model, train_loader, val_loader, device,
        target_type="binary",
        lr=1e-3, ...
    )

Design decisions documented inline:

    OPTIMIZER:       AdamW (decoupled weight decay, cleaner than Adam)
    LR SCHEDULER:    ReduceLROnPlateau (adapts to training dynamics)
    GRAD CLIPPING:   Fixed at max_norm=1.0 (safety net, not a tuning lever)

    LOSS FUNCTIONS:
        Binary:     BCEWithLogitsLoss (optionally class-weighted)
        Continuous: HuberLoss(delta=huber_delta)

            EDIT (post Stage_4_Assembly/04_targets.ipynb): delta is now a
            PARAMETER, not a fixed module constant. The pipeline computes
            a per-split, per-target delta as
                delta = 3.0 * 1.4826 * MAD(y_train)
            using ONLY that split's training window (see huber_delta.json,
            written by 04_targets.ipynb). The scale is estimated from data;
            the multiple (3.0 "robust sigma") is the fixed design decision
            kept constant across every split so cross-split R^2 comparison
            stays on a like-for-like loss. Callers should load
            huber_delta.json once and pass
                deltas['deltas'][f'{split_name}/market']   # or '/panel'
            as huber_delta to get_criterion / train_model.

            This matters more now than under the old pipeline's fixed
            HUBER_DELTA = 3.0, because the continuous target changed from
            an expanding z-score (std ~= 1 by construction, so "3" meant
            "3 sigma" almost by accident) to minret_5d_pct, a raw
            percentage with no such built-in scale. A fixed delta=3.0
            against a raw-percentage target would be a stand-in for
            "3 percentage points", which is not the same statement as
            "3 robust sigma of this specific target's dispersion" and
            would silently behave differently across splits.

            HUBER_DELTA below remains as the fallback used when a caller
            does not supply huber_delta explicitly (e.g. quick ad-hoc runs,
            or any code not yet updated to load huber_delta.json). Any run
            intended for the thesis results should always pass the
            per-split value explicitly rather than relying on this
            fallback.

    EARLY STOPPING:
        Binary:     monitors val AUC    (mode='max', higher = better)
        Continuous: monitors val R2     (mode='max', higher = better)
            R2 rather than Huber loss because:
            (a) Huber loss is not MSE, so its scale shifts with delta and
                is not directly interpretable as a quality metric -- doubly
                true now that delta varies by split rather than being a
                single fixed constant across the whole pipeline.
            (b) R2 is what gets reported in the thesis, so early stopping
                on R2 finds the model that maximises the reported metric.
            (c) Both targets use mode='max', simplifying the scheduler
                and early stopping configuration.

    OPTUNA PATTERN:  model_factory(trial) -> (model, train_kwargs)
                     Search space defined per-notebook, not here.
                     This file provides the generic training/search framework.
                     huber_delta is passed through train_kwargs like any
                     other training hyperparameter -- run_optuna requires
                     no changes, since it already forwards **train_kwargs
                     to train_model without inspecting individual keys.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import copy
import optuna
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import Callable, Optional

# ── Fallback Huber delta, used only when a caller does not pass huber_delta
# explicitly. Thesis-result runs should always load the per-split value from
# huber_delta.json (see Stage_4_Assembly/04_targets.ipynb) and pass it in --
# see the LOSS FUNCTIONS note above for why a single fixed constant is no
# longer an appropriate default under the raw-percentage target. ──
HUBER_DELTA = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    Saves a deepcopy of the best model state for restoration.

    Both binary and continuous now use mode='max':
        binary     -> maximise val AUC
        continuous -> maximise val R2
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best_score  = -np.inf if mode == "max" else np.inf
        self.best_epoch  = 0
        self.counter     = 0
        self.best_state  = None

    def step(self, score: float, epoch: int, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load best weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_criterion(
    target_type: str,
    pos_weight: torch.Tensor = None,
    device: torch.device = None,
    huber_delta: float = None,
) -> nn.Module:
    """
    Return the appropriate loss function for training.

    Binary:     BCEWithLogitsLoss (optionally class-weighted via pos_weight)
    Continuous: HuberLoss(delta=huber_delta), falling back to the module
                constant HUBER_DELTA if huber_delta is not supplied.

    huber_delta should be the per-split, per-target value computed by
    Stage_4_Assembly/04_targets.ipynb (huber_delta.json), NOT a value
    tuned against validation or test data -- it is estimated purely from
    each split's own training window.

    Note: Ridge and polymodel are not trained via this function -- they use
    their own sklearn/scipy fitting which internally minimises MSE. This
    asymmetry is intentional and documented in the thesis. All neural network
    models (KAN, MLP) use Huber loss via this function.

    The pos_weight for binary targets upweights crash samples (a minority
    class) so the model cannot achieve high accuracy by always predicting
    'no crash'.
    """
    if target_type == "binary":
        kwargs = {}
        if pos_weight is not None and device is not None:
            kwargs["pos_weight"] = pos_weight.to(device)
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        delta = huber_delta if huber_delta is not None else HUBER_DELTA
        # HuberLoss(delta): quadratic for |error| <= delta, linear beyond.
        # delta is scaled to this split's own training-window dispersion
        # (3.0 x robust sigma via MAD), so ordinary market days get the
        # full quadratic gradient while extreme crisis observations get a
        # capped gradient, preventing them from dominating parameter
        # updates -- without assuming the target has any particular fixed
        # scale (it does not, now that it is a raw percentage rather than
        # a pre-standardised z-score).
        return nn.HuberLoss(delta=delta)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE EPOCH: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    reg_fn: Optional[Callable] = None,
    reg_weight: float = 0.0,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train for one epoch. Returns average task loss (without regularisation).

    Parameters
    ----------
    reg_fn : callable, optional
        Regularisation: reg_fn(model) -> scalar loss.
        E.g. lambda m: m.regularization_loss() for Dense KAN coefficient L1.
        Sparse KAN and MLP pass None (no extra regularisation needed).
    reg_weight : float
        Multiplier for regularisation loss. Set via Optuna per notebook.
    max_grad_norm : float
        Fixed at 1.0. Prevents exploding gradients in deep spline networks
        without affecting converged training dynamics.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits    = model(X_batch)
        task_loss = criterion(logits, y_batch)

        loss = task_loss
        if reg_fn is not None and reg_weight > 0:
            loss = loss + reg_weight * reg_fn(model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += task_loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE EPOCH: EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_type: str = "binary",
) -> dict:
    """
    Evaluate model on a DataLoader. Returns metrics for logging and early stopping.

    Binary:
        loss   = BCEWithLogitsLoss value
        auc    = ROC-AUC
        metric = AUC  (early stopping maximises this)

    Continuous:
        loss   = Huber loss value  (logged but NOT used for early stopping)
        mse    = true MSE computed from raw squared errors, in minret_5d_pct
                 units (percentage points squared)
        r2     = 1 - MSE / Var(y_true)  (early stopping maximises this)
        metric = R2

    The separation of Huber loss from MSE/R2 is intentional:
        - Huber loss drives the optimizer during train_epoch()
        - MSE and R2 are computed purely for evaluation and early stopping
        - This means early stopping selects the model with the best R2
          (the reported metric), not the best Huber loss (the training
          proxy, whose scale now varies by split via huber_delta)
        - Both binary and continuous use mode='max' for early stopping
    """
    model.eval()
    all_outputs = []
    all_targets = []
    total_loss  = 0.0
    n_batches   = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)

        total_loss += loss.item()
        n_batches  += 1
        all_outputs.append(outputs.cpu())
        all_targets.append(y_batch.cpu())

    all_outputs = torch.cat(all_outputs).numpy().ravel()
    all_targets = torch.cat(all_targets).numpy().ravel()
    avg_loss    = total_loss / max(n_batches, 1)

    if target_type == "binary":
        y_prob = 1.0 / (1.0 + np.exp(-all_outputs))  # sigmoid

        try:
            auc = roc_auc_score(all_targets, y_prob) if len(np.unique(all_targets)) > 1 else 0.5
        except ValueError:
            auc = 0.5

        return {
            "loss":   avg_loss,
            "auc":    auc,
            "metric": auc,       # early stopping: maximise AUC
            "y_true": all_targets,
            "y_prob": y_prob,
        }

    else:
        y_pred = all_outputs

        # MSE and R2 computed from raw squared errors regardless of training loss
        mse    = np.mean((all_targets - y_pred) ** 2)
        ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
        r2     = 1.0 - np.sum((all_targets - y_pred) ** 2) / max(ss_tot, 1e-10)

        return {
            "loss":   avg_loss,  # Huber loss -- logged for diagnostics only
            "mse":    mse,       # true MSE, in minret_5d_pct units
            "r2":     r2,        # reported metric
            "metric": r2,        # early stopping: maximise R2
            "y_true": all_targets,
            "y_pred": y_pred,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL TRAINING RUN
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_type: str = "binary",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 300,
    patience: int = 20,
    pos_weight: torch.Tensor = None,
    huber_delta: float = None,
    reg_fn: Optional[Callable] = None,
    reg_weight: float = 0.0,
    verbose: bool = True,
    log_every: int = 10,
) -> dict:
    """
    Full training run with early stopping and LR scheduling.

    OPTIMIZER: AdamW
        Decoupled weight decay -- cleaner than Adam's coupled L2.
        weight_decay acts on all parameters uniformly.

    LR SCHEDULER: ReduceLROnPlateau
        Both binary and continuous use mode='max' (monitoring AUC and R2
        respectively -- both metrics where higher is better).
        Patience = max(patience // 3, 5) so LR reduces ~2 times before
        early stopping fires.
        Factor = 0.5 (halve LR each time). min_lr = 1e-6.

    EARLY STOPPING:
        Binary:     stops when val AUC stops improving
        Continuous: stops when val R2 stops improving
        Both use mode='max'. Best weights are restored after stopping.

    huber_delta : float, optional
        Only used when target_type == "continuous". The per-split value
        from huber_delta.json (see Stage_4_Assembly/04_targets.ipynb).
        Falls back to the module constant HUBER_DELTA if not supplied --
        thesis-result runs should always pass this explicitly.

    Returns
    -------
    dict with keys:
        best_val_metric : float  (AUC for binary, R2 for continuous)
        best_epoch      : int
        train_history   : list of per-epoch dicts
        val_history     : list of per-epoch dicts
        total_time      : float (seconds)
        model           : nn.Module with best weights restored
    """
    model = model.to(device)

    criterion = get_criterion(target_type, pos_weight, device, huber_delta)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Both targets now use mode='max' -- AUC and R2 are both "higher is better"
    scheduler_patience = max(patience // 3, 5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
        min_lr=1e-6,
    )

    early_stop = EarlyStopping(patience=patience, mode="max")

    train_history = []
    val_history   = []
    start_time    = time.time()

    for epoch in range(1, n_epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            reg_fn=reg_fn, reg_weight=reg_weight,
        )

        val_result = evaluate_epoch(model, val_loader, criterion, device, target_type)

        train_history.append({"epoch": epoch, "loss": train_loss})
        val_history.append({
            "epoch": epoch,
            **{k: v for k, v in val_result.items()
               if k not in ("y_true", "y_prob", "y_pred")}
        })

        scheduler.step(val_result["metric"])

        if verbose and (epoch % log_every == 0 or epoch == 1):
            current_lr = optimizer.param_groups[0]["lr"]
            if target_type == "binary":
                print(
                    f"  Epoch {epoch:>4d} | "
                    f"Train loss {train_loss:.4f} | "
                    f"Val loss {val_result['loss']:.4f}  AUC {val_result['auc']:.4f} | "
                    f"LR {current_lr:.1e}"
                )
            else:
                print(
                    f"  Epoch {epoch:>4d} | "
                    f"Train Huber {train_loss:.4f} | "
                    f"Val Huber {val_result['loss']:.4f}  "
                    f"MSE {val_result['mse']:.4f}  "
                    f"R² {val_result['r2']:.4f} | "
                    f"LR {current_lr:.1e}"
                )

        if early_stop.step(val_result["metric"], epoch, model):
            if verbose:
                metric_name = "AUC" if target_type == "binary" else "R²"
                print(
                    f"  Early stop at epoch {epoch}. "
                    f"Best val {metric_name}: {early_stop.best_score:.4f} "
                    f"at epoch {early_stop.best_epoch}"
                )
            break

    early_stop.restore_best(model)
    total_time = time.time() - start_time

    if verbose:
        print(f"  Training complete in {total_time:.1f}s")

    return {
        "best_val_metric": early_stop.best_score,
        "best_epoch":      early_stop.best_epoch,
        "train_history":   train_history,
        "val_history":     val_history,
        "total_time":      total_time,
        "model":           model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def run_optuna(
    model_factory: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_type: str = "binary",
    n_trials: int = 25,
    pos_weight: torch.Tensor = None,
    study_name: str = "study",
    storage_path: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Run Optuna hyperparameter search.

    DESIGN: model_factory pattern
        model_factory(trial) -> (model, train_kwargs)

        The factory is defined in each notebook with model-specific
        hyperparameters. training.py never needs to know what
        hyperparameters exist for each model type.

        For continuous targets, train_kwargs should include "huber_delta"
        (the per-split value loaded from huber_delta.json) alongside "lr",
        "weight_decay", etc. -- run_optuna requires no code changes for
        this, since it already forwards **train_kwargs to train_model
        without inspecting individual keys. Example factory:

            def my_factory(trial):
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                model = KAN([n_feat, 101, 14, 1])
                return model, {
                    "lr": lr,
                    "weight_decay": 1e-4,
                    "huber_delta": split_delta,  # from huber_delta.json
                }

    DIRECTION:
        Both binary and continuous now use "maximize":
            binary     -> maximize AUC
            continuous -> maximize R2

    STORAGE:
        If storage_path is provided, study is saved to SQLite for
        resumability and later analysis. Resume with:
            study = optuna.load_study(study_name, storage=f"sqlite:///{path}")

    Returns
    -------
    dict with keys:
        best_params : dict
        best_value  : float  (AUC for binary, R2 for continuous)
        study       : optuna.Study
        all_trials  : list of dicts
    """
    storage = None
    if storage_path is not None:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",   # both AUC and R2 are maximised
        load_if_exists=True,
    )

    def objective(trial):
        model, train_kwargs = model_factory(trial)

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            target_type=target_type,
            pos_weight=pos_weight,
            verbose=False,
            **train_kwargs,
        )

        return result["best_val_metric"]

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    all_trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if verbose:
        metric_name = "AUC" if target_type == "binary" else "R²"
        print(f"\n  Best {metric_name}: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "study":       study,
        "all_trials":  all_trials,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    train_result: dict,
    hyperparameters: dict,
    model_config: dict,
    path: Path,
):
    """
    Save everything needed to reconstruct and analyse a trained model.

    Contents:
        model_state_dict -- weights (spline coefficients, base weights, scalers)
        model_config     -- architecture params needed to reconstruct the class
        hyperparameters  -- Optuna best params (should include huber_delta
                             for continuous models, so it is recoverable
                             without re-reading huber_delta.json)
        best_val_metric  -- AUC (binary) or R2 (continuous)
        best_epoch       -- epoch at which early stopping triggered
        total_time       -- wall-clock training time in seconds
        train_history    -- per-epoch train loss
        val_history      -- per-epoch val metrics (includes both Huber loss
                            and MSE/R2 for continuous models)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config":     model_config,
        "hyperparameters":  hyperparameters,
        "best_val_metric":  train_result["best_val_metric"],
        "best_epoch":       train_result["best_epoch"],
        "total_time":       train_result["total_time"],
        "train_history":    train_result["train_history"],
        "val_history":      train_result["val_history"],
    }, path)


def load_checkpoint(model: nn.Module, path: Path) -> dict:
    """
    Load weights and metadata from a saved checkpoint.
    model must already be constructed with the matching architecture.
    Returns the full checkpoint dict.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt