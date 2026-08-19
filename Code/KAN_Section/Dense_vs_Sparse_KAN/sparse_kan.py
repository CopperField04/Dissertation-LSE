"""
sparse_kan.py — Sparse KAN with domain-structured connectivity.

Uses MaskedKANLinear: full-size KANLinear layers with binary masks
that enforce structural sparsity. Mathematically equivalent to
separate small KANLinear modules per subtheme/theme, but runs as
3 large matrix operations instead of hundreds of small ones.

Layer 0: features → subthemes (MaskedKANLinear — only within-subtheme edges active)
Layer 1: subthemes → themes   (MaskedKANLinear — only within-theme edges active)
Layer 2: themes → output      (KANLinear — fully dense)

═══════════════════════════════════════════════════════════════════════════
MAJOR REVISION — grid adaptation removed, BatchNorm added
═══════════════════════════════════════════════════════════════════════════

WHAT CHANGED AND WHY:

Earlier versions of this file threaded a `grid_eps` parameter through every
constructor to support calling `update_grid()` partway through training,
repositioning each layer's B-spline knots to fit the empirical distribution
of its inputs. This was investigated in depth and abandoned:

  1. update_grid()'s internal curve2coeff() step performs a least-squares
     refit of spline coefficients on the NEW knot positions. This solve
     was found, via direct diagnostic, to produce NaN/inf on legitimate
     ACTIVE (unmasked) edges — independent of grid_eps (reproduced
     identically at grid_eps=0.02 and 0.25) and independent of masking
     (masking cannot protect active edges, since they are not masked).
     A snapshot-and-revert guard was built to catch this, but across a
     real sweep the guard fired on a majority of scheduled attempts —
     the mechanism was rarely providing any benefit while adding real risk
     and substantial code complexity.

  2. A follow-up measurement of per-node activation ranges (across many
     trained Dense KAN checkpoints, same underlying architecture pattern)
     found the ROOT CAUSE: hidden-layer activation scale varies by orders
     of magnitude depending on dataset, target, and training progress —
     from a typical spread of 1-5 units to, in the worst observed case,
     a spread exceeding 400. No fixed OR adaptively-repositioned grid can
     be correct for both regimes simultaneously, and grid repositioning
     was treating a symptom (wrong scale) rather than the actual cause
     (nothing constrains the scale in the first place).

THE FIX: nn.BatchNorm1d is now inserted between every KAN layer (after
layer0's output / before layer1's input; after layer1's output / before
layer2's input). BatchNorm forces each unit's activation distribution
toward zero-mean, unit-variance BEFORE the affine (gamma/beta) step,
regardless of how many inputs feed that unit or what scale the target
requires — this is the invariant a fixed grid range was always silently
assuming and never actually had. With that invariant now genuinely
enforced, a FIXED grid_range=[-5.5, 5.5] (matching Layer 0, which has
always been fixed since its input is the pipeline's own z-scored, ±5
clipped feature data) is correct for every layer, and update_grid() is
no longer needed anywhere in this file. grid_eps has been removed
entirely, since it was only ever read inside update_grid().

INTERPRETABILITY: BatchNorm was chosen over LayerNorm specifically to
preserve per-edge interpretability. LayerNorm's statistics are computed
ACROSS UNITS within one sample, so a given unit's normalised output would
depend on every other unit's raw value for that row — this would entangle
subtheme/theme contributions and destroy the additive separability that
is the entire point of this architecture. BatchNorm's statistics are
computed ACROSS THE BATCH, per unit — unit i's output depends only on
unit i's own values (plus frozen running statistics at inference), so no
cross-unit entanglement occurs. At inference, BatchNorm reduces to a
fixed per-unit affine map (A_i * x + B_i, from the running mean/var),
which is in principle foldable directly into the adjacent spline's
coefficients for clean single-curve-per-edge symbolic extraction.

BatchNorm sits strictly BETWEEN layers and never touches spline_weight,
base_weight, or spline_scaler directly, so the five equivalence
requirements below and the masking mechanism are entirely unaffected.

NOTE ON ATTRIBUTE NAMES: SparseKAN exposes its three KAN layers as NAMED
attributes -- self.layer0, self.layer1, self.layer2 -- not as a
ModuleList called `.layers` the way efficient_kan.KAN's own wrapper
class does. The two BatchNorm layers are similarly named self.bn1,
self.bn2 (bn1 sits after layer0, before layer1; bn2 sits after layer1,
before layer2).

Five requirements for equivalence with separate modules:
    1. Per-node fan_in reinitialization (Kaiming scaling)
    2. Zero spline_weight for masked edges
    3. Gradient hooks on base_weight, spline_scaler, spline_weight
    4. (Unnecessary — hooks prevent drift, included as safety in re_zero_masked)
    5. Empty subthemes excluded from Layer 1 mask

Usage:
    taxonomy_df = pd.read_csv("themes/numbered_classified_moment_inventory_long.csv")
    feature_cols = [...]
    model = SparseKAN.from_taxonomy(taxonomy_df, feature_cols, grid_size=14)
"""

import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from efficient_kan import KANLinear


class MaskedKANLinear(KANLinear):
    """
    KANLinear with a binary mask enforcing structural sparsity.
    Masked edges have zero weights and receive zero gradients.
    Active edges are reinitialized with correct per-node fan_in
    (Kaiming scaling based on actual active inputs, not full layer width).

    The spline_scaler acts as a natural gate: when scaler=0, the
    spline_weight gradient is automatically zero through the chain rule.
    The spline_weight hook is therefore redundant but included for safety.
    """

    def __init__(self, in_features, out_features, mask,
                 grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
        """
        Args:
            mask: (out_features, in_features) binary tensor.
                  mask[q, p] = 1 if edge (p → q) exists, 0 if masked.

        grid_eps is no longer a parameter here -- see module docstring.
        This class now relies entirely on a fixed grid_range, with
        activation scale controlled upstream by BatchNorm (in SparseKAN,
        not here) rather than by adapting the grid to the data.
        """
        super().__init__(
            in_features, out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=True,
            grid_range=grid_range,
        )
        # Store mask as buffer (saved with checkpoint, not a parameter)
        self.register_buffer("mask", mask.float())

        # ── Requirement 1 & 2: Fix initialization ──
        with torch.no_grad():
            for q in range(out_features):
                actual_fan_in = int(mask[q].sum().item())
                active = mask[q].bool()

                if actual_fan_in == 0:
                    self.base_weight.data[q, :] = 0.0
                    self.spline_scaler.data[q, :] = 0.0
                    self.spline_weight.data[q, :, :] = 0.0
                    continue

                bound = 1.0 / math.sqrt(actual_fan_in)

                # FIX: build a fresh row tensor and assign the WHOLE row back via
                # __setitem__, rather than indexing with a boolean mask and calling
                # .uniform_() on the result. `self.base_weight.data[q, active]` with
                # a boolean mask is advanced indexing -- it returns a COPY, so the
                # in-place .uniform_() silently wrote to a discarded temporary.
                # `self.base_weight.data[q]` (no mask) is a plain slice view, and
                # `row[active] = ...` is __setitem__, which writes through correctly.
                row = torch.zeros(in_features)
                row[active] = torch.empty(actual_fan_in).uniform_(-bound, bound)
                self.base_weight.data[q] = row

                row2 = torch.zeros(in_features)
                row2[active] = torch.empty(actual_fan_in).uniform_(-bound, bound)
                self.spline_scaler.data[q] = row2

                self.spline_weight.data[q, ~active, :] = 0.0

        # ── Requirement 3: Gradient hooks ──
        # base_weight: essential (no upstream gate protects it)
        # spline_scaler: essential (no upstream gate protects it)
        # spline_weight: redundant (spline_scaler=0 zeros its gradient
        #                via chain rule) but harmless safety net
        self.base_weight.register_hook(
            lambda grad: grad * self.mask
        )
        self.spline_scaler.register_hook(
            lambda grad: grad * self.mask
        )
        self.spline_weight.register_hook(
            lambda grad: grad * self.mask.unsqueeze(-1)
        )

    def active_edges(self):
        """Number of active (unmasked) edges."""
        return int(self.mask.sum().item())

    def total_edges(self):
        """Total edges (active + masked)."""
        return self.in_features * self.out_features

    def re_zero_masked(self):
        """
        Re-zero masked parameters. Mathematically unnecessary (hooks
        prevent any updates to masked params during backward), and no
        longer serves the additional purpose it once had of correcting
        update_grid()'s direct writes -- update_grid() is no longer
        called anywhere in this architecture. Kept as a general
        defensive safety net at negligible cost.
        """
        with torch.no_grad():
            self.base_weight.data *= self.mask
            self.spline_scaler.data *= self.mask
            self.spline_weight.data *= self.mask.unsqueeze(-1)


class SparseKAN(nn.Module):
    """
    Three-layer Sparse KAN with domain-structured connectivity and
    BatchNorm-stabilised activation scale.

    Layer 0: features → subthemes (masked — within-subtheme edges only)
    BatchNorm1d(n_subthemes)
    Layer 1: subthemes → themes   (masked — within-theme edges only)
    BatchNorm1d(n_themes)
    Layer 2: themes → output      (dense — all themes connect)

    Wiring determined by feature taxonomy CSV. Masks stored as
    registered buffers — saved with checkpoint via state_dict.

    NOTE (interpretability): Feature/subtheme/theme NAMES are stored
    as plain Python attributes (_subtheme_names, _theme_names,
    _feature_names), NOT as registered buffers. They survive
    torch.save(model, ...) but NOT model.load_state_dict().
    For interpretability analysis, reload the taxonomy CSV alongside
    the checkpoint to recover human-readable names. The masks in the
    checkpoint contain the full structural information (which features
    connect to which subthemes) — the names are just labels.

    NOTE (attribute names): KAN layers are exposed as self.layer0/1/2,
    NOT as a `.layers` ModuleList. BatchNorm layers are self.bn1
    (after layer0), self.bn2 (after layer1). See module docstring.
    """

    def __init__(self, mask_0, mask_1, n_features, n_subthemes, n_themes,
                 grid_size, spline_order, grid_range,
                 subtheme_names=None, theme_names=None, feature_names=None):
        """
        Args:
            mask_0: (n_subthemes, n_features) binary tensor
            mask_1: (n_themes, n_subthemes) binary tensor
            n_features, n_subthemes, n_themes: dimensions
            grid_size, spline_order, grid_range: KANLinear params.
                grid_range is now FIXED for the model's lifetime -- see
                module docstring for why this is safe once BatchNorm
                controls activation scale upstream.
        """
        super().__init__()
        self.n_features = n_features
        self.n_subthemes = n_subthemes
        self.n_themes = n_themes
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Store metadata for interpretability
        self._subtheme_names = subtheme_names or [f"sub_{i}" for i in range(n_subthemes)]
        self._theme_names = theme_names or [f"theme_{i}" for i in range(n_themes)]
        if feature_names is not None:
            self._feature_names = feature_names

        # Layer 0: features → subthemes (masked)
        self.layer0 = MaskedKANLinear(
            n_features, n_subthemes, mask_0,
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            grid_range=grid_range,
        )

        # BatchNorm between layer0 and layer1: forces subtheme-score
        # activations toward zero-mean/unit-variance regardless of how
        # many features feed a given subtheme (fan_in varies a lot across
        # subthemes) or what scale the target requires. This is what
        # makes the fixed grid_range below correct for every subtheme
        # simultaneously.
        self.bn1 = nn.BatchNorm1d(n_subthemes, affine=False, eps=1e-5)

        # Layer 1: subthemes → themes (masked)
        self.layer1 = MaskedKANLinear(
            n_subthemes, n_themes, mask_1,
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            grid_range=grid_range,
        )

        # BatchNorm between layer1 and layer2: same rationale, for
        # theme-level scores feeding the final dense output layer.
        self.bn2 = nn.BatchNorm1d(n_themes, affine=False, eps=1e-5)

        # Layer 2: themes → output (fully dense)
        self.layer2 = KANLinear(
            n_themes, 1,
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            grid_range=grid_range,
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.bn1(x)
        x = torch.clamp(x, min=-5.0, max=5.0) 
        
        x = self.layer1(x)
        x = self.bn2(x)
        x = torch.clamp(x, min=-5.0, max=5.0) 
        
        x = self.layer2(x)
        return x

    def all_kan_layers(self):
        """Iterate over all KANLinear layers (masked and dense). Does NOT
        include the BatchNorm layers, deliberately -- L1 regularisation
        functions below use this to penalise spline/base weights only."""
        yield self.layer0
        yield self.layer1
        yield self.layer2

    def count_parameters(self):
        """
        Total trainable parameters (including masked zeros AND the two
        BatchNorm layers' gamma/beta -- memory/full-model footprint).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self):
        """
        Parameters on active (unmasked) KAN edges only. Does NOT include
        BatchNorm's gamma/beta (2*(n_subthemes+n_themes) parameters,
        negligible relative to spline coefficient counts, and not part
        of the "active edges" story this figure is meant to report).
        Use this for thesis reporting of sparse connectivity.
        Use count_parameters() for total memory overhead.
        """
        active = 0
        for layer in [self.layer0, self.layer1]:
            n_active_edges = int(layer.mask.sum().item())
            n_d = layer.spline_weight.shape[-1]  # grid_size + spline_order
            active += n_active_edges * (n_d + 2)  # spline_weight + scaler + base
        # Layer 2 is fully dense
        n_d_2 = self.layer2.spline_weight.shape[-1]
        active += self.n_themes * (n_d_2 + 2)
        return active

    def count_active_edges(self):
        """Active (unmasked) edges across all layers."""
        l2_edges = self.layer2.in_features * self.layer2.out_features
        return (self.layer0.active_edges() +
                self.layer1.active_edges() +
                l2_edges)

    def count_total_edges(self):
        """Total edges including masked."""
        l2_edges = self.layer2.in_features * self.layer2.out_features
        return (self.layer0.total_edges() +
                self.layer1.total_edges() +
                l2_edges)

    def re_zero_all_masked(self):
        """
        Re-zero masked params in all masked layers (layer0, layer1).
        Layer 2 is fully dense and has no mask, so it is not touched here.
        No longer tied to any grid-update mechanism (see module docstring)
        -- kept purely as a cheap defensive safety net, safe to call at
        any point in training.
        """
        self.layer0.re_zero_masked()
        self.layer1.re_zero_masked()

    def architecture_summary(self):
        """Print the wiring diagram."""
        l0_active = self.layer0.active_edges()
        l1_active = self.layer1.active_edges()
        l2_active = self.layer2.in_features * self.layer2.out_features
        total_active = l0_active + l1_active + l2_active

        print(f"  Sparse KAN Architecture (MaskedKANLinear + BatchNorm)")
        print(f"  {'='*60}")
        print(f"  Input features: {self.n_features}")
        print(f"  Grid: G={self.grid_size}, K={self.spline_order}, "
              f"range={self.grid_range} (FIXED, no grid adaptation)")
        print(f"  Active parameters: {self.count_active_parameters():,} "
              f"(memory footprint: {self.count_parameters():,}, "
              f"incl. BatchNorm gamma/beta)")
        print(f"  Active edges: {total_active:,} / {self.count_total_edges():,} "
              f"({total_active/self.count_total_edges()*100:.1f}%)")
        print()

        # Layer 0 details
        mask_0 = self.layer0.mask
        print(f"  Layer 0: {self.n_features} features → {self.n_subthemes} subthemes")
        for q in range(self.n_subthemes):
            n_active = int(mask_0[q].sum().item())
            if n_active > 0:
                print(f"    {self._subtheme_names[q]:<50} {n_active:>3} features")
        print(f"    Active: {l0_active} / {self.layer0.total_edges()} edges")
        print(f"    -> BatchNorm1d({self.n_subthemes})")
        print()

        # Layer 1 details
        mask_1 = self.layer1.mask
        print(f"  Layer 1: {self.n_subthemes} subthemes → {self.n_themes} themes")
        for j in range(self.n_themes):
            n_active = int(mask_1[j].sum().item())
            if n_active > 0:
                print(f"    {self._theme_names[j]:<50} {n_active:>3} subthemes")
        print(f"    Active: {l1_active} / {self.layer1.total_edges()} edges")
        print(f"    -> BatchNorm1d({self.n_themes})")
        print()

        print(f"  Layer 2: {self.n_themes} themes → 1 output (dense)")
        print(f"    Active: {l2_active} / {l2_active} edges")
        print()
        print(f"  Total active edges: {total_active}")

    def verify_masking(self):
        """
        Verify that all masked parameters are exactly zero.

        Uses torch.isfinite() explicitly rather than a plain
        'tensor.abs().max() > 0' check. Any comparison against NaN
        evaluates to False under IEEE 754, so a naive check would
        silently report "clear" if a masked position ever held NaN
        instead of zero. Confirmed via direct injection test during
        earlier debugging (a real NaN was written to a masked position
        and a naive check returned True). Both "non-zero" and
        "non-finite" are treated as failures here.

        Returns True if correct, prints diagnostics if not.
        """
        ok = True
        for name, layer in [("Layer 0", self.layer0), ("Layer 1", self.layer1)]:
            mask = layer.mask
            inv_mask = ~mask.bool()

            if inv_mask.sum() == 0:
                # No masked entries at all in this layer -- nothing to check.
                continue

            for pname, tensor in [
                ("base_weight", layer.base_weight.data[inv_mask]),
                ("spline_scaler", layer.spline_scaler.data[inv_mask]),
                ("spline_weight", layer.spline_weight.data[inv_mask]),
            ]:
                if tensor.numel() == 0:
                    continue
                n_nonfinite = (~torch.isfinite(tensor)).sum().item()
                finite_vals = tensor[torch.isfinite(tensor)]
                max_abs = finite_vals.abs().max().item() if finite_vals.numel() > 0 else 0.0
                if n_nonfinite > 0 or max_abs > 0:
                    print(f"  ERROR: {name} {pname} has {n_nonfinite} non-finite "
                        f"and/or non-zero masked entries (max finite abs="
                        f"{max_abs:.2e})")
                    ok = False

        if ok:
            print("  ✓ All masked parameters are exactly zero and finite")
        return ok

    @classmethod
    def from_taxonomy(cls, taxonomy_df, feature_cols, grid_size=14,
                      spline_order=3, grid_range=None):
        """
        Build a SparseKAN from a taxonomy DataFrame and feature column list.

        Args:
            taxonomy_df: DataFrame with columns 'column', 'subtheme_id',
                         'subtheme_name', 'theme_id', 'theme_name'
            feature_cols: list of feature column names from the data split
            grid_size, spline_order: KANLinear parameters
            grid_range: fixed grid range for all layers. Defaults to
                       [-5.5, 5.5], matching the ±5 clip applied to input
                       features upstream in the pipeline. No longer
                       adapted during training -- see module docstring
                       for why this is safe once BatchNorm is in place.

        Returns:
            SparseKAN with masks built from taxonomy.
        """
        if grid_range is None:
            grid_range = [-5.5, 5.5]

        # ── Map feature names to column indices ──
        col_to_idx = {col: i for i, col in enumerate(feature_cols)}
        n_features = len(feature_cols)

        # Filter taxonomy to features in this data split
        tax = taxonomy_df[taxonomy_df["column"].isin(feature_cols)].copy()
        tax["col_idx"] = tax["column"].map(col_to_idx)

        if len(tax) != n_features:
            missing = set(feature_cols) - set(tax["column"])
            if missing:
                print(f"  WARNING: {len(missing)} features not in taxonomy: "
                      f"{list(missing)[:5]}...")

        # ── Build subtheme and theme orderings ──
        subtheme_ids = sorted(tax["subtheme_id"].unique())
        theme_ids = sorted(tax["theme_id"].unique())
        n_subthemes = len(subtheme_ids)
        n_themes = len(theme_ids)

        sub_id_to_idx = {s: i for i, s in enumerate(subtheme_ids)}
        theme_id_to_idx = {t: i for i, t in enumerate(theme_ids)}

        # ── Build Layer 0 mask: (n_subthemes, n_features) ──
        mask_0 = torch.zeros(n_subthemes, n_features)
        for _, row in tax.iterrows():
            sub_idx = sub_id_to_idx[row["subtheme_id"]]
            feat_idx = row["col_idx"]
            mask_0[sub_idx, feat_idx] = 1.0

        # ── Build Layer 1 mask: (n_themes, n_subthemes) ──
        mask_1 = torch.zeros(n_themes, n_subthemes)
        for _, row in tax.iterrows():
            theme_idx = theme_id_to_idx[row["theme_id"]]
            sub_idx = sub_id_to_idx[row["subtheme_id"]]
            # Only connect subthemes that have active features
            if mask_0[sub_idx].sum() > 0:
                mask_1[theme_idx, sub_idx] = 1.0

        # ── Build name lists ──
        subtheme_names = []
        for s_id in subtheme_ids:
            name = tax.loc[tax["subtheme_id"] == s_id, "subtheme_name"].iloc[0]
            subtheme_names.append(f"{s_id} {name}")

        theme_names = []
        for t_id in theme_ids:
            name = tax.loc[tax["theme_id"] == t_id, "theme_name"].iloc[0]
            theme_names.append(f"{t_id} {name}")

        return cls(
            mask_0=mask_0,
            mask_1=mask_1,
            n_features=n_features,
            n_subthemes=n_subthemes,
            n_themes=n_themes,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
            subtheme_names=subtheme_names,
            theme_names=theme_names,
            feature_names=feature_cols,
        )


def sparse_kan_edge_l1(model):
    """
    Edge-pruning L1 for the Sparse KAN.
    Penalises |spline_scaler| + |base_weight| for all KAN layers.
    Masked entries are zero and contribute zero penalty automatically.
    BatchNorm parameters are NOT penalised (all_kan_layers() only yields
    the three KANLinear layers).

    NOTE: This includes Layer 2 (themes → output). Aggressive L1
    could drive a theme's connection to zero, disconnecting it from
    the output. This is by design — if a theme is uninformative,
    L1 can remove it. If you want to preserve all theme connections,
    use sparse_kan_internal_l1 instead.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.all_kan_layers():
        if hasattr(layer, "spline_scaler"):
            loss = loss + layer.spline_scaler.abs().sum()
        else:
            loss = loss + layer.spline_weight.abs().mean(dim=-1).sum()
        loss = loss + layer.base_weight.abs().sum()
    return loss


def sparse_kan_internal_l1(model):
    """
    L1 on masked layers only (Layer 0 and Layer 1).
    Preserves all theme → output connections in Layer 2.
    Use this if you want L1 for within-subtheme feature selection
    but don't want to risk disconnecting entire themes.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in [model.layer0, model.layer1]:
        loss = loss + layer.spline_scaler.abs().sum()
        loss = loss + layer.base_weight.abs().sum()
    return loss