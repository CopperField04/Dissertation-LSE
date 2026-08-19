"""
sparse_mlp.py — Sparse MLP with domain-structured connectivity.

Uses MaskedLinear: full-size nn.Linear layers with binary masks
that enforce structural sparsity. Same approach as Sparse KAN's
MaskedKANLinear but much simpler (one parameter tensor, one hook).

Layer 0: features → subthemes (MaskedLinear — within-subtheme edges only) → SiLU
Layer 1: subthemes → themes   (MaskedLinear — within-theme edges only)   → SiLU
Layer 2: themes → output      (nn.Linear — fully dense, no activation)

Architecture (full_moments):
    2204 features → 99 subthemes → 13 themes → 1
    Active edges: 2,316  |  Active parameters: 2,429 (weights + biases)
    (vs Sparse KAN: 2,316 edges, ~44,004 parameters — 18x more per edge)

Architecture (means_only):
    714 features → 99 subthemes → 13 themes → 1
    Active edges: 826  |  Active parameters: 939

KEY ARCHITECTURAL DIFFERENCE FROM SPARSE KAN:
    SiLU placement differs between KAN and MLP:
        KAN:  SiLU(x_1)*w_1 + SiLU(x_2)*w_2 + spline_1(x_1) + spline_2(x_2)
        MLP:  SiLU(x_1*w_1 + x_2*w_2 + bias)
    In the KAN, each feature gets its own nonlinear transformation (per-edge).
    In the MLP, features are linearly combined first, then one shared SiLU (per-node).
    This is a structural difference beyond just "splines vs no splines."

NOTE (interpretability): Feature/subtheme/theme NAMES are stored as
plain Python attributes, NOT registered buffers. Reload the taxonomy
CSV for human-readable names. Masks contain full structural info.

Usage:
    taxonomy_df = pd.read_csv("themes/combined_full_moments_theme_assignment.csv")
    feature_cols = [...]
    model = SparseMLP.from_taxonomy(taxonomy_df, feature_cols)
"""

import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class MaskedLinear(nn.Linear):
    """
    nn.Linear with a binary mask enforcing structural sparsity.

    Masked edges have zero weights and receive zero gradients.
    Active edges are reinitialized with correct per-node fan_in
    (Kaiming scaling based on actual active inputs, not full layer width).

    Much simpler than MaskedKANLinear: one parameter tensor (weight),
    one gradient hook, one init fix.
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        """
        Args:
            mask: (out_features, in_features) binary tensor.
                  mask[q, p] = 1 if edge (p → q) exists, 0 if masked.
        """
        super().__init__(in_features, out_features, bias=bias)

        # Store mask as buffer (saved with checkpoint)
        self.register_buffer("mask", mask.float())

        # ── Fix initialization ──
        with torch.no_grad():
            for q in range(out_features):
                actual_fan_in = int(mask[q].sum().item())
                active = mask[q].bool()

                if actual_fan_in == 0:
                    # Dead node — zero weight and bias
                    self.weight.data[q, :] = 0.0
                    if self.bias is not None:
                        self.bias.data[q] = 0.0
                    continue

                # Kaiming uniform bound: 1 / sqrt(fan_in)
                # Matches nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
                bound = 1.0 / math.sqrt(actual_fan_in)

                # FIX: build a fresh row and assign the WHOLE row back via
                # __setitem__, rather than indexing with a boolean mask and
                # calling .uniform_() on the result. self.weight.data[q, active]
                # with a boolean mask is advanced indexing -- it returns a COPY,
                # so the in-place .uniform_() silently wrote to a discarded
                # temporary, leaving every active edge at nn.Linear's own default
                # init (scaled by the full in_features, not the true per-node
                # fan_in). Confirmed empirically in sparse_kan.py's identical
                # pattern: measured weight scale matched 1/sqrt(in_features)
                # exactly, not 1/sqrt(fan_in), across 145 real subthemes.
                row = torch.zeros(in_features)
                row[active] = torch.empty(actual_fan_in).uniform_(-bound, bound)
                self.weight.data[q] = row

        # ── Gradient hook ──
        self.weight.register_hook(lambda grad: grad * self.mask)

    def active_edges(self):
        """Number of active (unmasked) edges."""
        return int(self.mask.sum().item())

    def total_edges(self):
        """Total edges (active + masked)."""
        return self.in_features * self.out_features

    def re_zero_masked(self):
        """Re-zero masked weights. Mathematically unnecessary but defensive."""
        with torch.no_grad():
            self.weight.data *= self.mask


class SparseMLP(nn.Module):
    """
    Three-layer Sparse MLP with domain-structured connectivity.

    Layer 0: features → subthemes (masked) + SiLU
    Layer 1: subthemes → themes   (masked) + SiLU
    Layer 2: themes → output      (dense, no activation)

    Same structural sparsity as SparseKAN but with nn.Linear + SiLU
    instead of KANLinear B-splines. Isolates the spline contribution.
    """

    def __init__(self, mask_0, mask_1, n_features, n_subthemes, n_themes,
                 subtheme_names=None, theme_names=None, feature_names=None):
        super().__init__()

        self.n_features = n_features
        self.n_subthemes = n_subthemes
        self.n_themes = n_themes

        self._subtheme_names = subtheme_names or [f"sub_{i}" for i in range(n_subthemes)]
        self._theme_names = theme_names or [f"theme_{i}" for i in range(n_themes)]
        if feature_names is not None:
            self._feature_names = feature_names

        self.activation = nn.SiLU()

        # Layer 0: features → subthemes (masked)
        self.layer0 = MaskedLinear(n_features, n_subthemes, mask_0)

        # Layer 1: subthemes → themes (masked)
        self.layer1 = MaskedLinear(n_subthemes, n_themes, mask_1)

        # Layer 2: themes → output (fully dense)
        self.layer2 = nn.Linear(n_themes, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)    # NaN safety
        x = self.activation(self.layer0(x))  # (batch, n_subthemes)
        x = self.activation(self.layer1(x))  # (batch, n_themes)
        x = self.layer2(x)                   # (batch, 1)
        return x

    def all_linear_layers(self):
        """Iterate over all Linear layers (masked and dense)."""
        yield self.layer0
        yield self.layer1
        yield self.layer2

    def count_parameters(self):
        """Total trainable parameters (including masked zeros — memory footprint)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self):
        """
        Parameters on active (unmasked) edges only, plus biases.
        Per active edge: 1 weight. Per node: 1 bias.
        """
        active = 0
        for layer in [self.layer0, self.layer1]:
            n_active_edges = int(layer.mask.sum().item())
            active += n_active_edges  # weights
            active += layer.out_features  # biases
        # Layer 2 is fully dense
        active += self.layer2.in_features * self.layer2.out_features
        active += self.layer2.out_features
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
        """Re-zero masked params in all masked layers."""
        self.layer0.re_zero_masked()
        self.layer1.re_zero_masked()

    def architecture_summary(self):
        """Print the wiring diagram."""
        l0_active = self.layer0.active_edges()
        l1_active = self.layer1.active_edges()
        l2_active = self.layer2.in_features * self.layer2.out_features
        total_active = l0_active + l1_active + l2_active

        print(f"  Sparse MLP Architecture (MaskedLinear)")
        print(f"  {'='*60}")
        print(f"  Input features: {self.n_features}")
        print(f"  Activation: SiLU (per-node, after linear combination)")
        print(f"  Active parameters: {self.count_active_parameters():,} "
              f"(memory footprint: {self.count_parameters():,})")
        print(f"  Active edges: {total_active:,} / {self.count_total_edges():,} "
              f"({total_active/self.count_total_edges()*100:.1f}%)")
        print()

        mask_0 = self.layer0.mask
        print(f"  Layer 0: {self.n_features} features → {self.n_subthemes} subthemes + SiLU")
        for q in range(self.n_subthemes):
            n_active = int(mask_0[q].sum().item())
            if n_active > 0:
                print(f"    {self._subtheme_names[q]:<50} {n_active:>3} features")
        print(f"    Active: {l0_active} / {self.layer0.total_edges()} edges")
        print()

        mask_1 = self.layer1.mask
        print(f"  Layer 1: {self.n_subthemes} subthemes → {self.n_themes} themes + SiLU")
        for j in range(self.n_themes):
            n_active = int(mask_1[j].sum().item())
            if n_active > 0:
                print(f"    {self._theme_names[j]:<50} {n_active:>3} subthemes")
        print(f"    Active: {l1_active} / {self.layer1.total_edges()} edges")
        print()

        print(f"  Layer 2: {self.n_themes} themes → 1 output (dense, no activation)")
        print(f"    Active: {l2_active} / {l2_active} edges")
        print()
        print(f"  Total active edges: {total_active}")

    def verify_masking(self):
        """Verify that all masked parameters are exactly zero."""
        ok = True
        for name, layer in [("Layer 0", self.layer0), ("Layer 1", self.layer1)]:
            mask = layer.mask
            inv_mask = ~mask.bool()

            w_masked = layer.weight.data[inv_mask]
            if w_masked.numel() > 0 and w_masked.abs().max() > 0:
                print(f"  ERROR: {name} weight has non-zero masked entries "
                      f"(max={w_masked.abs().max():.2e})")
                ok = False

        if ok:
            print("  ✓ All masked parameters are exactly zero")
        return ok

    @classmethod
    def from_taxonomy(cls, taxonomy_df, feature_cols):
        """Build a SparseMLP from a taxonomy DataFrame and feature column list."""
        col_to_idx = {col: i for i, col in enumerate(feature_cols)}
        n_features = len(feature_cols)

        tax = taxonomy_df[taxonomy_df["column"].isin(feature_cols)].copy()
        tax["col_idx"] = tax["column"].map(col_to_idx)

        if len(tax) != n_features:
            missing = set(feature_cols) - set(tax["column"])
            if missing:
                print(f"  WARNING: {len(missing)} features not in taxonomy: "
                      f"{list(missing)[:5]}...")

        subtheme_ids = sorted(tax["subtheme_id"].unique())
        theme_ids = sorted(tax["theme_id"].unique())
        n_subthemes = len(subtheme_ids)
        n_themes = len(theme_ids)

        sub_id_to_idx = {s: i for i, s in enumerate(subtheme_ids)}
        theme_id_to_idx = {t: i for i, t in enumerate(theme_ids)}

        mask_0 = torch.zeros(n_subthemes, n_features)
        for _, row in tax.iterrows():
            sub_idx = sub_id_to_idx[row["subtheme_id"]]
            feat_idx = row["col_idx"]
            mask_0[sub_idx, feat_idx] = 1.0

        mask_1 = torch.zeros(n_themes, n_subthemes)
        for _, row in tax.iterrows():
            theme_idx = theme_id_to_idx[row["theme_id"]]
            sub_idx = sub_id_to_idx[row["subtheme_id"]]
            if mask_0[sub_idx].sum() > 0:
                mask_1[theme_idx, sub_idx] = 1.0

        subtheme_names = []
        for s_id in subtheme_ids:
            name = tax.loc[tax["subtheme_id"] == s_id, "subtheme_name"].iloc[0]
            subtheme_names.append(f"{s_id} {name}")

        theme_names = []
        for t_id in theme_ids:
            name = tax.loc[tax["theme_id"] == t_id, "theme_name"].iloc[0]
            theme_names.append(f"{t_id} {name}")

        return cls(
            mask_0=mask_0, mask_1=mask_1,
            n_features=n_features, n_subthemes=n_subthemes, n_themes=n_themes,
            subtheme_names=subtheme_names, theme_names=theme_names,
            feature_names=feature_cols,
        )


def sparse_mlp_weight_l1(model):
    """
    L1 on all Linear layer weight matrices in the Sparse MLP.
    Masked entries are zero and contribute zero penalty automatically.
    NOTE: Includes Layer 2 (themes → output).
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.all_linear_layers():
        loss = loss + layer.weight.abs().sum()
    return loss


def sparse_mlp_internal_l1(model):
    """L1 on masked layers only. Preserves all theme → output connections."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in [model.layer0, model.layer1]:
        loss = loss + layer.weight.abs().sum()
    return loss