"""
sparse_kan.py — Sparse KAN with domain-structured connectivity (V4).

Uses MaskedKANLinear: full-size KANLinear layers with binary masks
that enforce structural sparsity. Mathematically equivalent to
separate small KANLinear modules per subtheme/theme, but runs as
3 large matrix operations instead of hundreds of small ones.

Layer 0: features → subthemes (MaskedKANLinear — only within-subtheme edges active)
Layer 1: subthemes → themes   (MaskedKANLinear — only within-theme edges active)
Layer 2: themes → output      (KANLinear — fully dense)

V4 UPDATE:
Allows Layer 0 to have an independent grid size (grid_size_0, e.g., G=3)
while deeper layers maintain full capacity (grid_size_12, e.g., G=14).
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
    """

    def __init__(self, in_features, out_features, mask,
                 grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, grid_range=[-1, 1]):
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
        self.register_buffer("mask", mask.float())

        # ── Corrected Initialization (Fixing the silent copy bug) ──
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

                row = torch.zeros(in_features)
                row[active] = torch.empty(actual_fan_in).uniform_(-bound, bound)
                self.base_weight.data[q] = row

                row2 = torch.zeros(in_features)
                row2[active] = torch.empty(actual_fan_in).uniform_(-bound, bound)
                self.spline_scaler.data[q] = row2

                self.spline_weight.data[q, ~active, :] = 0.0

        # ── Gradient hooks ──
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
        return int(self.mask.sum().item())

    def total_edges(self):
        return self.in_features * self.out_features

    def re_zero_masked(self):
        with torch.no_grad():
            self.base_weight.data *= self.mask
            self.spline_scaler.data *= self.mask
            self.spline_weight.data *= self.mask.unsqueeze(-1)


class SparseKAN(nn.Module):
    """
    Three-layer Sparse KAN with domain-structured connectivity and
    BatchNorm-stabilised activation scale (V4 with split Layer 0 grid size).
    """

    def __init__(self, mask_0, mask_1, n_features, n_subthemes, n_themes,
                 grid_size_0, grid_size_12, spline_order, grid_range,
                 subtheme_names=None, theme_names=None, feature_names=None):
        super().__init__()
        self.n_features = n_features
        self.n_subthemes = n_subthemes
        self.n_themes = n_themes
        self.grid_size_0 = grid_size_0
        self.grid_size_12 = grid_size_12
        self.spline_order = spline_order
        self.grid_range = grid_range

        self._subtheme_names = subtheme_names or [f"sub_{i}" for i in range(n_subthemes)]
        self._theme_names = theme_names or [f"theme_{i}" for i in range(n_themes)]
        if feature_names is not None:
            self._feature_names = feature_names

        # Layer 0: uses grid_size_0 (stiffer aggregation for 1699 features)
        self.layer0 = MaskedKANLinear(
            n_features, n_subthemes, mask_0,
            grid_size=grid_size_0, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            grid_range=grid_range,
        )
        self.bn1 = nn.BatchNorm1d(n_subthemes, affine=False, eps=1e-5)

        # Layer 1: uses grid_size_12
        self.layer1 = MaskedKANLinear(
            n_subthemes, n_themes, mask_1,
            grid_size=grid_size_12, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            grid_range=grid_range,
        )
        self.bn2 = nn.BatchNorm1d(n_themes, affine=False, eps=1e-5)

        # Layer 2: uses grid_size_12
        self.layer2 = KANLinear(
            n_themes, 1,
            grid_size=grid_size_12, spline_order=spline_order,
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
        yield self.layer0
        yield self.layer1
        yield self.layer2

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self):
        active = 0
        for layer, gs in zip([self.layer0, self.layer1], [self.grid_size_0, self.grid_size_12]):
            n_active_edges = int(layer.mask.sum().item())
            n_d = gs + self.spline_order
            active += n_active_edges * (n_d + 2)
        n_d_2 = self.grid_size_12 + self.spline_order
        active += self.n_themes * (n_d_2 + 2)
        return active

    def count_active_edges(self):
        l2_edges = self.layer2.in_features * self.layer2.out_features
        return (self.layer0.active_edges() +
                self.layer1.active_edges() +
                l2_edges)

    def count_total_edges(self):
        l2_edges = self.layer2.in_features * self.layer2.out_features
        return (self.layer0.total_edges() +
                self.layer1.total_edges() +
                l2_edges)

    def re_zero_all_masked(self):
        self.layer0.re_zero_masked()
        self.layer1.re_zero_masked()

    def verify_masking(self):
        ok = True
        for name, layer in [("Layer 0", self.layer0), ("Layer 1", self.layer1)]:
            mask = layer.mask
            inv_mask = ~mask.bool()

            if inv_mask.sum() == 0:
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
                    ok = False

        return ok

    @classmethod
    def from_taxonomy(cls, taxonomy_df, feature_cols, grid_size_0=3,
                      grid_size_12=14, spline_order=3, grid_range=None):
        if grid_range is None:
            grid_range = [-5.5, 5.5]

        col_to_idx = {col: i for i, col in enumerate(feature_cols)}
        n_features = len(feature_cols)

        tax = taxonomy_df[taxonomy_df["column"].isin(feature_cols)].copy()
        tax["col_idx"] = tax["column"].map(col_to_idx)

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
            mask_0=mask_0,
            mask_1=mask_1,
            n_features=n_features,
            n_subthemes=n_subthemes,
            n_themes=n_themes,
            grid_size_0=grid_size_0,
            grid_size_12=grid_size_12,
            spline_order=spline_order,
            grid_range=grid_range,
            subtheme_names=subtheme_names,
            theme_names=theme_names,
            feature_names=feature_cols,
        )


def sparse_kan_edge_l1(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.all_kan_layers():
        if hasattr(layer, "spline_scaler"):
            loss = loss + layer.spline_scaler.abs().sum()
        else:
            loss = loss + layer.spline_weight.abs().mean(dim=-1).sum()
        loss = loss + layer.base_weight.abs().sum()
    return loss


def sparse_kan_internal_l1(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in [model.layer0, model.layer1]:
        loss = loss + layer.spline_scaler.abs().sum()
        loss = loss + layer.base_weight.abs().sum()
    return loss