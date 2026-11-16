"""JAI FS-3200D field: NerfactoField with configurable output channels.

This is a thin wrapper around ``NerfactoField`` that overrides the MLP color
head to produce an arbitrary number of output channels.  For the JAI dual-source
setup:

- Visible branch: ``JaiField(num_channels=3)`` predicts RGB.
- NIR branch: ``JaiField(num_channels=1)`` predicts single-channel NIR.

Structurally identical to ``ThermalNerfactoField`` but kept as a separate class
so the JAI code path is fully decoupled from the thermal pipeline.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField


class JaiField(NerfactoField):
    """NerfactoField with a configurable number of output channels.

    The only difference from ``NerfactoField`` is that ``self.mlp_head`` is
    rebuilt with ``out_dim=num_channels`` (default 3 for the base class).
    """

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_channels: int = 3,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        output_scale: float = 1.0,
        output_init: Optional[float] = None,
        average_init_density: float = 1.0,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            implementation=implementation,
            average_init_density=average_init_density,
        )
        # Override the MLP head to output the desired number of channels.
        # The base class builds an mlp_head with out_dim=3 (RGB).
        # We rebuild it with out_dim=num_channels.
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=num_channels,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )
        # Optional scaled sigmoid: output in (0, output_scale).  Only when output_scale != 1 is the head wrapped, so default
        # models keep the original module tree (and load old checkpoints unchanged).  Callers that use field.mlp_head directly
        # (e.g. scripts/jai_embfix_common.branch_shade) get the scaled output automatically (2026-10-27).
        self.output_scale = float(output_scale)
        self.output_init = None if output_init is None else float(output_init)
        if self.output_scale != 1.0 and self.output_init is not None:
            # Centred scaled sigmoid (2026-10-29, user-approved NIR fix): the head outputs raw values, then
            # output_scale * sigmoid(z + b) with a learnable bias b initialised to logit(output_init / output_scale), so the
            # initial output is about output_init (e.g. 0.5, as the unscaled head) instead of output_scale / 2, while the output
            # range stays (0, output_scale).  Only used when output_init is set; existing checkpoints keep the legacy head.
            self.mlp_head = nn.Sequential(
                MLP(
                    in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                    num_layers=num_layers_color,
                    layer_width=hidden_dim_color,
                    out_dim=num_channels,
                    activation=nn.ReLU(),
                    out_activation=None,
                    implementation=implementation,
                ),
                _ScaledSigmoid(self.output_scale, self.output_init),
            )
        elif self.output_scale != 1.0:
            self.mlp_head = nn.Sequential(self.mlp_head, _Scale(self.output_scale))


class _ScaledSigmoid(nn.Module):
    """scale * sigmoid(x + b), learnable scalar bias b initialised so that the output at x = 0 equals `init`."""

    def __init__(self, scale: float, init: float) -> None:
        super().__init__()
        assert 0.0 < init < scale, (init, scale)
        self.scale = float(scale)
        self.bias = nn.Parameter(torch.tensor(math.log((init / scale) / (1.0 - init / scale)), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * torch.sigmoid(x.float() + self.bias)


class _Scale(nn.Module):
    """Multiply by a fixed factor (no parameters)."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = float(factor)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.factor
