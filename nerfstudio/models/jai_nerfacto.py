"""JAI FS-3200D dual-branch nerfacto model.

Two separate neural radiance fields share the same scene geometry but predict
different spectral outputs:

- **Visible branch** (``field_visible``): predicts 3-channel RGB, supervised
  via Bayer-select loss against Source0 raw data.
- **NIR branch** (``field_nir``): predicts 1-channel NIR, supervised via
  reconstruction loss against Source1 raw data.

Both branches share a viewpoint per frame (``shared_viewpoint = True``) — there
is no cross-camera extrinsic transform between Source0 and Source1.

Architecture follows the same pattern as ``ThermalNerfactoModel`` (separate
density fields with per-branch proposal networks) but:

- Replaces the ``is_thermal`` masking paradigm with paired Source0/Source1 samples.
- Uses ``bayer_select_loss()`` instead of thermal pixel-wise loss.
- Uses ``nir_reconstruction_loss()`` instead of thermal reconstruction.
- Does NOT use ``RGBTRenderer``, ``tv_pixel_loss``, ``cross_channel_loss``,
  or any thermal-specific regularization.

This model is NOT yet registered as a training method.  Full ``ns-train``
integration is deferred to a later task.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.jai_field import JaiField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.jai_losses import (
    bayer_select_loss,
    nir_reconstruction_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JaiNerfactoModelConfig(ModelConfig):
    """Configuration for the JAI dual-branch nerfacto model."""

    _target: Type = field(default_factory=lambda: JaiNerfactoModel)

    # --- Architecture ---
    near_plane: float = 0.05
    far_plane: float = 1000.0
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    hidden_dim: int = 64
    hidden_dim_color: int = 64
    hidden_dim_transient: int = 64
    num_levels: int = 16
    base_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    num_nerf_samples_per_ray: int = 48
    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_slope: float = 10.0
    proposal_weights_anneal_max_num_iters: int = 1000
    use_single_jitter: bool = True
    predict_normals: bool = False
    disable_scene_contraction: bool = False
    use_gradient_scaling: bool = False
    implementation: Literal["tcnn", "torch"] = "tcnn"
    use_appearance_embedding: bool = True
    use_average_appearance_embedding: bool = True
    appearance_embed_dim: int = 32
    average_init_density: float = 0.01
    use_average_init_density: bool = False
    """Pass average_init_density to both fields and all proposal networks (as stock nerfacto does).  Default False keeps the
    historical JAI behaviour (multiplier 1.0, the config value was never applied) so existing checkpoints render unchanged.
    When True, a buffer ``average_init_density_applied`` is saved in the checkpoint as a record."""

    # --- JAI-specific loss weights ---
    bayer_loss_mult: float = 1.0
    """Bayer-select loss multiplier for Source0 visible branch."""
    nir_loss_mult: float = 1.0
    """NIR reconstruction loss multiplier for Source1 NIR branch."""
    use_plant_mask_loss: bool = False
    nir_output_scale: float = 1.0
    """Upper bound of the NIR branch output (scaled sigmoid).  1.0 = original behaviour.  Use with
    JaiDataParserConfig.reflectance_clip_max >= this value to model unclipped panel-relative NIR (2026-10-27)."""
    nir_output_init: Optional[float] = None
    """Only with nir_output_scale != 1: initial NIR output of the centred scaled-sigmoid head (learnable bias before the
    sigmoid), e.g. 0.5 = the unscaled head's initial value.  None = legacy head (scale * sigmoid), which existing
    checkpoints use (2026-10-29)."""
    mask_nonfinite_targets: bool = False
    """If True, pixels whose Source0 OR Source1 target is NaN/inf (saturated raw pixels written as NaN by the recalibration)
    are excluded from the bayer and NIR losses of both branches.  False = original behaviour (2026-10-27)."""
    """If True and the dataset emits a ``plant_mask`` key (see
    ``JaiDataParserConfig.plant_mask_dir``), both the Bayer and NIR
    reconstruction losses are weighted by the per-ray mask value before
    reduction.  Background rays (mask=0) contribute zero gradient; plant
    rays (mask=1) are unchanged; soft masks are interpolated.

    This is a **foreground-masked loss**, NOT foreground-only pixel sampling.
    Rays are still drawn uniformly from the full image; only their loss
    weight is masked.  Full foreground-only sampling is future work —
    implementing it cleanly requires a custom pixel sampler or rejection
    sampling inside the datamanager, which is deliberately out of scope
    for this change.

    Default False → the dataset may still carry the mask but training is
    byte-for-byte identical to the old unmasked path."""
    bayer_loss_criterion: Literal["l1", "mse"] = "l1"
    """Loss criterion for Bayer-select loss."""
    nir_loss_criterion: Literal["l1", "mse"] = "l1"
    """Loss criterion for NIR reconstruction loss."""
    interlevel_loss_mult: float = 1.0
    """Proposal interlevel loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0
    """Ref-NeRF orientation loss multiplier for the visible branch, applied
    only when ``predict_normals=True``.  Default 0.0 means the loss is wired
    but does not affect training — set to a positive value (vanilla nerfacto
    uses 0.0001) to actually regularize normals.  Keeping the default at 0.0
    preserves byte-for-byte backward compatibility on existing DN /
    reflectance runs that never trained with normals."""
    pred_normal_loss_mult: float = 0.0
    """Predicted-normal loss multiplier for the visible branch, applied only
    when ``predict_normals=True``.  Default 0.0 means no effect.  Vanilla
    nerfacto uses 0.001.  This drives the MLP's predicted normals to match
    the density-gradient normals."""

    # --- Camera optimizer ---
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3")
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class JaiNerfactoModel(Model):
    """JAI dual-branch nerfacto model.

    Two separate fields + proposal networks predict visible RGB (3ch) and NIR
    (1ch) from the same scene.  Both branches share the same camera poses
    (shared viewpoint).
    """

    config: JaiNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = (
            self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0
        )

        # average_init_density: applied only with the switch (default off = historical multiplier 1.0)
        init_density = float(self.config.average_init_density) if getattr(self.config, "use_average_init_density", False) else 1.0
        if getattr(self.config, "use_average_init_density", False):
            self.register_buffer("average_init_density_applied", torch.tensor(init_density))

        # Shared field kwargs
        field_kwargs = dict(
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            implementation=self.config.implementation,
            average_init_density=init_density,
        )

        # --- Visible branch (3-channel RGB) ---
        self.field_visible = JaiField(
            self.scene_box.aabb, num_channels=3, **field_kwargs,
        )

        # --- NIR branch (1-channel) ---
        self.field_nir = JaiField(
            self.scene_box.aabb, num_channels=1, output_scale=getattr(self.config, "nir_output_scale", 1.0), output_init=getattr(self.config, "nir_output_init", None), **field_kwargs,
        )

        # --- Camera optimizer (shared viewpoint — one set of poses) ---
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu",
        )

        # --- Proposal networks (one set per branch) ---
        def _build_proposal_nets():
            nets = torch.nn.ModuleList()
            density_fns = []
            num_prop = self.config.num_proposal_iterations
            if self.config.use_same_proposal_network:
                assert len(self.config.proposal_net_args_list) == 1
                net = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **self.config.proposal_net_args_list[0],
                    implementation=self.config.implementation,
                    average_init_density=init_density,
                )
                nets.append(net)
                density_fns.extend([net.density_fn for _ in range(num_prop)])
            else:
                for i in range(num_prop):
                    args = self.config.proposal_net_args_list[
                        min(i, len(self.config.proposal_net_args_list) - 1)
                    ]
                    net = HashMLPDensityField(
                        self.scene_box.aabb,
                        spatial_distortion=scene_contraction,
                        **args,
                        implementation=self.config.implementation,
                        average_init_density=init_density,
                    )
                    nets.append(net)
                density_fns.extend([n.density_fn for n in nets])
            return nets, density_fns

        self.proposal_networks_visible, self.density_fns_visible = _build_proposal_nets()
        self.proposal_networks_nir, self.density_fns_nir = _build_proposal_nets()

        # --- Samplers ---
        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        initial_sampler = None
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        sampler_kwargs = dict(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )
        self.proposal_sampler_visible = ProposalNetworkSampler(**sampler_kwargs)
        self.proposal_sampler_nir = ProposalNetworkSampler(**sampler_kwargs)

        # --- Collider ---
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane,
        )

        # --- Renderers ---
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_nir = RGBRenderer(
            background_color=self.config.background_color, num_channels=1,
            clamp_max=max(1.0, float(getattr(self.config, "nir_output_scale", 1.0))),
        )
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        # Normals renderer — only used when predict_normals is True; otherwise
        # idle.  Construction is unconditional so state_dict stays consistent
        # across config changes.
        self.renderer_normals = NormalsRenderer()
        self.normals_shader = NormalsShader()

        # --- Metrics ---
        from torchmetrics.image import PeakSignalNoiseRatio

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.step = 0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _branch_forward(
        self,
        ray_bundle: RayBundle,
        field: JaiField,
        renderer: RGBRenderer,
        proposal_sampler: ProposalNetworkSampler,
        density_fns: list,
    ) -> Dict[str, Tensor]:
        """Run one branch (visible or NIR) through proposal sampling + field + rendering."""
        ray_samples, weights_list, ray_samples_list = proposal_sampler(
            ray_bundle, density_fns=density_fns,
        )
        field_outputs = field.forward(
            ray_samples, compute_normals=self.config.predict_normals,
        )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(
                field_outputs, ray_samples,
            )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "density": field_outputs[FieldHeadNames.DENSITY],
        }

        # Per-vertex normals for point-cloud / mesh export.  Only active when
        # the config asks for normals so the default training path is
        # unchanged.  When training AND predict_normals, we additionally
        # capture the Ref-NeRF orientation loss and the predicted-normal
        # consistency loss.  Both use the un-rendered per-sample normals and
        # matching weights (detached where nerfacto detaches them), so they
        # can be applied at the top level in get_loss_dict.
        if self.config.predict_normals:
            normals = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights,
            )
            pred_normals = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights,
            )
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

            if self.training:
                outputs["rendered_orientation_loss"] = orientation_loss(
                    weights.detach(),
                    field_outputs[FieldHeadNames.NORMALS],
                    ray_bundle.directions,
                )
                outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                    weights.detach(),
                    field_outputs[FieldHeadNames.NORMALS].detach(),
                    field_outputs[FieldHeadNames.PRED_NORMALS],
                )
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        return outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        # Deep copy BEFORE camera optimizer (optimizer creates non-leaf tensors
        # that can't be deepcopied).
        ray_bundle_nir = copy.deepcopy(ray_bundle)

        # Shared camera optimizer (same poses for both branches)
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
            self.camera_optimizer.apply_to_raybundle(ray_bundle_nir)

        # Set near/far planes on the ray bundles
        ray_bundle = self.collider(ray_bundle)
        ray_bundle_nir = self.collider(ray_bundle_nir)

        # Visible branch (3ch RGB)
        vis_outputs = self._branch_forward(
            ray_bundle,
            self.field_visible,
            self.renderer_rgb,
            self.proposal_sampler_visible,
            self.density_fns_visible,
        )

        # NIR branch (1ch)
        nir_outputs = self._branch_forward(
            ray_bundle_nir,
            self.field_nir,
            self.renderer_nir,
            self.proposal_sampler_nir,
            self.density_fns_nir,
        )

        # Merge outputs with prefixed keys
        outputs: Dict[str, Tensor] = {}

        # Standard keys expected by the viewer (aliases for visible branch)
        outputs["rgb"] = vis_outputs["rgb"]                      # (N, 3)
        outputs["accumulation"] = vis_outputs["accumulation"]
        outputs["depth"] = vis_outputs["depth"]

        # Normals (only present when predict_normals=True) — surface at the
        # top level so the point-cloud exporter can pick them up via its
        # `normal_output_name` (default: "normals").  Also propagate the
        # rendered orientation / pred-normal loss tensors when training so
        # get_loss_dict can add them without re-running the field.
        if "normals" in vis_outputs:
            outputs["normals"] = vis_outputs["normals"]
            outputs["pred_normals"] = vis_outputs["pred_normals"]
        if self.training and "rendered_orientation_loss" in vis_outputs:
            outputs["rendered_orientation_loss"] = vis_outputs["rendered_orientation_loss"]
            outputs["rendered_pred_normal_loss"] = vis_outputs["rendered_pred_normal_loss"]

        # Branch-specific keys
        outputs["rgb_visible"] = vis_outputs["rgb"]              # (N, 3)
        outputs["accumulation_visible"] = vis_outputs["accumulation"]
        outputs["depth_visible"] = vis_outputs["depth"]

        outputs["nir"] = nir_outputs["rgb"]                      # (N, 1)
        outputs["accumulation_nir"] = nir_outputs["accumulation"]
        outputs["depth_nir"] = nir_outputs["depth"]

        # --- Display-only viewer outputs (no spatial context needed) ---
        if not self.training:
            # Grayscale luminance from visible branch — avoids misleading
            # unconstrained-channel artifacts in the raw RGB display.
            vis_rgb = vis_outputs["rgb"]  # (N, 3)
            lum = (0.299 * vis_rgb[..., 0:1]
                   + 0.587 * vis_rgb[..., 1:2]
                   + 0.114 * vis_rgb[..., 2:3])   # (N, 1)
            outputs["visible_gray"] = lum.expand(*lum.shape[:-1], 3)  # (N, 3)

            # NIR as 3-channel grayscale for viewer display
            nir_1ch = nir_outputs["rgb"]  # (N, 1)
            outputs["nir_gray"] = nir_1ch.expand(*nir_1ch.shape[:-1], 3)  # (N, 3)

            # --- Per-channel single-band reflectance views ---
            # Expose each Bayer channel of the visible branch (and NIR) as its
            # own single-channel output.  In the viewer these appear as separate
            # entries in the "Output Render" dropdown, each with a colormap
            # selector (pick "gray" to read absolute reflectance magnitude, or
            # keep the default "turbo" for high-contrast false-color).
            #
            # Colormap `normalize` defaults to False (see ColormapOptions), so
            # the value shown is the true reflectance in [0, 1] — NOT rescaled
            # per-frame.  That is what makes cross-channel comparison valid:
            # e.g. on a shaded/occluded leaf the G band reads relatively
            # brighter than the R and B bands because the overlying leaf
            # transmits green while absorbing red/blue.
            #
            # Names are intensity-mode-agnostic (the value is per-channel
            # reflectance in `intensity_mode="reflectance"`, per-channel
            # normalized DN in `"dn"`).  Kept as 1-channel so the viewer offers
            # the colormap dropdown; the base aggregator reshapes to (H, W, 1).
            outputs["visible_r"] = vis_rgb[..., 0:1]  # (N, 1) — R band
            outputs["visible_g"] = vis_rgb[..., 1:2]  # (N, 1) — G band
            outputs["visible_b"] = vis_rgb[..., 2:3]  # (N, 1) — B band
            outputs["nir_single"] = nir_1ch           # (N, 1) — NIR band

            # --- Greenness / channel-dominance views ---
            # The raw per-channel outputs above are shown on a fixed [0, 1]
            # colormap, so a small absolute lead of one channel (e.g. G only
            # ~0.03 above R and B on a leaf) is visually imperceptible even
            # though it is numerically real.  These derived outputs encode
            # *which channel dominates* directly, so a slight green lead pops:
            #
            #   visible_exg   = 2G - R - B   (Excess Green index, McKinnon/Woebbecke)
            #                   > 0 where green dominates; classic vegetation
            #                   greenness measure.  Single-channel → in the
            #                   viewer pick "turbo"/"gray"; bright = greener.
            #   visible_g_chroma = G / (R + G + B)   (green chromaticity)
            #                   illumination/shading-normalized; ~0.33 is
            #                   neutral, > 0.33 means green-dominant.
            #
            # Both are display-only diagnostics; they do NOT feed any loss.
            r_ch, g_ch, b_ch = vis_rgb[..., 0:1], vis_rgb[..., 1:2], vis_rgb[..., 2:3]
            outputs["visible_exg"] = 2.0 * g_ch - r_ch - b_ch          # (N, 1)
            outputs["visible_g_chroma"] = g_ch / (r_ch + g_ch + b_ch + 1e-6)  # (N, 1)

        if self.training and "weights_list" in vis_outputs:
            outputs["weights_list_visible"] = vis_outputs["weights_list"]
            outputs["ray_samples_list_visible"] = vis_outputs["ray_samples_list"]
            outputs["weights_list_nir"] = nir_outputs["weights_list"]
            outputs["ray_samples_list_nir"] = nir_outputs["ray_samples_list"]

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Override to add spatially-aware Bayer mosaic viewer output."""
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Add bayer_mosaic: apply BayerRG mask to the visible RGB prediction
        # to show what the model predicts at each supervised Bayer position.
        if "rgb" in outputs and outputs["rgb"].dim() == 3:
            H, W, _ = outputs["rgb"].shape
            from nerfstudio.utils.bayer_utils import bayer_mask_3ch
            mask = torch.from_numpy(bayer_mask_3ch(H, W)).to(outputs["rgb"].device)  # (H,W,3)
            bayer_val = (outputs["rgb"] * mask).sum(dim=-1, keepdim=True)  # (H,W,1)
            outputs["bayer_mosaic"] = bayer_val.expand(-1, -1, 3)  # (H,W,3) grayscale

        return outputs

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def get_loss_dict(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        metrics_dict: Dict = None,
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = {}

        # ------------------------------------------------------------------
        # Supervision separation (important — do not collapse into one signal)
        # ------------------------------------------------------------------
        # The JAI model has TWO independent fields (visible + NIR), each with
        # its own proposal network and renderer, sharing ONLY the camera pose.
        # Supervision below is split accordingly:
        #
        #   - bayer_loss  reads ONLY ``batch["source0_normalized"]`` and
        #                 supervises ONLY ``outputs["rgb_visible"]``.
        #                 Target and prediction never touch Source1.
        #   - nir_loss    reads ONLY ``batch["source1_normalized"]`` and
        #                 supervises ONLY ``outputs["nir"]``.
        #                 Target and prediction never touch Source0.
        #
        # This is true in both ``intensity_mode="dn"`` and
        # ``intensity_mode="reflectance"`` because the dataset emits both
        # tensors unchanged across modes — only the numeric range of the
        # tensor differs between modes.  There is no point in the code where
        # the two supervision signals are averaged, concatenated, or
        # otherwise collapsed.

        # --- Foreground plant mask (optional) ---
        # Gated on a model-config flag so default behavior is byte-for-byte
        # identical to the unmasked path.  See JaiDataParserConfig.plant_mask_dir
        # and JaiNerfactoModelConfig.use_plant_mask_loss.
        use_mask = (
            self.config.use_plant_mask_loss
            and "plant_mask" in batch
            and batch["plant_mask"] is not None
        )
        plant_w: Tensor = None  # type: ignore[assignment]
        if use_mask:
            plant_w = batch["plant_mask"].to(self.device).reshape(-1)  # (N,)

        # --- Non-finite target masking (optional, default off) ---
        # Saturated raw pixels are stored as NaN in the recalibrated reflectance; a pixel is dropped from BOTH losses when either
        # source is non-finite, so the two branches see the same pixel set.  Implemented through the existing weighted path.
        if getattr(self.config, "mask_nonfinite_targets", False):   # getattr: configs saved before 2026-10-27 lack the field
            t0 = batch["source0_normalized"].to(self.device); t1 = batch["source1_normalized"].to(self.device)
            finite_w = (torch.isfinite(t0).reshape(t0.shape[0], -1).all(-1) & torch.isfinite(t1).reshape(t1.shape[0], -1).all(-1)).float()
            batch = dict(batch); batch["source0_normalized"] = torch.nan_to_num(t0, nan=0.0, posinf=0.0, neginf=0.0); batch["source1_normalized"] = torch.nan_to_num(t1, nan=0.0, posinf=0.0, neginf=0.0)
            plant_w = finite_w if plant_w is None else plant_w * finite_w
            use_mask = True

        # --- Bayer-select loss (Source0 visible branch) ---
        pred_rgb = outputs["rgb_visible"]                       # (N, 3)
        target_s0 = batch["source0_normalized"].to(self.device)  # (N, 1)
        bayer_mask = batch["source0_bayer_mask"].to(self.device)  # (N, 3)

        if use_mask:
            per_pixel_bayer = bayer_select_loss(
                pred_rgb, target_s0, bayer_mask,
                criterion=self.config.bayer_loss_criterion,
                reduction="none",
            )  # (N,)
            denom = plant_w.sum().clamp_min(1.0)
            bayer_loss = (per_pixel_bayer * plant_w).sum() / denom
        else:
            bayer_loss = bayer_select_loss(
                pred_rgb, target_s0, bayer_mask,
                criterion=self.config.bayer_loss_criterion,
            )
        loss_dict["bayer_loss"] = self.config.bayer_loss_mult * bayer_loss

        # --- NIR reconstruction loss (Source1) ---
        pred_nir = outputs["nir"]                                # (N, 1)
        target_s1 = batch["source1_normalized"].to(self.device)   # (N, 1)

        if use_mask:
            per_pixel_nir = nir_reconstruction_loss(
                pred_nir, target_s1,
                criterion=self.config.nir_loss_criterion,
                reduction="none",
            )  # (N,)
            denom = plant_w.sum().clamp_min(1.0)
            nir_loss = (per_pixel_nir * plant_w).sum() / denom
        else:
            nir_loss = nir_reconstruction_loss(
                pred_nir, target_s1,
                criterion=self.config.nir_loss_criterion,
            )
        loss_dict["nir_loss"] = self.config.nir_loss_mult * nir_loss

        # --- Proposal interlevel + distortion losses ---
        if self.training:
            for suffix in ("_visible", "_nir"):
                wl_key = f"weights_list{suffix}"
                rl_key = f"ray_samples_list{suffix}"
                if wl_key in outputs:
                    il = self.config.interlevel_loss_mult * interlevel_loss(
                        outputs[wl_key], outputs[rl_key],
                    )
                    dl = self.config.distortion_loss_mult * distortion_loss(
                        outputs[wl_key], outputs[rl_key],
                    )
                    loss_dict[f"interlevel_loss{suffix}"] = il
                    loss_dict[f"distortion_loss{suffix}"] = dl

            # --- Normal losses (Ref-NeRF orientation + predicted-normal) ---
            # Gated on both ``predict_normals`` AND a positive weight, so the
            # default config is a zero-impact wire-up.  These terms come from
            # the VISIBLE branch only — the NIR branch does not predict
            # normals (num_channels=1 field).
            if (
                self.config.predict_normals
                and "rendered_orientation_loss" in outputs
            ):
                if self.config.orientation_loss_mult > 0.0:
                    loss_dict["orientation_loss"] = (
                        self.config.orientation_loss_mult
                        * torch.mean(outputs["rendered_orientation_loss"])
                    )
                if self.config.pred_normal_loss_mult > 0.0:
                    loss_dict["pred_normal_loss"] = (
                        self.config.pred_normal_loss_mult
                        * torch.mean(outputs["rendered_pred_normal_loss"])
                    )

            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics_dict(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        metrics_dict: Dict[str, float] = {}
        if self.training:
            for suffix in ("_visible", "_nir"):
                wl_key = f"weights_list{suffix}"
                rl_key = f"ray_samples_list{suffix}"
                if wl_key in outputs:
                    metrics_dict[f"distortion{suffix}"] = distortion_loss(
                        outputs[wl_key], outputs[rl_key],
                    )
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    # ------------------------------------------------------------------
    # Param groups
    # ------------------------------------------------------------------

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups: Dict[str, List[Parameter]] = {}
        param_groups["fields_visible"] = list(self.field_visible.parameters())
        param_groups["fields_nir"] = list(self.field_nir.parameters())
        param_groups["proposal_networks_visible"] = list(
            self.proposal_networks_visible.parameters()
        )
        param_groups["proposal_networks_nir"] = list(
            self.proposal_networks_nir.parameters()
        )
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    # ------------------------------------------------------------------
    # Training callbacks (proposal weight annealing)
    # ------------------------------------------------------------------

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        callbacks: List[TrainingCallback] = []

        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                self.step = step
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler_visible.set_anneal(anneal)
                self.proposal_sampler_nir.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler_visible.step_cb,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler_nir.step_cb,
                )
            )

        return callbacks

    # ------------------------------------------------------------------
    # Visualization (eval-time images)
    # ------------------------------------------------------------------

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        """Minimal image output for viewer / eval."""
        images_dict: Dict[str, Tensor] = {}

        # Visible branch RGB
        if "rgb_visible" in outputs:
            images_dict["rgb_visible"] = outputs["rgb_visible"]
        if "accumulation_visible" in outputs:
            images_dict["accumulation_visible"] = colormaps.apply_colormap(
                outputs["accumulation_visible"],
            )
        if "depth_visible" in outputs:
            images_dict["depth_visible"] = colormaps.apply_depth_colormap(
                outputs["depth_visible"],
                accumulation=outputs.get("accumulation_visible"),
            )

        # NIR branch (expand to 3ch for visualization)
        if "nir" in outputs:
            images_dict["nir"] = outputs["nir"].expand(-1, -1, 3)
        if "accumulation_nir" in outputs:
            images_dict["accumulation_nir"] = colormaps.apply_colormap(
                outputs["accumulation_nir"],
            )
        if "depth_nir" in outputs:
            images_dict["depth_nir"] = colormaps.apply_depth_colormap(
                outputs["depth_nir"],
                accumulation=outputs.get("accumulation_nir"),
            )

        metrics_dict: Dict[str, float] = {}
        return metrics_dict, images_dict
