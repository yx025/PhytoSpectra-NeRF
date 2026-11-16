#!/usr/bin/env python3
"""Task 6 — Structural validation of the JAI model / field path.

Instantiates JaiNerfactoModel with a real JAI dataset, runs a lightweight
forward pass with synthetic rays, and verifies that:
- the model constructs without error
- both branches produce outputs with expected shapes
- get_loss_dict correctly invokes bayer_select_loss and nir_reconstruction_loss
- param groups are correctly separated

This does NOT do real training — it validates the structural wiring only.

Usage
-----
::
    python scripts/5.test_jai_model.py data/JAI/3.18.2026/cucumber1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ImportError:
    print("ERROR: torch is not available.")
    sys.exit(1)

from nerfstudio.data.dataparsers.jai_dataparser import JaiDataParserConfig, JaiDataParser
from nerfstudio.data.datasets.jai_dataset import JaiDataset
from nerfstudio.process_data.jai_utils import DEFAULT_EFFECTIVE_BITS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 6 — Validate JAI model/field structural wiring.",
    )
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS)
    parser.add_argument("--file-ext", type=str, default="raw", choices=["raw", "npy"])
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    passes = 0
    failures = 0

    def _check(name: str, ok: bool, detail: str = ""):
        nonlocal passes, failures
        status = "PASS" if ok else "FAIL"
        if ok:
            passes += 1
        else:
            failures += 1
        msg = f"  {status}: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    print(f"Scene: {scene_dir}")
    config = JaiDataParserConfig(
        data=scene_dir,
        effective_bits=args.effective_bits,
        file_ext=args.file_ext,
    )
    dp = JaiDataParser(config)
    outputs = dp.get_dataparser_outputs("train")
    dataset = JaiDataset(dataparser_outputs=outputs, scale_factor=1.0)
    n = len(dataset)
    print(f"Dataset: {n} frames\n")

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
    print("=== Model construction ===")

    from nerfstudio.models.jai_nerfacto import JaiNerfactoModelConfig, JaiNerfactoModel

    model_config = JaiNerfactoModelConfig(
        implementation="torch",  # use pure torch for CPU testing
    )

    try:
        model = model_config.setup(
            scene_box=outputs.scene_box,
            num_train_data=n,
            metadata=outputs.metadata,
        )
        _check("model construction", True)
    except Exception as e:
        _check("model construction", False, str(e))
        print(f"\nRESULT: MODEL CONSTRUCTION FAILED.")
        sys.exit(1)

    model.train()
    _check("model is JaiNerfactoModel",
           type(model).__name__ == "JaiNerfactoModel")

    # ------------------------------------------------------------------
    # Check fields
    # ------------------------------------------------------------------
    print("\n=== Field structure ===")

    _check("field_visible exists", hasattr(model, "field_visible"))
    _check("field_nir exists", hasattr(model, "field_nir"))

    from nerfstudio.fields.jai_field import JaiField
    _check("field_visible is JaiField",
           isinstance(model.field_visible, JaiField))
    _check("field_nir is JaiField",
           isinstance(model.field_nir, JaiField))

    # Check MLP head output dims
    vis_out_dim = model.field_visible.mlp_head.out_dim
    nir_out_dim = model.field_nir.mlp_head.out_dim
    _check("field_visible mlp_head out_dim == 3",
           vis_out_dim == 3, f"got {vis_out_dim}")
    _check("field_nir mlp_head out_dim == 1",
           nir_out_dim == 1, f"got {nir_out_dim}")

    # ------------------------------------------------------------------
    # Check param groups
    # ------------------------------------------------------------------
    print("\n=== Param groups ===")
    param_groups = model.get_param_groups()
    expected_groups = [
        "fields_visible", "fields_nir",
        "proposal_networks_visible", "proposal_networks_nir",
    ]
    for g in expected_groups:
        _check(f"param group '{g}' exists",
               g in param_groups,
               f"params={len(param_groups.get(g, []))}")

    # ------------------------------------------------------------------
    # Synthetic forward pass
    # ------------------------------------------------------------------
    print("\n=== Synthetic forward pass ===")

    # Create a minimal synthetic ray bundle
    num_rays = 16
    origins = torch.zeros(num_rays, 3)
    origins[:, 2] = -1.0  # camera at z=-1
    directions = torch.zeros(num_rays, 3)
    directions[:, 2] = 1.0  # looking along +z

    from nerfstudio.cameras.rays import RayBundle

    ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=torch.ones(num_rays, 1),
        camera_indices=torch.zeros(num_rays, 1, dtype=torch.int32),
        nears=torch.ones(num_rays, 1) * model.config.near_plane,
        fars=torch.ones(num_rays, 1) * model.config.far_plane,
    )

    try:
        with torch.no_grad():
            model.eval()
            model_outputs = model.get_outputs(ray_bundle)
        _check("forward pass succeeded", True)
    except Exception as e:
        _check("forward pass succeeded", False, str(e))
        print(f"\nForward pass failed. Skipping output checks.")
        # Print summary and exit
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passes} passed, {failures} failed")
        print(f"{'='*60}")
        sys.exit(1 if failures > 0 else 0)

    # Check output shapes
    _check("rgb_visible in outputs", "rgb_visible" in model_outputs)
    _check("nir in outputs", "nir" in model_outputs)

    if "rgb_visible" in model_outputs:
        vis_shape = list(model_outputs["rgb_visible"].shape)
        _check("rgb_visible shape == (N, 3)",
               vis_shape == [num_rays, 3], f"got {vis_shape}")

    if "nir" in model_outputs:
        nir_shape = list(model_outputs["nir"].shape)
        _check("nir shape == (N, 1)",
               nir_shape == [num_rays, 1], f"got {nir_shape}")

    _check("accumulation_visible in outputs",
           "accumulation_visible" in model_outputs)
    _check("accumulation_nir in outputs",
           "accumulation_nir" in model_outputs)
    _check("depth_visible in outputs",
           "depth_visible" in model_outputs)
    _check("depth_nir in outputs",
           "depth_nir" in model_outputs)

    # ------------------------------------------------------------------
    # Test loss computation with real dataset sample
    # ------------------------------------------------------------------
    print("\n=== Loss computation (real dataset sample) ===")

    sample = dataset[0]

    # Simulate a batch with the correct shapes for loss computation.
    # In real training, the pixel sampler would flatten (H,W) into (N,).
    # Here we take a small patch for testing.
    patch_h, patch_w = 4, 4
    batch = {
        "source0_normalized": sample["source0_normalized"][:patch_h, :patch_w, :],  # (4,4,1)
        "source1_normalized": sample["source1_normalized"][:patch_h, :patch_w, :],  # (4,4,1)
        "source0_bayer_mask": sample["source0_bayer_mask"][:patch_h, :patch_w, :],  # (4,4,3)
    }

    # Create matching predictions
    pred_outputs = {
        "rgb_visible": torch.rand(patch_h, patch_w, 3),
        "nir": torch.rand(patch_h, patch_w, 1),
    }

    # Temporarily switch model to eval to skip training-only losses
    model.eval()
    try:
        loss_dict = model.get_loss_dict(pred_outputs, batch)
        _check("get_loss_dict succeeded", True)
    except Exception as e:
        _check("get_loss_dict succeeded", False, str(e))
        loss_dict = {}

    if loss_dict:
        _check("bayer_loss in loss_dict", "bayer_loss" in loss_dict)
        _check("nir_loss in loss_dict", "nir_loss" in loss_dict)

        if "bayer_loss" in loss_dict:
            bl = loss_dict["bayer_loss"].item()
            _check("bayer_loss is finite", torch.isfinite(loss_dict["bayer_loss"]).item(),
                   f"value={bl:.6f}")
            _check("bayer_loss > 0 (random pred)", bl > 0, f"value={bl:.6f}")

        if "nir_loss" in loss_dict:
            nl = loss_dict["nir_loss"].item()
            _check("nir_loss is finite", torch.isfinite(loss_dict["nir_loss"]).item(),
                   f"value={nl:.6f}")
            _check("nir_loss > 0 (random pred)", nl > 0, f"value={nl:.6f}")

        print(f"\n    Loss values: bayer_loss={loss_dict.get('bayer_loss', 'N/A')}, "
              f"nir_loss={loss_dict.get('nir_loss', 'N/A')}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  STRUCTURAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total checks : {passes + failures}")
    print(f"  Passed       : {passes}")
    print(f"  Failed       : {failures}")
    print()

    if failures > 0:
        print("RESULT: SOME CHECKS FAILED.")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
