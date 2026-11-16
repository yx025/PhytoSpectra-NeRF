#!/usr/bin/env python3
"""Task 7 — End-to-end smoke test for JAI training pipeline.

Builds the full pipeline (dataparser -> dataset -> datamanager -> model -> loss)
and runs a few training iterations to prove the wiring is functional.

This bypasses ns-train CLI to avoid unrelated import issues (e.g. gsplat
version mismatch in splatfacto) while exercising the actual training path.

Usage
-----
::
    python scripts/6.test_jai_training.py data/JAI/3.18.2026/cucumber1
    python scripts/6.test_jai_training.py data/JAI/3.18.2026/cucumber1 --num-iters 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.jai_dataparser import JaiDataParserConfig
from nerfstudio.data.datasets.jai_dataset import JaiDataset
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.models.jai_nerfacto import JaiNerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.process_data.jai_utils import DEFAULT_EFFECTIVE_BITS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 7 — End-to-end JAI training smoke test.",
    )
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-rays", type=int, default=256)
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS)
    parser.add_argument("--file-ext", type=str, default="raw", choices=["raw", "npy"])
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    device = torch.device("cpu")
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

    print(f"Scene: {scene_dir}")
    print(f"Iterations: {args.num_iters}, Rays/batch: {args.num_rays}")
    print()

    # ------------------------------------------------------------------
    # Build pipeline config
    # ------------------------------------------------------------------
    print("=== Pipeline construction ===")

    pipeline_config = VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[JaiDataset],
            dataparser=JaiDataParserConfig(
                data=scene_dir,
                effective_bits=args.effective_bits,
                file_ext=args.file_ext,
            ),
            train_num_rays_per_batch=args.num_rays,
            eval_num_rays_per_batch=args.num_rays,
            pixel_sampler=PixelSamplerConfig(),
        ),
        model=JaiNerfactoModelConfig(
            implementation="torch",  # CPU-friendly
            bayer_loss_mult=1.0,
            nir_loss_mult=1.0,
        ),
    )

    try:
        pipeline = pipeline_config.setup(device=device, test_mode="test")
        pipeline.to(device)
        _check("pipeline construction", True)
    except Exception as e:
        _check("pipeline construction", False, str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _check("pipeline type", type(pipeline).__name__ == "VanillaPipeline")
    _check("model type", type(pipeline.model).__name__ == "JaiNerfactoModel")

    # ------------------------------------------------------------------
    # Check datamanager
    # ------------------------------------------------------------------
    print("\n=== DataManager ===")
    dm = pipeline.datamanager
    _check("datamanager has train_dataset",
           hasattr(dm, "train_dataset") and dm.train_dataset is not None)
    _check("train_dataset is JaiDataset",
           type(dm.train_dataset).__name__ == "JaiDataset")
    n_train = len(dm.train_dataset)
    _check("train_dataset non-empty", n_train > 0, f"n={n_train}")

    # ------------------------------------------------------------------
    # Get one training batch
    # ------------------------------------------------------------------
    print("\n=== Training batch ===")

    try:
        ray_bundle, batch = dm.next_train(step=0)
        _check("next_train succeeded", True)
    except Exception as e:
        _check("next_train succeeded", False, str(e))
        import traceback
        traceback.print_exc()
        print(f"\n  Batch creation failed. Cannot proceed with training.")
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passes} passed, {failures} failed")
        sys.exit(1)

    print(f"  ray_bundle: origins={list(ray_bundle.origins.shape)}, "
          f"directions={list(ray_bundle.directions.shape)}")
    print(f"  batch keys: {sorted(batch.keys())}")

    for key in ("source0_normalized", "source1_normalized", "source0_bayer_mask"):
        if key in batch:
            v = batch[key]
            _check(f"batch['{key}'] shape/dtype",
                   isinstance(v, torch.Tensor),
                   f"shape={list(v.shape)}, dtype={v.dtype}")
        else:
            _check(f"batch['{key}'] present", False, "missing from batch")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    print("\n=== Forward pass ===")

    pipeline.model.train()
    try:
        model_outputs = pipeline.model.get_outputs(ray_bundle)
        _check("forward pass succeeded", True)
    except Exception as e:
        _check("forward pass succeeded", False, str(e))
        import traceback
        traceback.print_exc()
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passes} passed, {failures} failed")
        sys.exit(1)

    _check("rgb_visible in outputs", "rgb_visible" in model_outputs,
           f"shape={list(model_outputs['rgb_visible'].shape)}" if "rgb_visible" in model_outputs else "")
    _check("nir in outputs", "nir" in model_outputs,
           f"shape={list(model_outputs['nir'].shape)}" if "nir" in model_outputs else "")

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    print("\n=== Loss computation ===")

    try:
        metrics_dict = pipeline.model.get_metrics_dict(model_outputs, batch)
        loss_dict = pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
        _check("get_loss_dict succeeded", True)
    except Exception as e:
        _check("get_loss_dict succeeded", False, str(e))
        import traceback
        traceback.print_exc()
        loss_dict = {}

    for loss_name in ("bayer_loss", "nir_loss"):
        if loss_name in loss_dict:
            v = loss_dict[loss_name]
            _check(f"{loss_name} finite and > 0",
                   torch.isfinite(v).item() and v.item() > 0,
                   f"value={v.item():.6f}")
        else:
            _check(f"{loss_name} in loss_dict", False)

    print(f"\n  Loss dict: {{{', '.join(f'{k}: {v.item():.6f}' for k, v in loss_dict.items())}}}")

    # ------------------------------------------------------------------
    # Multi-step training loop
    # ------------------------------------------------------------------
    print(f"\n=== Training loop ({args.num_iters} iterations) ===")

    # Build optimizers manually
    param_groups = pipeline.model.get_param_groups()
    optimizers = {}
    for name, params in param_groups.items():
        if len(params) > 0:
            optimizers[name] = torch.optim.Adam(params, lr=1e-3)

    losses_over_time = []
    try:
        for step in range(args.num_iters):
            # Get batch
            ray_bundle, batch = dm.next_train(step=step)

            # Forward
            model_outputs = pipeline.model.get_outputs(ray_bundle)

            # Loss
            metrics_dict = pipeline.model.get_metrics_dict(model_outputs, batch)
            loss_dict = pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
            total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

            # Backward
            for opt in optimizers.values():
                opt.zero_grad()
            total_loss.backward()
            for opt in optimizers.values():
                opt.step()

            bl = loss_dict.get("bayer_loss", torch.tensor(0)).item()
            nl = loss_dict.get("nir_loss", torch.tensor(0)).item()
            losses_over_time.append((bl, nl))
            print(f"    step {step}: total={total_loss.item():.6f}  "
                  f"bayer={bl:.6f}  nir={nl:.6f}")

        _check(f"training loop completed ({args.num_iters} steps)", True)
    except Exception as e:
        _check(f"training loop completed", False, str(e))
        import traceback
        traceback.print_exc()

    if losses_over_time:
        _check("all bayer losses finite",
               all(torch.isfinite(torch.tensor(bl)).item() for bl, _ in losses_over_time))
        _check("all nir losses finite",
               all(torch.isfinite(torch.tensor(nl)).item() for _, nl in losses_over_time))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  TRAINING SMOKE TEST SUMMARY")
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
