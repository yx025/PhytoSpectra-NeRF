#!/usr/bin/env python3
"""Pose integration validation — verify real poses are loaded from ns-process-data.

Usage
-----
::
    python scripts/8.test_jai_poses.py \\
        data/JAI/3.18.2026/cucumber1 \\
        --poses-source data/JAI/3.18.2026/cucumber1/source0_png_nsproc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from nerfstudio.data.dataparsers.jai_dataparser import JaiDataParserConfig, JaiDataParser
from nerfstudio.process_data.jai_utils import DEFAULT_EFFECTIVE_BITS


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JAI pose integration.")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--poses-source", type=Path, required=True)
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS)
    parser.add_argument("--file-ext", type=str, default="raw", choices=["raw", "npy"])
    args = parser.parse_args()

    passes = 0
    failures = 0

    def _check(name, ok, detail=""):
        nonlocal passes, failures
        if ok:
            passes += 1
        else:
            failures += 1
        status = "PASS" if ok else "FAIL"
        msg = f"  {status}: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    scene_dir = args.scene_dir.resolve()
    poses_source = args.poses_source.resolve()

    print(f"Scene:        {scene_dir}")
    print(f"Poses source: {poses_source}")
    print()

    # ------------------------------------------------------------------
    # Build dataparser with real poses
    # ------------------------------------------------------------------
    print("=== Dataparser with real poses ===")
    config = JaiDataParserConfig(
        data=scene_dir,
        poses_source=poses_source,
        effective_bits=args.effective_bits,
        file_ext=args.file_ext,
    )
    dp = JaiDataParser(config)
    outputs = dp.get_dataparser_outputs("train")
    meta = outputs.metadata
    cams = outputs.cameras

    n = len(outputs.image_filenames)
    print(f"  Train frames: {n}")
    _check("has_real_poses is True", meta.get("has_real_poses") is True)
    _check("container_bits == 16", meta["container_bits"] == 16)
    _check("effective_bits == 12", meta["effective_bits"] == 12)

    # ------------------------------------------------------------------
    # Camera validation
    # ------------------------------------------------------------------
    print(f"\n=== Camera validation ===")
    raw_w, raw_h = meta["width"], meta["height"]
    print(f"  Raw resolution: {raw_w}x{raw_h}")
    print(f"  Cameras: n={len(cams)}, "
          f"fx={cams.fx[0].item():.2f}, fy={cams.fy[0].item():.2f}, "
          f"cx={cams.cx[0].item():.2f}, cy={cams.cy[0].item():.2f}")
    print(f"  Width={cams.width[0].item()}, Height={cams.height[0].item()}")

    _check("camera count == frame count", len(cams) == n)
    _check("camera width == raw width", cams.width[0].item() == raw_w)
    _check("camera height == raw height", cams.height[0].item() == raw_h)

    # Poses should NOT be identity
    c2w = cams.camera_to_worlds  # (N, 3, 4)
    identity_34 = torch.eye(4, dtype=torch.float32)[:3, :]
    is_all_identity = all(
        torch.allclose(c2w[i], identity_34, atol=1e-5) for i in range(n)
    )
    _check("poses are NOT all identity", not is_all_identity)

    # Check first pose is a valid rotation (det ≈ 1)
    R0 = c2w[0, :3, :3]
    det = torch.det(R0).item()
    _check("first pose rotation det ≈ 1", abs(det - 1.0) < 0.01, f"det={det:.6f}")

    # Translation should have some spread
    translations = c2w[:, :3, 3]  # (N, 3)
    t_range = translations.max(dim=0).values - translations.min(dim=0).values
    _check("translation has spatial spread", t_range.max().item() > 0.1,
           f"range={t_range.tolist()}")

    # Distortion params
    if cams.distortion_params is not None:
        dp_shape = list(cams.distortion_params.shape)
        _check("distortion_params shape", dp_shape == [n, 6], f"got {dp_shape}")
        k1 = cams.distortion_params[0, 0].item()
        _check("distortion k1 is nonzero", abs(k1) > 1e-6, f"k1={k1:.6f}")
    else:
        _check("distortion_params present", False, "None")

    # ------------------------------------------------------------------
    # Intrinsics scaling check
    # ------------------------------------------------------------------
    print(f"\n=== Intrinsics scaling ===")
    import json
    tj_path = poses_source / "transforms.json" if poses_source.is_dir() else poses_source
    with open(tj_path) as f:
        tj = json.load(f)
    tj_w, tj_h = tj["w"], tj["h"]
    scale_x = raw_w / tj_w
    scale_y = raw_h / tj_h
    expected_fx = tj["fl_x"] * scale_x
    expected_fy = tj["fl_y"] * scale_y
    print(f"  transforms.json res: {tj_w}x{tj_h}")
    print(f"  Scale factors: x={scale_x:.1f}, y={scale_y:.1f}")
    print(f"  Expected fx={expected_fx:.2f}, fy={expected_fy:.2f}")
    print(f"  Actual   fx={cams.fx[0].item():.2f}, fy={cams.fy[0].item():.2f}")

    _check("fx correctly scaled", abs(cams.fx[0].item() - expected_fx) < 0.01)
    _check("fy correctly scaled", abs(cams.fy[0].item() - expected_fy) < 0.01)

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------
    w = meta.get("jai_warnings", [])
    if w:
        print(f"\n=== Warnings ({len(w)}) ===")
        for wi in w:
            print(f"  - {wi}")

    # ------------------------------------------------------------------
    # Compare with placeholder mode
    # ------------------------------------------------------------------
    print(f"\n=== Comparison: placeholder mode ===")
    config_ph = JaiDataParserConfig(
        data=scene_dir, effective_bits=args.effective_bits, file_ext=args.file_ext,
    )
    dp_ph = JaiDataParser(config_ph)
    out_ph = dp_ph.get_dataparser_outputs("train")
    _check("placeholder has_real_poses is False",
           out_ph.metadata.get("has_real_poses") is False)
    # Placeholder poses are all identity
    c2w_ph = out_ph.cameras.camera_to_worlds
    is_ph_identity = all(
        torch.allclose(c2w_ph[i], identity_34, atol=1e-5)
        for i in range(len(out_ph.image_filenames))
    )
    _check("placeholder poses are identity", is_ph_identity)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  POSE INTEGRATION SUMMARY")
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
