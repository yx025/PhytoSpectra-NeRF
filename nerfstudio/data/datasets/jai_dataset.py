"""JAI FS-3200D dual-source dataset.

Loads paired Source0 (BayerRG12) and Source1 (Mono12 NIR) raw images from disk,
normalizes them, and returns structured per-sample dicts suitable for the JAI
training pipeline.

This dataset consumes metadata produced by ``JaiDataParser`` and calls the raw
loading utilities in ``jai_utils.py`` and the Bayer mask utilities in
``bayer_utils.py``.

Training dict (returned by ``get_data()`` / ``__getitem__``)
-------------------------------------------------------------
Only image-shaped tensors compatible with the pixel sampler and collation:

=======================  ==============  ===========================================
Key                      Shape / dtype   Description
=======================  ==============  ===========================================
``image_idx``            int             Index into the dataset (for camera lookup)
``image``                (H,W,3) f32     Source0 mosaic repeated 3× (viewer compat)
``source0_normalized``   (H,W,1) f32     Source0 normalized to [0,1] via effective bits
``source1_normalized``   (H,W,1) f32     Source1 normalized to [0,1] via effective bits
``source0_bayer_mask``   (H,W,3) f32     One-hot BayerRG mask [R,G,B] per pixel
=======================  ==============  ===========================================

Full inspection dict (returned by ``get_full_sample()``)
---------------------------------------------------------
All of the above plus raw arrays, metadata scalars, and file paths.  Used by
inspection/test scripts, NOT by the training pipeline.

Normalization
-------------
``normalize_raw()`` is called with ``bit_depth=effective_bits`` (default 12),
NOT ``container_bits`` (16).  This maps raw DN values 0–4095 to float32 0.0–1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.process_data.jai_utils import (
    JaiFramePair,
    load_raw,
    load_npy,
    load_npy_float,
    normalize_raw,
)
from nerfstudio.utils.bayer_utils import apply_bayer_black_level_rggb, bayer_mask_3ch


class JaiDataset(InputDataset):
    """Dataset for JAI FS-3200D dual-source raw captures.

    Overrides ``get_data()`` to bypass the PIL-based image loading in the base
    class, since JAI raw files are headerless binary (not PNG/JPEG).

    Expects ``dataparser_outputs.metadata`` to contain the keys produced by
    ``JaiDataParser``:

    - ``jai_pairs``: list of ``JaiFramePair``
    - ``effective_bits``: int
    - ``container_bits``: int
    - ``width``, ``height``: int
    - ``source0_paths``, ``source1_paths``: list of Path
    - ``frame_indices``: list of int
    """

    # Large tensors — keep off GPU until the sampler moves them.
    exclude_batch_keys_from_device: List[str] = [
        "image",
        "source0_normalized",
        "source1_normalized",
        "source0_bayer_mask",
        "plant_mask",
    ]

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
    ):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)

        # Unpack metadata from the dataparser
        self._pairs: List[JaiFramePair] = list(self.metadata["jai_pairs"])
        self._effective_bits: int = int(self.metadata["effective_bits"])
        self._container_bits: int = int(self.metadata["container_bits"])
        self._width: int = int(self.metadata["width"])
        self._height: int = int(self.metadata["height"])
        self._shared_viewpoint: bool = bool(self.metadata.get("shared_viewpoint", True))

        # --- Intensity mode (DN vs reflectance) ---
        # "dn":         Source0/1 are raw uint16 DN; divide by 2**effective_bits - 1.
        # "reflectance": Source0/1 are float32 .npy already in [0,1]; pass through.
        self._intensity_mode: str = str(self.metadata.get("intensity_mode", "dn"))
        if self._intensity_mode not in ("dn", "reflectance"):
            raise ValueError(
                f"Unknown intensity_mode={self._intensity_mode!r}; expected 'dn' or 'reflectance'."
            )

        # Optional black levels (only applied in DN mode; reflectance files already
        # have black level subtracted at preprocessing time).
        self._bayer_black_level = self.metadata.get("bayer_black_level")  # tuple[4] or None
        self._nir_black_level = self.metadata.get("nir_black_level")      # float or None

        # --- Plant mask directory (optional) ---
        # When set, each frame loads an additional (H, W, 1) float32 mask from
        # ``<plant_mask_dir>/<Source0-stem>.mask.npy``.  Missing masks degrade
        # gracefully to all-ones (= no masking) with a one-time warning.
        _mask_dir = self.metadata.get("plant_mask_dir")
        self._plant_mask_dir = None if _mask_dir is None else Path(_mask_dir)
        self._plant_mask_missing_warned: bool = False

        # Pre-compute the Bayer mask once (it's deterministic from dimensions).
        # Shape: (H, W, 3) float32, one-hot [R, G, B] per pixel.
        self._bayer_mask: np.ndarray = bayer_mask_3ch(self._height, self._width)

        if scale_factor != 1.0:
            raise NotImplementedError(
                "JaiDataset does not support scale_factor != 1.0 yet. "
                "Raw Bayer data cannot be naively resized without breaking "
                "the mosaic pattern alignment."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def _load_raw_array(self, pair: JaiFramePair, source: str) -> np.ndarray:
        """Load a raw (H, W) uint16 array for the given source of a pair.

        Used only in ``intensity_mode = "dn"``.

        Args:
            pair: The frame pair.
            source: ``"source0"`` or ``"source1"``.

        Returns:
            numpy array of shape (H, W) with dtype uint16.
        """
        meta = pair.source0 if source == "source0" else pair.source1
        path = meta.path

        if meta.ext == "raw":
            arr = load_raw(path, meta.width, meta.height)
        elif meta.ext == "npy":
            arr = load_npy(path)
            if arr.shape != (meta.height, meta.width):
                raise ValueError(
                    f"Shape mismatch: {path.name} has shape {arr.shape}, "
                    f"expected ({meta.height}, {meta.width})."
                )
        else:
            raise ValueError(f"Unsupported extension: {meta.ext}")

        return arr

    def _load_reflectance_array(self, pair: JaiFramePair, source: str) -> np.ndarray:
        """Load a float32 reflectance array (H, W) in [0, 1] for the given source.

        Used only in ``intensity_mode = "reflectance"``.  Requires the file to
        be a .npy; .raw is ambiguous about dtype and is not supported here.
        """
        meta = pair.source0 if source == "source0" else pair.source1
        path = meta.path
        if meta.ext != "npy":
            raise ValueError(
                f"intensity_mode='reflectance' requires file_ext='npy' so that "
                f"reflectance float32 data can be loaded directly, but got "
                f"{path.name!r} (ext={meta.ext!r})."
            )
        arr = load_npy_float(path)
        if arr.shape != (meta.height, meta.width):
            raise ValueError(
                f"Shape mismatch: {path.name} has shape {arr.shape}, "
                f"expected ({meta.height}, {meta.width})."
            )
        return arr

    def _load_and_normalize(self, pair: JaiFramePair):
        """Load both sources for a pair and produce [0,1] torch tensors.

        Returns:
            (s0_raw, s1_raw, s0_norm_t, s1_norm_t, bayer_mask_t)

            In DN mode ``s0_raw`` / ``s1_raw`` are uint16 arrays.
            In reflectance mode they are float32 arrays already in [0,1]; they
            are returned as-is (useful for inspection/export).
        """
        bayer_mask_t = torch.from_numpy(self._bayer_mask)     # (H, W, 3)

        if self._intensity_mode == "reflectance":
            s0_ref = self._load_reflectance_array(pair, "source0")  # float32 [0,1]
            s1_ref = self._load_reflectance_array(pair, "source1")  # float32 [0,1]

            # Clamp defensively — specular highlights can exceed 1.0 slightly.
            # Upper bound configurable (JaiDataParserConfig.reflectance_clip_max, default 1.0 = original); NaN passes through.
            cmax = float(self.metadata.get("reflectance_clip_max", 1.0))
            s0_norm = np.clip(s0_ref, 0.0, cmax)
            s1_norm = np.clip(s1_ref, 0.0, cmax)

            s0_norm_t = torch.from_numpy(np.ascontiguousarray(s0_norm)).unsqueeze(-1)
            s1_norm_t = torch.from_numpy(np.ascontiguousarray(s1_norm)).unsqueeze(-1)
            return s0_ref, s1_ref, s0_norm_t, s1_norm_t, bayer_mask_t

        # --- DN mode (default) -------------------------------------------------
        s0_raw = self._load_raw_array(pair, "source0")
        s1_raw = self._load_raw_array(pair, "source1")

        assert s0_raw.shape == (pair.height, pair.width), (
            f"Source0 shape {s0_raw.shape} != expected ({pair.height}, {pair.width})"
        )
        assert s1_raw.shape == (pair.height, pair.width), (
            f"Source1 shape {s1_raw.shape} != expected ({pair.height}, {pair.width})"
        )

        # --- Bayer (Source0) black level ---
        # Per-Bayer-channel subtraction when the user supplies a (R, G1, G2, B)
        # tuple; otherwise the default-path behavior (no subtraction) is
        # preserved byte-for-byte.
        if self._bayer_black_level is not None:
            s0_bl_sub = apply_bayer_black_level_rggb(
                s0_raw, tuple(self._bayer_black_level),  # type: ignore[arg-type]
            )
            s0_norm = normalize_raw(
                np.clip(s0_bl_sub, 0, None).astype(np.uint16),
                bit_depth=self._effective_bits,
            )
        else:
            s0_norm = normalize_raw(s0_raw, bit_depth=self._effective_bits)

        # --- NIR (Source1) black level ---
        s1_bl = float(self._nir_black_level) if self._nir_black_level is not None else 0.0
        s1_norm = normalize_raw(
            s1_raw, bit_depth=self._effective_bits, black_level=s1_bl,
        )

        s0_norm_t = torch.from_numpy(s0_norm).unsqueeze(-1)   # (H, W, 1)
        s1_norm_t = torch.from_numpy(s1_norm).unsqueeze(-1)   # (H, W, 1)

        return s0_raw, s1_raw, s0_norm_t, s1_norm_t, bayer_mask_t

    def _load_plant_mask(self, pair: JaiFramePair) -> torch.Tensor:
        """Return a (H, W, 1) float32 plant mask in [0, 1] for this frame.

        Matching rule: ``plant_mask_dir / (pair.source0.path.stem + ".mask.npy")``.
        If the file is missing, returns all ones (= no masking) and warns once.
        The mask is clamped to [0, 1] so callers can safely use it as a loss
        weight without extra checks.
        """
        h, w = pair.height, pair.width
        if self._plant_mask_dir is None:
            return torch.ones((h, w, 1), dtype=torch.float32)
        path = self._plant_mask_dir / (pair.source0.path.stem + ".mask.npy")
        if not path.exists():
            if not self._plant_mask_missing_warned:
                print(
                    f"[JaiDataset] plant_mask missing: {path}. "
                    f"Falling back to all-ones mask (no masking). "
                    f"Future warnings suppressed."
                )
                self._plant_mask_missing_warned = True
            return torch.ones((h, w, 1), dtype=torch.float32)
        arr = np.load(path)
        if arr.shape != (h, w):
            raise ValueError(
                f"plant_mask shape {arr.shape} != expected ({h}, {w}) for {path.name}"
            )
        t = torch.from_numpy(arr.astype(np.float32, copy=False)).clamp(0.0, 1.0)
        return t.unsqueeze(-1)  # (H, W, 1)

    def get_data(self, image_idx: int, image_type: str = "float32") -> Dict:
        """Load and return a training-compatible sample for one paired frame.

        Returns ONLY image-shaped tensors that survive collation and pixel
        sampling.  Scalar metadata and Path objects are excluded — use
        ``get_full_sample()`` for inspection scripts.

        Args:
            image_idx: Index into the dataset.
            image_type: Ignored. Present for base class compatibility.

        Returns:
            Dict with training-compatible keys only.
        """
        pair = self._pairs[image_idx]
        _, _, s0_norm_t, s1_norm_t, bayer_mask_t = self._load_and_normalize(pair)
        plant_mask_t = self._load_plant_mask(pair)   # (H, W, 1) float32 in [0,1]

        return {
            "image_idx": image_idx,
            "image": s0_norm_t.expand(-1, -1, 3),   # (H, W, 3) for viewer compatibility
            "source0_normalized": s0_norm_t,        # (H, W, 1)
            "source1_normalized": s1_norm_t,        # (H, W, 1)
            "source0_bayer_mask": bayer_mask_t,     # (H, W, 3)
            "plant_mask": plant_mask_t,             # (H, W, 1)  1.0=plant, 0.0=bg
        }

    def get_full_sample(self, image_idx: int) -> Dict:
        """Load and return the complete sample with all metadata.

        This includes raw arrays, scalar metadata, and Path objects that are
        NOT suitable for the training pipeline's collation/pixel sampling, but
        are useful for inspection and test scripts.
        """
        pair = self._pairs[image_idx]
        s0_raw, s1_raw, s0_norm_t, s1_norm_t, bayer_mask_t = (
            self._load_and_normalize(pair)
        )

        s0_raw_t = torch.from_numpy(s0_raw.astype(np.int32))
        s1_raw_t = torch.from_numpy(s1_raw.astype(np.int32))

        return {
            "image_idx": image_idx,
            "image": s0_norm_t,
            "source0_raw": s0_raw_t,
            "source1_raw": s1_raw_t,
            "source0_normalized": s0_norm_t,
            "source1_normalized": s1_norm_t,
            "source0_bayer_mask": bayer_mask_t,
            "source_id": 0,
            "frame_index": pair.frame_idx,
            "viewpoint_id": pair.viewpoint_id,
            "shared_viewpoint": self._shared_viewpoint,
            "source0_path": pair.source0.path,
            "source1_path": pair.source1.path,
            "width": pair.width,
            "height": pair.height,
            "container_bits": pair.container_bits,
            "effective_bits": pair.effective_bits,
        }

    def get_metadata(self, data: Dict) -> Dict:
        """Return additional metadata for the sample.

        For JaiDataset, all metadata is already included in ``get_data()``,
        so this returns an empty dict.
        """
        return {}
