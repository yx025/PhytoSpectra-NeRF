"""JAI FS-3200D dual-source raw image loading utilities.

Supports:
- Source0: BayerRG12 (visible, Bayer mosaic)
- Source1: Mono12 (NIR, single-channel)

Both sources are saved as headerless .raw (uint16 little-endian) and optionally .npy.
Metadata (source name, frame index, width, height, bit depth) is parsed from filenames
when they follow the pattern: Source0_000123_1920x1200_16b.raw

Container vs effective bit depth
--------------------------------
The filename encodes the *container* (storage) bit depth — typically 16b for uint16
files.  The *effective* (sensor) bit depth may be smaller.  For the JAI FS-3200D both
BayerRG12 and Mono12 sensors produce 12-bit data stored in a 16-bit container, so
pixel values occupy the range 0–4095 even though the file is uint16.

Normalization defaults to the effective bit depth (12) so that ``normalize_raw()``
maps 0–4095 → 0.0–1.0.  The container bit depth is still stored in metadata for
provenance but is NOT used for normalization unless explicitly requested.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Filename metadata parsing
# ---------------------------------------------------------------------------

# Pattern: Source0_000123_1920x1200_16b.raw  (or .npy)
_FILENAME_RE = re.compile(
    r"^(?P<source>Source\d+)"
    r"_(?P<frame>\d+)"
    r"_(?P<width>\d+)x(?P<height>\d+)"
    r"_(?P<bitdepth>\d+)b"
    r"\.(?P<ext>raw|npy)$"
)

SUPPORTED_BIT_DEPTHS = (8, 10, 12, 14, 16)

# Default effective (sensor) bit depth for JAI FS-3200D.
# Both BayerRG12 and Mono12 produce 12-bit data in a 16-bit container.
DEFAULT_EFFECTIVE_BITS = 12


@dataclass(frozen=True)
class RawFileMetadata:
    """Metadata extracted from a JAI raw filename.

    Attributes:
        container_bits: Bit depth encoded in the filename (e.g. 16 for ``16b``).
            This is the *storage* bit depth — the dtype width of the raw file.
        effective_bits: The actual sensor bit depth used for normalization
            (e.g. 12 for BayerRG12 / Mono12).  Pixel values are in
            ``0 .. 2**effective_bits - 1``.
    """

    source: str           # e.g. "Source0", "Source1"
    frame_idx: int        # e.g. 123
    width: int            # pixels
    height: int           # pixels
    container_bits: int   # from filename, e.g. 16
    effective_bits: int   # sensor bit depth, e.g. 12
    ext: str              # "raw" or "npy"
    path: Path            # full path to the file

    @property
    def is_bayer(self) -> bool:
        """Source0 is BayerRG; all others are mono."""
        return self.source == "Source0"

    @property
    def is_nir(self) -> bool:
        return self.source == "Source1"

    @property
    def expected_raw_bytes(self) -> int:
        """Expected file size for headerless uint16 .raw."""
        return self.width * self.height * 2  # uint16 = 2 bytes


def parse_filename(
    path: Path,
    effective_bits: Optional[int] = None,
) -> RawFileMetadata:
    """Parse JAI raw filename into structured metadata.

    Args:
        path: Path to a .raw or .npy file whose stem matches the expected pattern.
        effective_bits: Override the effective (sensor) bit depth.  If *None*,
            defaults to ``DEFAULT_EFFECTIVE_BITS`` (12 for JAI FS-3200D).

    Returns:
        RawFileMetadata with all parsed fields.

    Raises:
        ValueError: If the filename does not match the expected pattern or
            contains an unsupported bit depth.
    """
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected pattern "
            f"'Source<N>_<frame>_<W>x<H>_<bits>b.(raw|npy)': {path.name}"
        )

    container_bits = int(match.group("bitdepth"))
    if container_bits not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            f"Unsupported container bit depth {container_bits} in '{path.name}'. "
            f"Supported: {SUPPORTED_BIT_DEPTHS}"
        )

    eff = effective_bits if effective_bits is not None else DEFAULT_EFFECTIVE_BITS
    if eff not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            f"Unsupported effective bit depth {eff}. "
            f"Supported: {SUPPORTED_BIT_DEPTHS}"
        )
    if eff > container_bits:
        raise ValueError(
            f"effective_bits ({eff}) cannot exceed container_bits ({container_bits})."
        )

    return RawFileMetadata(
        source=match.group("source"),
        frame_idx=int(match.group("frame")),
        width=int(match.group("width")),
        height=int(match.group("height")),
        container_bits=container_bits,
        effective_bits=eff,
        ext=match.group("ext"),
        path=path,
    )


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------

def load_raw(
    path: Path,
    width: int,
    height: int,
) -> npt.NDArray[np.uint16]:
    """Load a headerless .raw file as a uint16 (H, W) array.

    The file is expected to contain exactly ``width * height`` uint16 values
    stored in little-endian byte order (the default on x86).

    Args:
        path: Path to the .raw file.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        numpy array of shape (height, width) with dtype uint16.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file size does not match the expected dimensions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    expected_bytes = width * height * 2
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for '{path.name}': "
            f"expected {expected_bytes} bytes ({width}x{height} uint16), "
            f"got {actual_bytes} bytes."
        )

    data = np.fromfile(path, dtype=np.dtype("<u2"))  # little-endian uint16
    return data.reshape((height, width))


def load_npy(path: Path) -> npt.NDArray[np.uint16]:
    """Load a .npy file and return it as a uint16 2-D array.

    Args:
        path: Path to the .npy file.

    Returns:
        numpy array of shape (H, W) with dtype uint16.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded array is not 2-D.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Npy file not found: {path}")

    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array from '{path.name}', got shape {arr.shape}."
        )
    return arr.astype(np.uint16)


def load_npy_float(path: Path) -> npt.NDArray[np.float32]:
    """Load a .npy file as a float32 2-D array WITHOUT integer coercion.

    Used by the reflectance input mode (``JaiDataParserConfig.intensity_mode =
    "reflectance"``).  Values are assumed to already be in the physically
    meaningful reflectance range ``[0, 1]`` (possibly a bit above 1 for
    specular pixels).  No normalization is applied.

    Args:
        path: Path to the .npy file.

    Returns:
        numpy array of shape (H, W) with dtype float32.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Npy file not found: {path}")

    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array from '{path.name}', got shape {arr.shape}."
        )
    return arr.astype(np.float32, copy=False)


def load_jai_raw(
    path: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    effective_bits: Optional[int] = None,
) -> Tuple[npt.NDArray[np.uint16], RawFileMetadata]:
    """Load a JAI raw image, dispatching by file extension.

    If *width* and *height* are not provided they are inferred from the
    filename (which must follow the naming convention).

    Args:
        path: Path to a .raw or .npy file.
        width: Override width (optional; parsed from filename if omitted).
        height: Override height (optional; parsed from filename if omitted).
        effective_bits: Override effective sensor bit depth (default: 12).

    Returns:
        Tuple of (image array [H, W] uint16, parsed metadata).

    Raises:
        ValueError: On format/dimension mismatches.
    """
    path = Path(path)
    meta = parse_filename(path, effective_bits=effective_bits)

    w = width if width is not None else meta.width
    h = height if height is not None else meta.height

    if meta.ext == "raw":
        arr = load_raw(path, w, h)
    elif meta.ext == "npy":
        arr = load_npy(path)
        if arr.shape != (h, w):
            raise ValueError(
                f"Npy array shape {arr.shape} does not match expected "
                f"({h}, {w}) from filename '{path.name}'."
            )
    else:
        raise ValueError(f"Unsupported extension '.{meta.ext}' for '{path.name}'.")

    return arr, meta


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_raw(
    image: npt.NDArray[np.uint16],
    bit_depth: int = DEFAULT_EFFECTIVE_BITS,
    black_level: float = 0.0,
    white_level: Optional[float] = None,
) -> npt.NDArray[np.float32]:
    """Normalize a raw uint16 image to float32 in [0, 1].

    Applies:  out = clamp((image - black_level) / (white_level - black_level), 0, 1)

    The *bit_depth* parameter should be the **effective** (sensor) bit depth,
    NOT the container/storage bit depth.  For the JAI FS-3200D this is 12
    (values 0–4095) even though files are stored as uint16.

    Args:
        image: Raw image array (H, W), dtype uint16.
        bit_depth: **Effective** sensor bit depth.  Default is
            ``DEFAULT_EFFECTIVE_BITS`` (12).  This determines the default
            *white_level* = ``2**bit_depth - 1``.
        black_level: Sensor black level (in raw DN units). Default 0.
        white_level: Sensor white/saturation level.  If *None*, computed as
            ``2**bit_depth - 1`` (= 4095 for 12-bit).

    Returns:
        float32 array of shape (H, W) in [0, 1].

    Raises:
        ValueError: If bit_depth is not supported or white_level <= black_level.
    """
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            f"Unsupported bit depth {bit_depth}. Supported: {SUPPORTED_BIT_DEPTHS}"
        )

    if white_level is None:
        white_level = float(2**bit_depth - 1)

    if white_level <= black_level:
        raise ValueError(
            f"white_level ({white_level}) must be greater than "
            f"black_level ({black_level})."
        )

    out = (image.astype(np.float32) - black_level) / (white_level - black_level)
    np.clip(out, 0.0, 1.0, out=out)
    return out


# ---------------------------------------------------------------------------
# Per-source intrinsics (optional, populated from calibration or transforms.json)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceIntrinsics:
    """Per-source camera intrinsics.

    Populated when calibration data is available (e.g. from transforms.json or
    a separate calibration file).  All fields are required except distortion,
    which is model-dependent and optional.

    For the JAI FS-3200D, Source0 (visible/Bayer) and Source1 (NIR) may have
    slightly different intrinsics even though they share the same viewpoint,
    because the prism-based spectral split can introduce per-channel offsets.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: Optional[Tuple[float, ...]] = None
    """Distortion coefficients, model-dependent (e.g. (k1, k2, p1, p2, k3) for OpenCV)."""
    distortion_model: Optional[str] = None
    """Distortion model name, e.g. ``"opencv"``, ``"fisheye"``, ``"none"``."""


# ---------------------------------------------------------------------------
# Frame pair dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JaiFramePair:
    """A validated Source0 + Source1 pair from the same capture instant.

    Both sources share the same viewpoint — the JAI FS-3200D uses a prism to
    split light onto two sensors from a single optical axis.  There is no
    cross-camera extrinsic transform between Source0 and Source1.

    ``viewpoint_id`` explicitly encodes this: both sources in a pair share the
    same viewpoint identity.  For sequential captures, ``viewpoint_id`` equals
    ``frame_idx``.
    """

    frame_idx: int
    source0: RawFileMetadata   # BayerRG12
    source1: RawFileMetadata   # Mono12 NIR
    width: int
    height: int
    container_bits: int
    effective_bits: int
    viewpoint_id: int          # Same for Source0 and Source1 within a pair


# ---------------------------------------------------------------------------
# Scene directory scanner
# ---------------------------------------------------------------------------

def scan_scene_directory(
    scene_dir: Path,
    effective_bits: int = DEFAULT_EFFECTIVE_BITS,
    file_ext: str = "raw",
) -> Tuple[List[JaiFramePair], List[str]]:
    """Scan a scene directory and pair Source0 / Source1 files by frame index.

    Args:
        scene_dir: Path to the scene root containing raw files.
        effective_bits: Effective sensor bit depth (default: 12).
        file_ext: File extension to scan for (``"raw"`` or ``"npy"``).

    Returns:
        Tuple of:
        - List of validated ``JaiFramePair`` objects, sorted by frame index.
        - List of human-readable warning strings (skipped frames, etc.).

    Raises:
        FileNotFoundError: If *scene_dir* does not exist.
        ValueError: If no valid pairs are found, or on irrecoverable
            inconsistencies (e.g. mixed resolutions across the scene).
    """
    scene_dir = Path(scene_dir)
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # Collect all parseable files grouped by (source, frame_idx)
    by_source: Dict[str, Dict[int, RawFileMetadata]] = defaultdict(dict)
    parse_errors: List[str] = []

    for p in sorted(scene_dir.glob(f"*.{file_ext}")):
        try:
            meta = parse_filename(p, effective_bits=effective_bits)
        except ValueError as e:
            parse_errors.append(str(e))
            continue
        by_source[meta.source][meta.frame_idx] = meta

    warnings: List[str] = list(parse_errors)

    source0_frames = by_source.get("Source0", {})
    source1_frames = by_source.get("Source1", {})

    if not source0_frames:
        raise ValueError(f"No Source0 .{file_ext} files found in {scene_dir}")
    if not source1_frames:
        raise ValueError(f"No Source1 .{file_ext} files found in {scene_dir}")

    # Determine paired frame indices
    paired_indices = sorted(set(source0_frames.keys()) & set(source1_frames.keys()))

    # Report unpaired frames
    s0_only = set(source0_frames.keys()) - set(source1_frames.keys())
    s1_only = set(source1_frames.keys()) - set(source0_frames.keys())
    if s0_only:
        warnings.append(
            f"Source0-only frames (no Source1 match), skipped: "
            f"{sorted(s0_only)}"
        )
    if s1_only:
        warnings.append(
            f"Source1-only frames (no Source0 match), skipped: "
            f"{sorted(s1_only)}"
        )

    if not paired_indices:
        raise ValueError(
            f"No paired Source0+Source1 frames found in {scene_dir}. "
            f"Source0 frames: {sorted(source0_frames.keys())}, "
            f"Source1 frames: {sorted(source1_frames.keys())}"
        )

    # Validate consistency across all paired frames
    pairs: List[JaiFramePair] = []
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None
    ref_container: Optional[int] = None

    for idx in paired_indices:
        s0 = source0_frames[idx]
        s1 = source1_frames[idx]

        # Validate dimensions match within the pair
        if (s0.width, s0.height) != (s1.width, s1.height):
            warnings.append(
                f"Frame {idx}: dimension mismatch Source0=({s0.width}x{s0.height}) "
                f"vs Source1=({s1.width}x{s1.height}), skipped."
            )
            continue

        # Validate container bit depth matches within the pair
        if s0.container_bits != s1.container_bits:
            warnings.append(
                f"Frame {idx}: container bit depth mismatch Source0={s0.container_bits}b "
                f"vs Source1={s1.container_bits}b, skipped."
            )
            continue

        # Validate consistency with scene-wide reference
        if ref_w is None:
            ref_w, ref_h, ref_container = s0.width, s0.height, s0.container_bits
        else:
            if (s0.width, s0.height) != (ref_w, ref_h):
                warnings.append(
                    f"Frame {idx}: resolution ({s0.width}x{s0.height}) differs from "
                    f"scene reference ({ref_w}x{ref_h}), skipped."
                )
                continue
            if s0.container_bits != ref_container:
                warnings.append(
                    f"Frame {idx}: container bits ({s0.container_bits}) differs from "
                    f"scene reference ({ref_container}), skipped."
                )
                continue

        pairs.append(JaiFramePair(
            frame_idx=idx,
            source0=s0,
            source1=s1,
            width=s0.width,
            height=s0.height,
            container_bits=s0.container_bits,
            effective_bits=s0.effective_bits,
            viewpoint_id=idx,
        ))

    if not pairs:
        raise ValueError(
            f"All paired frames failed validation in {scene_dir}. "
            f"See warnings: {warnings}"
        )

    return pairs, warnings


# ---------------------------------------------------------------------------
# Optional transforms.json reader
# ---------------------------------------------------------------------------

def load_transforms_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a transforms.json file if it exists.

    This is a forward-looking helper.  The current JAI data does not include
    transforms.json; when it does (e.g. after running COLMAP or manual
    calibration), this function will read it.

    Args:
        path: Path to transforms.json.

    Returns:
        Parsed dict, or None if the file does not exist.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_intrinsics_block(d: Dict[str, Any]) -> Optional[SourceIntrinsics]:
    """Parse a per-source intrinsics dict into a ``SourceIntrinsics``.

    Expected keys: ``fx``, ``fy``, ``cx``, ``cy``, ``width``, ``height``.
    Optional keys: ``distortion_coeffs`` (list of floats),
    ``distortion_model`` (string).

    Returns None if the dict is missing required keys.
    """
    required = ("fx", "fy", "cx", "cy", "width", "height")
    if not all(k in d for k in required):
        return None
    dist = d.get("distortion_coeffs")
    return SourceIntrinsics(
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        width=int(d["width"]),
        height=int(d["height"]),
        distortion_coeffs=tuple(dist) if dist is not None else None,
        distortion_model=d.get("distortion_model"),
    )


def parse_source_intrinsics(
    transforms: Optional[Dict[str, Any]],
) -> Tuple[Optional[SourceIntrinsics], Optional[SourceIntrinsics]]:
    """Extract per-source intrinsics from a loaded transforms.json dict.

    Looks for top-level keys ``"source0_intrinsics"`` and
    ``"source1_intrinsics"``.  Each should be a dict with at least
    ``fx, fy, cx, cy, width, height``.

    Args:
        transforms: Parsed transforms.json dict, or None.

    Returns:
        (source0_intrinsics, source1_intrinsics) — either or both may be None
        if the data is not present or incomplete.
    """
    if transforms is None:
        return None, None
    s0 = transforms.get("source0_intrinsics")
    s1 = transforms.get("source1_intrinsics")
    return (
        _parse_intrinsics_block(s0) if isinstance(s0, dict) else None,
        _parse_intrinsics_block(s1) if isinstance(s1, dict) else None,
    )
