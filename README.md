# PhytoSpectra-NeRF

A dual-branch neural radiance field for plant reflectance reconstruction from a
**JAI FS-3200D** two-sensor camera (Bayer visible + near-infrared).

The model (`jai-nerfacto`) trains two nerfacto-style fields that share one camera pose per frame:

- a **visible branch** that predicts R, G, B and is supervised directly on the raw Bayer mosaic
  (each pixel is compared only with the colour its sensor site measured, so no demosaicing is needed);
- a **NIR branch** that predicts one near-infrared channel.

Trained on panel-calibrated reflectance, the model exports a 3D point cloud that carries
four float reflectance values (R, G, B, NIR) and a surface normal per point.

This repository contains only the code needed to preprocess data, train the model, and export
point clouds. It does not contain data, trained models, or results.

The code is built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) v1.0.2. The `nerfstudio/` package is included
so that the repository runs after one `pip install -e .`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `nerfstudio/models/jai_nerfacto.py` | The dual-branch model (`jai-nerfacto`) |
| `nerfstudio/fields/jai_field.py` | Nerfacto field with a configurable number of output channels |
| `nerfstudio/data/dataparsers/jai_dataparser.py` | Dataparser (`jai-data`): pairs Source0/Source1 frames, loads poses |
| `nerfstudio/data/datasets/jai_dataset.py` | Dataset: normalized mosaic, NIR image, Bayer mask |
| `nerfstudio/model_components/jai_losses.py`, `jai_metrics.py` | Bayer-select loss, NIR loss, metrics |
| `nerfstudio/process_data/jai_utils.py`, `nerfstudio/utils/bayer_utils.py` | Raw file I/O and Bayer helpers |
| `nerfstudio/configs/method_configs.py`, `dataparser_configs.py` | Registration of `jai-nerfacto` and `jai-data` |
| `scripts/jai_pick_panel_roi_local.py` | Pick the calibration-panel region (needs a display) |
| `scripts/jai_estimate_black_level.py` | Black level from dark frames |
| `scripts/jai_compute_reflectance.py` | Raw DN to reflectance `.npy` |
| `scripts/jai_source0_to_jpg_wb.py` | White-balanced JPGs for COLMAP |
| `run_colmap_fixed_intrinsics_me.sh`, `run_colmap_fixed_intrinsics.sh` | COLMAP with frozen intrinsics (sequential / exhaustive matching) |
| `scripts/jai_pose_precheck.py`, `scripts/jai_flag_bad_frames.py` | Pose quality check, removal of bad frames |
| `scripts/jai_export_4ch_reflectance_pointcloud.py` | Export a 4-channel float reflectance point cloud |
| `scripts/jai_export_reflectance_pointcloud.py` | Export for models trained on raw DN |
| `scripts/jai_export_4ch_from_ckpt.py`, `jai_export_4ch_seeded.py` | Export from a checkpoint without the data; seeded export |
| `scripts/jai_export_embfix*.py`, `jai_embfix_common.py`, `jai_embfix_validate.py` | Embedding-corrected reflectance export and its validation |
| `scripts/1.test_jai_raw.py` … `8.test_jai_poses.py` | Step-by-step tests of each layer |

The other Nerfstudio and ThermalNeRF methods ship unchanged as part of the package; they are
not the subject of this repository.

---

## Installation

Tested on Linux with an NVIDIA GPU, Python 3.10, PyTorch 2.1.2, CUDA 11.8 and COLMAP 3.11.1.

```bash
git clone https://github.com/yx025/PhytoSpectra-NeRF.git
cd PhytoSpectra-NeRF

conda create -n phytospectra -y python=3.10
conda activate phytospectra
python -m pip install --upgrade pip setuptools wheel

# PyTorch + CUDA 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# tiny-cuda-nn (built from source; set the value to your GPU's compute capability:
# 75 Turing, 80 A100, 86 RTX 30xx / A6000, 89 RTX 40xx / Ada, 90 H100)
pip install ninja
CUDA_HOME="$CONDA_PREFIX" TCNN_CUDA_ARCHITECTURES=86 \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# COLMAP (for camera poses)
conda install -y -c conda-forge colmap=3.11.1

# This repository (editable) + Open3D for the export scripts
pip install -e .
pip install open3d
```

Check the installation:

```bash
python -c "
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.configs.dataparser_configs import dataparsers
assert 'jai-nerfacto' in method_configs and 'jai-data' in dataparsers
print('jai-nerfacto and jai-data are registered')
"
```

If tiny-cuda-nn does not build, the model also runs with the slower pure-PyTorch encoding:
add `--pipeline.model.implementation torch` to the training command.

---

## Data layout

One folder per scene. File names are parsed by the code and must follow this pattern:

```
Source0_000123_2048x1536_16b.raw     # Source0 = Bayer visible (RGGB)
Source1_000123_2048x1536_16b.raw     # Source1 = NIR, same frame index
```

`<source>_<frame index>_<width>x<height>_<bit depth>b.<raw|npy>`. A `.raw` file is a headerless
little-endian uint16 image. The two sources of one frame share one camera pose.

A scene folder after all preprocessing steps:

```
data/JAI/5.8/1.cucumber1p/
├── Source0_*.raw, Source1_*.raw    # captured frames
├── black/                          # dark frames (lens covered) + black_level.json
├── panel_roi.json                  # calibration-panel region
├── jpg/                            # JPGs for COLMAP
├── colmap_out/                     # poses (transforms.json)
└── reflectance/                    # Source0_*.npy, Source1_*.npy (float32 reflectance)
```

---

## Workflow

All commands are run from the repository root. The example scene is `data/JAI/5.8/1.cucumber1p`.

### 1. Pick the calibration-panel region (once per scene, on a machine with a display)

The picker reads a `.npy` image. Convert one frame of each source first:

```bash
python -c "
import numpy as np, sys
from pathlib import Path
from nerfstudio.process_data.jai_utils import load_jai_raw
for p in sys.argv[1:]:
    img, _ = load_jai_raw(Path(p)); np.save(Path(p).with_suffix('.npy'), img)
" data/JAI/5.8/1.cucumber1p/Source0_000000_2048x1536_16b.raw \
  data/JAI/5.8/1.cucumber1p/Source1_000000_2048x1536_16b.raw
```

Then click the top-left and bottom-right corners of the panel in each image:

```bash
python scripts/jai_pick_panel_roi_local.py \
    --npy  data/JAI/5.8/1.cucumber1p/Source0_000000_2048x1536_16b.npy \
    --kind bayer \
    --out  data/JAI/5.8/1.cucumber1p/panel_roi.json

python scripts/jai_pick_panel_roi_local.py \
    --npy  data/JAI/5.8/1.cucumber1p/Source1_000000_2048x1536_16b.npy \
    --kind nir \
    --out  data/JAI/5.8/1.cucumber1p/panel_roi.json
```

Both calls write into the same JSON file. The nominal panel reflectance defaults to 0.95 for
R, G, B and 0.52 for NIR; change it with `--panel-reflectance-bayer R G B` and
`--panel-reflectance-nir`, or edit the JSON. Delete the two helper `.npy` files afterwards so
that the scene folder holds only `.raw` frames.

### 2. Estimate the black level (from dark frames)

```bash
python scripts/jai_estimate_black_level.py \
    --scene "data/JAI/5.8/1.cucumber1p/black" \
    --out   "data/JAI/5.8/1.cucumber1p/black/black_level.json"
```

### 3. Make JPGs for COLMAP

```bash
python scripts/jai_source0_to_jpg_wb.py \
    --input-dir  "data/JAI/5.8/1.cucumber1p" \
    --output-dir "data/JAI/5.8/1.cucumber1p/jpg" \
    --input-ext  raw \
    --wb-mode    panel \
    --panel-roi  "data/JAI/5.8/1.cucumber1p/panel_roi.json" \
    --gamma      1.5
```

These JPGs are used only for pose estimation. Training never reads them.

### 4. Estimate camera poses with COLMAP

Open the script and set `INTRINSICS` to your own camera calibration
(`fx, fy, cx, cy, k1, k2, p1, p2`); the intrinsics are kept fixed during reconstruction.
Then pass the scene folder:

```bash
# one continuous orbit: sequential matching
./run_colmap_fixed_intrinsics_me.sh data/JAI/5.8/1.cucumber1p

# several orbits (for example two rings at different heights): exhaustive matching
./run_colmap_fixed_intrinsics.sh data/JAI/5.8/1.cucumber1p
```

The result is `data/JAI/5.8/1.cucumber1p/colmap_out/transforms.json`.

Optional checks before training:

```bash
python scripts/jai_pose_precheck.py --root data/JAI/5.8        # pose quality of every scene under the folder
python scripts/jai_flag_bad_frames.py --scene data/JAI/5.8/1.cucumber1p            # list bad frames
python scripts/jai_flag_bad_frames.py --scene data/JAI/5.8/1.cucumber1p --apply    # move them to excluded_frames/
```

Frames that COLMAP did not register should be removed: the dataparser gives an unmatched
frame an identity pose and would still train on it.

### 5. Compute reflectance

```bash
python scripts/jai_compute_reflectance.py \
    --scene       data/JAI/5.8/1.cucumber1p \
    --out         data/JAI/5.8/1.cucumber1p/reflectance \
    --panel-roi   data/JAI/5.8/1.cucumber1p/panel_roi.json \
    --black-level data/JAI/5.8/1.cucumber1p/black/black_level.json \
    --file-ext    raw
```

Reflectance is computed on the raw mosaic, with a separate panel value and black level for
each Bayer site (R, G1, G2, B), and written as float32 `.npy` with the same file names.

### 6. Train

**Route B: reflectance input (main route).**

```bash
nohup env CUDA_VISIBLE_DEVICES=3 ns-train jai-nerfacto \
  --viewer.quit-on-train-completion True \
  --pipeline.model.predict-normals True \
  --pipeline.model.orientation-loss-mult 0.0001 \
  --pipeline.model.pred-normal-loss-mult 0.001 \
  jai-data \
  --data "data/JAI/5.8/1.cucumber1p/reflectance" \
  --file-ext npy \
  --intensity-mode reflectance \
  --poses-source "data/JAI/5.8/1.cucumber1p/colmap_out" \
  > 5.8_cucumber1p_reflectance.log 2>&1 &
```

**Route A: raw digital numbers (no calibration panel needed).**

```bash
nohup env CUDA_VISIBLE_DEVICES=2 ns-train jai-nerfacto \
  --viewer.quit-on-train-completion True \
  --pipeline.model.predict-normals True \
  --pipeline.model.orientation-loss-mult 0.0001 \
  --pipeline.model.pred-normal-loss-mult 0.001 \
  jai-data \
  --data "data/JAI/5.8/1.cucumber1p" \
  --file-ext raw \
  --intensity-mode dn \
  --poses-source "data/JAI/5.8/1.cucumber1p/colmap_out" \
  > 5.8_cucumber1p_dn.log 2>&1 &
```

Training writes to `outputs/unnamed/jai-nerfacto/<timestamp>/`. The default length is
40,000 iterations (`--max-num-iterations`). Model options go before `jai-data`, data options after it.

Look at a trained model in the viewer:

```bash
ns-viewer --load-config outputs/unnamed/jai-nerfacto/<timestamp>/config.yml
```

### 7. Export a point cloud

**Route B model: four float reflectance channels.**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/jai_export_4ch_reflectance_pointcloud.py \
  --load-config outputs/unnamed/jai-nerfacto/<timestamp>/config.yml \
  --output exports/pcd/5.8/cucumber1p/refl_4ch.ply \
  --num-points 3000000 \
  --remove-outliers \
  --obb-center   0.0000000000 0.0000000000 -1.0000000000 \
  --obb-rotation 0.0000000000 0.0000000000  0.0000000000 \
  --obb-scale    2.0000000000 2.0000000000  2.0000000000
```

The PLY holds `x y z`, normals, 8-bit colour for viewing, and the float properties
`reflectance_r`, `reflectance_g`, `reflectance_b`, `reflectance_nir`. The three `--obb-*`
values define the crop box (centre, rotation, size) in the model's coordinates; read them
from the crop tool of the viewer.

**Route A model.** Same arguments, with `scripts/jai_export_reflectance_pointcloud.py`
(8-bit colour plus a float NIR property).

**Without the training data.** `scripts/jai_export_4ch_from_ckpt.py` takes the same
`--load-config`, `--output` and `--obb-*` arguments and needs only the checkpoint.

### 8. Embedding-corrected reflectance (optional)

Each training frame has its own appearance embedding, and a standard export uses the mean
embedding for all points. The scripts below recompute the reflectance of an exported
point cloud without that simplification. Geometry is kept exactly; only reflectance changes.

```bash
# A  : average over the appearance embeddings of all training frames
# A' : average over the frames that see the point, each with its own embedding
python scripts/jai_export_embfix.py \
  --load-config outputs/unnamed/jai-nerfacto/<timestamp>/config.yml \
  --old-ply     exports/pcd/5.8/cucumber1p/refl_4ch.ply \
  --out-dir     exports/pcd/5.8/cucumber1p/embfix

# A' variants: median over views, and views steeper than --max-angle removed
python scripts/jai_export_embfix_variants.py \
  --load-config outputs/unnamed/jai-nerfacto/<timestamp>/config.yml \
  --old-ply     exports/pcd/5.8/cucumber1p/refl_4ch.ply \
  --fix-dir     exports/pcd/5.8/cucumber1p/embfix

# compare old / A / A' with the measured pixels of the held-out frames
python scripts/jai_embfix_validate.py \
  --load-config outputs/unnamed/jai-nerfacto/<timestamp>/config.yml \
  --old-ply     exports/pcd/5.8/cucumber1p/refl_4ch.ply \
  --fix-dir     exports/pcd/5.8/cucumber1p/embfix
```

Outputs are `refl_4ch_A.ply` and `refl_4ch_Aprime.ply` (plus `_median` and `_nograze`
variants), with the same point order as the input PLY.

---

## Useful options

Data options (after `jai-data`):

| Option | Default | Meaning |
|--------|---------|---------|
| `--data` | | Folder with the `Source0_*` / `Source1_*` files |
| `--poses-source` | none | COLMAP output folder or a `transforms.json`. Without it, poses are placeholders |
| `--file-ext` | `raw` | `raw` or `npy` |
| `--intensity-mode` | `dn` | `dn` (uint16 raw values) or `reflectance` (float32 `.npy`, needs `--file-ext npy`) |
| `--effective-bits` | `12` | Sensor bit depth used to normalize raw values |
| `--bayer-black-level R G1 G2 B`, `--nir-black-level` | none | Black level in `dn` mode |
| `--reflectance-clip-max` | `1.0` | Upper clip of the reflectance input |
| `--eval-fraction` | `0.1` | Share of frames held out for evaluation |
| `--plant-mask-dir` | none | Folder of per-frame masks `<Source0 file stem>.mask.npy` (bool, raw resolution) |

Model options (before `jai-data`, prefix `--pipeline.model.`):

| Option | Default | Meaning |
|--------|---------|---------|
| `bayer-loss-mult`, `nir-loss-mult` | `1.0` | Weight of the visible and NIR losses |
| `predict-normals` | `False` | Predict surface normals (needed for normals in the export) |
| `orientation-loss-mult`, `pred-normal-loss-mult` | `0.0` | Normal regularizers, active only with `predict-normals True` |
| `use-plant-mask-loss` | `False` | Weight both losses by the plant mask from `--plant-mask-dir` |
| `nir-output-scale` | `1.0` | Upper bound of the NIR output; use with `--reflectance-clip-max` for NIR above 1 |
| `nir-output-init` | none | Initial NIR output when `nir-output-scale` is not 1 (for example `0.5`) |
| `mask-nonfinite-targets` | `False` | Leave pixels with NaN/inf targets out of the losses |
| `use-average-init-density` | `False` | Apply `average-init-density` (0.01) to all fields, as stock nerfacto does |
| `implementation` | `tcnn` | `torch` runs without tiny-cuda-nn |

All switches default to the original behaviour, so older checkpoints still load.

---

## Tests

Each script tests one layer on a real scene folder and is run directly (not with pytest):

```bash
python scripts/1.test_jai_raw.py --source0 <scene>/Source0_000001_2048x1536_16b.raw \
                                 --source1 <scene>/Source1_000001_2048x1536_16b.raw
python scripts/2.test_jai_dataparser.py <scene>
python scripts/3.test_jai_dataset.py    <scene>
python scripts/4.test_jai_losses.py     <scene>
python scripts/5.test_jai_model.py      <scene>
python scripts/6.test_jai_training.py   <scene> --num-iters 10
python scripts/7.test_jai_eval.py       <scene>
python scripts/8.test_jai_poses.py      <scene> --poses-source <scene>/colmap_out
```

---

## Troubleshooting

- **Blocky colours or a blurred reconstruction.** Check the poses first
  (`scripts/jai_pose_precheck.py`); pose quality is the most common cause.
- **COLMAP registers only part of the images.** For captures with more than one orbit, use
  the exhaustive script. Both scripts keep the largest sub-model automatically.
- **Dark reflectance images.** This is expected for an under-exposed but unclipped capture.
  Do not rescale the data; adjust `bayer-loss-mult` / `nir-loss-mult` if geometry suffers.
- **tiny-cuda-nn build fails.** Train with `--pipeline.model.implementation torch`.

---

## License and acknowledgements

Released under the Apache License 2.0 (see `LICENSE`).

This work builds on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
(Tancik et al., 2023) and [ThermalNeRF](https://arxiv.org/abs/2407.15337) (Lin et al., 2024).
The files listed under *Repository layout* were added or modified for the JAI dual-source
pipeline; the rest of the `nerfstudio/` package is their work.
