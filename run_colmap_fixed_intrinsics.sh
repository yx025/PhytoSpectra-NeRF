#!/bin/bash

set -e

# ===========================================================================
# COLMAP with FROZEN intrinsics for MULTI-ORBIT captures (e.g. two camera
# rings at different heights around one tall plant).
#
# Why this script exists (vs run_colmap_fixed_intrinsics_me.sh)
# ------------------------------------------------------------
# The "_me" script uses sequential_matcher, which assumes ONE continuous
# ordered capture path. This dataset was shot in TWO passes — an upper ring
# and a lower ring — which the filenames make explicit:
#     ring A:  Source0_000000  ... Source0_000148   (149 imgs)
#     ring B:  Source0_9000000 ... Source0_9000143  (144 imgs)
# Sorted, ring A is immediately followed by ring B, so sequential matching
# only links the two rings at that single seam and never matches the many
# upper<->lower image pairs that actually see the same middle of the plant.
# Result: two disconnected sub-models and a broken reconstruction.
#
# Fix: EXHAUSTIVE matching (every image pair). At ~293 images this is cheap on
# a GPU and guarantees the cross-ring overlaps get found, tying both orbits
# into one model. For much larger sets (>~1500 imgs) switch MATCHER to
# "vocab_tree" below.
# ===========================================================================

# Pick a GPU with free VRAM (check nvidia-smi). COLMAP defaults to GPU 0;
# CUDA_VISIBLE_DEVICES remaps it onto the physical card you want.
export CUDA_VISIBLE_DEVICES=0

# Scene path can be passed as the first argument (for batch loops);
# falls back to the hardcoded default when called with no args.
SCENE="${1:-data/JAI/5.12/3.soybean2}"
OUT="$SCENE/colmap_out"
IMG_DIR="$OUT/images"
COL="$OUT/colmap"

# Matcher: "exhaustive" (robust for multi-orbit, use this) or "vocab_tree"
# (for very large image counts; requires VOCAB_TREE_PATH below).
MATCHER="exhaustive"
# Download once if you ever need vocab_tree:
#   https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin
VOCAB_TREE_PATH="$HOME/.cache/colmap/vocab_tree_flickr100K_words256K.bin"

# MATLAB camera calibration intrinsics
# Order: fx, fy, cx, cy, k1, k2, p1, p2
#INTRINSICS="1731.9709281628,1733.3525473429,998.6794211203,742.2791920584,-0.1943561588,0.0969853120,0.0000000000,0.0000000000" #4.28 cucumber & strawberry
#INTRINSICS="1733.1381892968,1734.0530054530,998.9335185579,765.8987510152,-0.1984375759,0.1122714198,0.0000000000,0.0000000000" #4.28 soybean
#INTRINSICS="1723.7357859833,1725.1243407672,990.4650438165,754.3119446845,-0.1962356021,0.0976476569,0.0000000000,0.0000000000" #4.30 pepper
INTRINSICS="1711.3281539841,1712.8763339556,1012.0656565491,771.2506580245,-0.1981885106,0.1044166656,0.0000000000,0.0000000000" #5.8 and 5.12

echo "Scene: $SCENE"
echo "Output: $OUT"
echo "Image dir: $IMG_DIR"
echo "Matcher: $MATCHER"
echo "Intrinsics: $INTRINSICS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Step 1: Copy images to nerfstudio layout (no COLMAP)"
ns-process-data images \
    --data "$SCENE/jpg" \
    --output-dir "$OUT" \
    --skip-colmap

echo "Step 2: Remove old COLMAP results"
rm -f "$COL/database.db"
rm -rf "$COL/sparse"
mkdir -p "$COL/sparse"

echo "Step 3: COLMAP feature extraction — denser SIFT, VRAM-safe (GPU)"
colmap feature_extractor \
    --database_path "$COL/database.db" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "$INTRINSICS" \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0 \
    --SiftExtraction.max_num_features 20000 \
    --SiftExtraction.peak_threshold 0.0033 \
    --SiftExtraction.edge_threshold 20 \
    --SiftExtraction.first_octave 0 \
    --SiftExtraction.max_image_size 2048

echo "Step 4: COLMAP matching — EXHAUSTIVE to bridge the two orbits"
# Every image pair is tried, so upper-ring and lower-ring views that overlap
# on the plant's mid-section get matched and the two orbits fuse into one
# model. guided_matching tightens matches using the (frozen) intrinsics.
if [ "$MATCHER" = "vocab_tree" ]; then
    if [ ! -f "$VOCAB_TREE_PATH" ]; then
        echo "ERROR: MATCHER=vocab_tree but vocab tree not found at:"
        echo "  $VOCAB_TREE_PATH"
        echo "Download it (see header) or set MATCHER=exhaustive."
        exit 1
    fi
    colmap vocab_tree_matcher \
        --database_path "$COL/database.db" \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index 0 \
        --SiftMatching.max_ratio 0.9 \
        --SiftMatching.max_distance 0.85 \
        --SiftMatching.max_num_matches 20000 \
        --SiftMatching.guided_matching 1 \
        --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE_PATH"
else
    colmap exhaustive_matcher \
        --database_path "$COL/database.db" \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index 0 \
        --SiftMatching.max_ratio 0.9 \
        --SiftMatching.max_distance 0.85 \
        --SiftMatching.max_num_matches 20000 \
        --SiftMatching.guided_matching 1 \
        --ExhaustiveMatching.block_size 50
fi

echo "Step 5: COLMAP mapper with frozen intrinsics + relaxed thresholds for plants"
# multiple_models 1 (default) lets it still emit sub-models if a few frames
# can't merge; Step 5.4 then promotes the largest. With exhaustive matching the
# two orbits should normally fuse into a single sparse/0.
colmap mapper \
    --database_path "$COL/database.db" \
    --image_path "$IMG_DIR" \
    --output_path "$COL/sparse" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.init_min_num_inliers 50 \
    --Mapper.abs_pose_min_num_inliers 15 \
    --Mapper.abs_pose_min_inlier_ratio 0.2 \
    --Mapper.min_num_matches 15 \
    --Mapper.tri_min_angle 1.0 \
    --Mapper.filter_min_tri_angle 1.0

echo "Step 5.4: Ensure sparse/0 is the LARGEST sub-model"
# COLMAP's mapper can emit several disconnected sub-models (sparse/0, sparse/1,
# ...) and does NOT guarantee sparse/0 is the biggest. ns-process-data only ever
# reads sparse/0, so if a tiny junk cluster lands there you silently get a
# near-empty transforms.json ("found poses for X% of images"). Pick the model
# with the most registered images and make it sparse/0 (keeping a backup).
python - "$COL/sparse" <<'PY'
import os, struct, sys, shutil
root = sys.argv[1]
def n_images(p):
    try:
        with open(os.path.join(p, "images.bin"), "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except Exception:
        return -1
models = {d: n_images(os.path.join(root, d))
          for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
models = {d: n for d, n in models.items() if n >= 0}
if not models:
    print("  no sub-models found"); sys.exit(0)
best = max(models, key=models.get)
print("  sub-models (images):", {d: models[d] for d in sorted(models)})
if best == "0":
    print(f"  sparse/0 already largest ({models['0']} images) — no swap")
else:
    print(f"  largest is sparse/{best} ({models[best]} images); swapping into sparse/0")
    if os.path.isdir(os.path.join(root, "0")):
        bak = os.path.join(root, "0_not_largest")
        shutil.rmtree(bak, ignore_errors=True)
        os.rename(os.path.join(root, "0"), bak)
    os.rename(os.path.join(root, best), os.path.join(root, "0"))
PY

echo "Step 5.5: Analyze COLMAP reconstruction"
# Watch '#registered images': if it is ~293 the two orbits fused. If it is
# ~149 or ~144 only one ring registered -> the rings don't visually overlap
# enough (re-shoot with more mid-height overlap), or try MATCHER=vocab_tree.
colmap model_analyzer \
    --path "$COL/sparse/0"

echo "Step 6: Convert COLMAP sparse model to transforms.json"
ns-process-data images \
    --data "$SCENE/jpg" \
    --output-dir "$OUT" \
    --skip-colmap \
    --skip-image-processing

echo "Step 7: Check transforms.json intrinsics"
python - <<PY
import json

path = "$OUT/transforms.json"

with open(path, "r") as f:
    data = json.load(f)

print("transforms.json:", path)

n_frames = len(data.get("frames", []))
print("registered frames in transforms.json:", n_frames)
for k in ["w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]:
    print(k, "=", data.get(k))
PY

echo "Done."
