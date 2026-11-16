"""Seeded wrapper around jai_export_4ch_reflectance_pointcloud.py (the original exporter, unchanged).

The original exporter samples random training pixels with an unseeded RNG; this wrapper seeds
python / numpy / torch first so the sampled geometry is reproducible, then runs the exporter
with the remaining arguments.  Used to create a fresh mean-embedding baseline for scenes whose
old export cannot be traced to its training rays.

Usage: python scripts/jai_export_4ch_seeded.py --seed 0 -- <jai_export_4ch_reflectance_pointcloud args>
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

argv = sys.argv[1:]
seed = 0
if argv[:1] == ["--seed"]:
    seed, argv = int(argv[1]), argv[2:]
if argv[:1] == ["--"]:
    argv = argv[1:]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jai_export_4ch_reflectance_pointcloud as exporter  # noqa: E402

sys.argv = ["jai_export_4ch_reflectance_pointcloud.py"] + argv
raise SystemExit(exporter.main())
