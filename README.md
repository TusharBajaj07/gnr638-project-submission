# GNR 638 Project 1 — Geospatial Map Stitching + MCQ

**Team:** 23b2107, 23b2108

## Pipeline

1. **Stitch** the patches in `<test_dir>/patches/` into a single map using
   Jigsaw V7-BB (best-buddy graph + Hungarian fill).
2. **Answer** each MCQ in `<test_dir>/test.csv` using
   Qwen2.5-VL-7B-Instruct-AWQ with `max_pixels = 2048 * 28 * 28`.
   Constrained-decoding head over digits {1..5}; abstain (5) when top-prob < 0.40.

Validated on a 144-question dev benchmark across 14 cities: **score +59.50**
(46.5% accuracy with no hallucinations).

## Run

```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir /abs/path/to/test_dir
```

`submission.csv` is written to the current working directory.

## Files
- `inference.py` — main entry
- `setup.bash` — environment + weights setup (needs internet)
- `src/stitching/jigsaw_v7_bb.py` and deps — stitching pipeline
