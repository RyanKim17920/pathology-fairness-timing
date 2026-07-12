#!/usr/bin/env python
"""Tile on-disk CPTAC SVS slides into per-slide parquet tiles.

Reuses the extractor primitives in nanopath-jepa/prepare.py
(`_openslide_grid_rows`, `_openslide_mpp`) so the tiling contract (level 0 read,
rescaled to 0.5 MPP / 20x, 512px tiles, JPEG q95, tissue threshold 230,
non-overlapping grid, optional linspace cap) is identical to the rest of the
pipeline.

Slides are read FLAT from --svs-dir as <slide_id>.svs. Labels come from a
Patho-Bench split TSV (columns: case_id, slide_id, subtype). Each resolved slide
is tiled in a worker process and written to
    <out-dir>/slides_full/<slide_id>.parquet
(atomic .part rename, snappy). A <out-dir>/labels.tsv is written with
(case_id, slide_id, subtype). Existing per-slide parquet files are skipped so the
run is resumable.

--dry-run resolves slides + labels, prints counts and a tile estimate, and NEVER
opens or tiles a slide.
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

# --- Reuse the existing extractor primitives from prepare.py ---------------
NANOPATH_JEPA_DIR = "/admin/home/ryan.kim/nanopath/base-models/nanopath-jepa"
if NANOPATH_JEPA_DIR not in sys.path:
    sys.path.insert(0, NANOPATH_JEPA_DIR)

from prepare import (  # noqa: E402
    _openslide_grid_rows,
    _openslide_mpp,  # noqa: F401  (re-exported / used for measurement tooling)
    PARQUET_ROW_GROUP_SIZE,
)

# 0 = LUAD, 1 = LSCC  (Patho-Bench cptac_lung/subtype convention)
SUBTYPE_NAMES = {0: "LUAD", 1: "LSCC"}

DEFAULT_SPLIT_TSV = "/data/Patho-Bench/splits/cptac_lung/subtype/k=all.tsv"
DEFAULT_SVS_DIR = "/data/Patho-Bench/datasets/CPTAC/all"
DEFAULT_OUT_DIR = "/data/ryan.kim/cptac_tiles/cptac_lung"


def _default_workers():
    env = os.environ.get("PREPARE_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return min(8, os.cpu_count() or 1)


def resolve_cohort(split_tsv, svs_dir):
    """Read the split TSV, map slide_id -> (case_id, label), and check disk.

    Returns (records, missing) where records is a list of dicts for slides that
    exist on disk and missing is the list of slide_ids that do not.
    """
    import pandas as pd

    df = pd.read_csv(split_tsv, sep="\t")
    for col in ("case_id", "slide_id", "subtype"):
        if col not in df.columns:
            raise ValueError(f"split TSV {split_tsv} missing required column '{col}'")
    df = df[["case_id", "slide_id", "subtype"]].drop_duplicates(subset="slide_id")

    svs_dir = Path(svs_dir)
    records, missing = [], []
    for row in df.itertuples(index=False):
        slide_id = str(row.slide_id)
        case_id = str(row.case_id)
        label = int(row.subtype)
        svs_path = svs_dir / f"{slide_id}.svs"
        rec = {
            "case_id": case_id,
            "slide_id": slide_id,
            "subtype": label,
            "svs_path": str(svs_path),
        }
        if svs_path.exists():
            records.append(rec)
        else:
            missing.append(rec)
    return records, missing


def _class_counts(records):
    slide_counts = {0: 0, 1: 0}
    case_labels = {}
    for r in records:
        slide_counts[r["subtype"]] = slide_counts.get(r["subtype"], 0) + 1
        case_labels[r["case_id"]] = r["subtype"]
    case_counts = {0: 0, 1: 0}
    for lab in case_labels.values():
        case_counts[lab] = case_counts.get(lab, 0) + 1
    return slide_counts, case_counts, len(case_labels)


def _extract_one(args):
    """Worker: tile a single slide to <out_dir>/slides_full/<slide_id>.parquet.

    Mirrors prepare._cptac_pda_os_extract_one: open with openslide, call
    _openslide_grid_rows(image_col="image"), attach case_id + tile_idx, write
    snappy parquet via atomic .part rename. Skips if the parquet already exists.
    """
    import openslide
    import pyarrow as pa
    import pyarrow.parquet as pq

    case_id, slide_id, svs_path, cache_dir, cap = args
    cache_path = Path(cache_dir) / f"{slide_id}.parquet"
    if cache_path.exists():
        return slide_id, pq.read_metadata(cache_path).num_rows, "skipped"

    slide = openslide.OpenSlide(str(svs_path))
    try:
        rows = _openslide_grid_rows(slide, slide_id, image_col="image", cap=cap)
    finally:
        slide.close()
    for i, row in enumerate(rows):
        row["case_id"], row["tile_idx"] = case_id, i

    tmp = cache_path.with_suffix(".parquet.part")
    pq.write_table(
        pa.table({k: [r[k] for r in rows] for k in ("case_id", "slide_id", "tile_idx", "image")}),
        tmp,
        compression="snappy",
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )
    os.replace(tmp, cache_path)
    return slide_id, len(rows), "tiled"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["lung", "luad_lscc", "gbm", "ccrcc"], default="lung",
                    help="Which CPTAC cohort. 'lung'/'luad_lscc' use the cptac_lung/subtype split.")
    ap.add_argument("--split-tsv", default=DEFAULT_SPLIT_TSV)
    ap.add_argument("--svs-dir", default=DEFAULT_SVS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--cap", type=int, default=768, help="Max tiles per slide (linspace subsample). 0 = uncapped.")
    ap.add_argument("--workers", type=int, default=_default_workers())
    ap.add_argument("--limit", type=int, default=None, help="Only tile the first N resolved slides (smoke test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve slides + labels, print counts, and DO NOT open or tile any slide.")
    args = ap.parse_args()

    records, missing = resolve_cohort(args.split_tsv, args.svs_dir)
    n_split = len(records) + len(missing)
    slide_counts, case_counts, n_cases = _class_counts(records)

    print(f"[cohort]      {args.cohort}")
    print(f"[split-tsv]   {args.split_tsv}")
    print(f"[svs-dir]     {args.svs_dir}")
    print(f"[out-dir]     {args.out_dir}")
    print(f"[cap]         {args.cap}   [workers] {args.workers}")
    print(f"[split]       {n_split} unique slides in split")
    print(f"[found]       {len(records)} slides found on disk")
    print(f"[missing]     {len(missing)} slides missing from disk")
    print(f"[class/slide] LUAD(0)={slide_counts.get(0,0)}  LSCC(1)={slide_counts.get(1,0)}")
    print(f"[class/case]  cases={n_cases}  LUAD(0)={case_counts.get(0,0)}  LSCC(1)={case_counts.get(1,0)}")
    if missing:
        preview = ", ".join(r["slide_id"] for r in missing[:10])
        print(f"[missing ids] {preview}{' ...' if len(missing) > 10 else ''}")

    cap_txt = f"cap={args.cap}" if args.cap else "uncapped"
    est_per_slide = args.cap if args.cap else 900  # ~measured uncapped tissue tiles/slide
    print(f"[estimate]    ~{est_per_slide} tiles/slide ({cap_txt}); "
          f"~{est_per_slide * len(records):,} tiles across {len(records)} slides")

    if args.dry_run:
        print("[dry-run]     resolve+count only; no slides opened, nothing tiled.")
        return

    out_dir = Path(args.out_dir)
    slide_cache = out_dir / "slides_full"
    slide_cache.mkdir(parents=True, exist_ok=True)

    # labels.tsv over all resolved slides (written up front; harmless to overwrite).
    labels_path = out_dir / "labels.tsv"
    labels_path.write_text(
        "case_id\tslide_id\tsubtype\n"
        + "\n".join(f"{r['case_id']}\t{r['slide_id']}\t{r['subtype']}" for r in records)
        + "\n"
    )

    todo = records if args.limit is None else records[: args.limit]
    jobs = [(r["case_id"], r["slide_id"], r["svs_path"], str(slide_cache), args.cap) for r in todo]
    print(f"[tiling]      {len(jobs)} slides -> {slide_cache} ({args.workers} workers)")

    with mp.Pool(args.workers) as pool:
        for slide_id, n_tiles, status in pool.imap_unordered(_extract_one, jobs):
            print(f"  [{status:7}] {slide_id}: {n_tiles} tiles", flush=True)

    print(f"[done]        wrote per-slide parquet to {slide_cache}; labels at {labels_path}")


if __name__ == "__main__":
    main()
