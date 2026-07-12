#!/usr/bin/env python
"""Push/pull nanopath fairness tile cohorts to a PRIVATE HF dataset repo.

Keeps the local cluster footprint minimal:  tile -> push -> (optional) --clean.
At run time, pull only the cohort a job needs, then delete it again.

Requires: `huggingface_hub` + a logged-in token (already present for user ryankim17920).

Examples
--------
  # one-time: push the small metadata (CSVs/folds/holdout lists)
  python hf_tiles.py push-metadata

  # push a tiled cohort into <repo>/<cohort>/ , then free the local copy
  python hf_tiles.py push --cohort cptac_lung --dir /data/ryan.kim/cptac_tiles/cptac_lung --clean

  # pull it back just-in-time for an eval run
  python hf_tiles.py pull --cohort cptac_lung --dest /data/ryan.kim/cptac_tiles/cptac_lung

  # see what's stored
  python hf_tiles.py list
"""
import argparse
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

REPO = "ryankim17920/nanopath-fairness-tiles"
REPO_TYPE = "dataset"
META_DIR = "/admin/home/ryan.kim/nt/data/metadata"


def _api():
    api = HfApi()
    api.create_repo(REPO, repo_type=REPO_TYPE, private=True, exist_ok=True)
    return api


def push(cohort: str, folder: str, clean: bool):
    api = _api()
    # upload_folder nests under <cohort>/ via path_in_repo and handles large
    # trees (parquet tiles go through LFS). Resumable across reruns.
    api.upload_folder(
        folder_path=folder, path_in_repo=cohort,
        repo_id=REPO, repo_type=REPO_TYPE,
        commit_message=f"add cohort {cohort}",
    )
    # verify presence
    files = [f for f in api.list_repo_files(REPO, repo_type=REPO_TYPE) if f.startswith(cohort + "/")]
    print(f"pushed {cohort}: {len(files)} files now in repo under {cohort}/")
    if clean:
        if files:
            shutil.rmtree(folder)
            print(f"cleaned local {folder}")
        else:
            print("REFUSING to clean: no files verified in repo", file=sys.stderr)
            sys.exit(2)


def push_metadata():
    api = _api()
    api.upload_folder(
        folder_path=META_DIR, path_in_repo="metadata",
        repo_id=REPO, repo_type=REPO_TYPE,
        ignore_patterns=["*.raw.csv"],
        commit_message="add demographics/labels/folds metadata",
    )
    print("metadata pushed")


def pull(cohort: str, dest: str):
    snapshot_download(
        repo_id=REPO, repo_type=REPO_TYPE,
        allow_patterns=[f"{cohort}/*"], local_dir=dest,
    )
    print(f"pulled {cohort} -> {dest}")


def ls():
    api = _api()
    files = list(api.list_repo_files(REPO, repo_type=REPO_TYPE))
    cohorts = sorted({f.split("/")[0] for f in files})
    print(f"repo {REPO} ({len(files)} files) cohorts: {cohorts}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("push"); sp.add_argument("--cohort", required=True); sp.add_argument("--dir", required=True); sp.add_argument("--clean", action="store_true")
    sub.add_parser("push-metadata")
    pl = sub.add_parser("pull"); pl.add_argument("--cohort", required=True); pl.add_argument("--dest", required=True)
    sub.add_parser("list")
    a = p.parse_args()
    if a.cmd == "push": push(a.cohort, a.dir, a.clean)
    elif a.cmd == "push-metadata": push_metadata()
    elif a.cmd == "pull": pull(a.cohort, a.dest)
    elif a.cmd == "list": ls()


if __name__ == "__main__":
    main()
