#!/usr/bin/env python3
"""Download one cohort directory from a Hugging Face dataset repository."""

import argparse


REPO = None


def pull(cohort, dest):
    if not REPO:
        raise ValueError("set hf_tiles.REPO or pass --repo")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=[f"{cohort}/**"],
        local_dir=dest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    global REPO
    REPO = args.repo
    pull(args.cohort, args.dest)


if __name__ == "__main__":
    main()
