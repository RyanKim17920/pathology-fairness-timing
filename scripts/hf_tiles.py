#!/usr/bin/env python3
"""Download one cohort directory from a Hugging Face dataset repository."""

import argparse


REPO = None
REVISION = None


def pull(cohort, dest, *, repo=None, revision=None):
    repo = repo or REPO
    revision = revision or REVISION
    if not repo:
        raise ValueError("set hf_tiles.REPO or pass --repo")
    if not revision:
        raise ValueError("set hf_tiles.REVISION or pass --revision")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{cohort}/**"],
        local_dir=dest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True,
                        help="immutable dataset commit SHA")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    global REPO, REVISION
    REPO = args.repo
    REVISION = args.revision
    pull(args.cohort, args.dest)


if __name__ == "__main__":
    main()
