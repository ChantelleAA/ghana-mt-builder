"""
push_dataset_to_hf.py  —  maintainer tool (incremental sync)
============================================================
Keeps the HuggingFace dataset that `ghana_corpus.py` reads from in sync with
your local `bible_parallel_text_datasets/` directory.

By default it APPENDS: it lists what is already on HuggingFace, compares against
your local CSVs, and uploads only the files that are not there yet.  This is
what you run after building a few more Ghanaian datasets or caching a new
reference language — only the new files go up.

Users never run this; they only read the data, and `ghana_corpus.py`
automatically picks up anything new on HuggingFace the next time it runs.

What gets uploaded
------------------
  • every Ghanaian language CSV   ({Name}_{code}_v{id}.csv)
  • english_cache.csv
  • reference_caches/*.csv
Empty / header-only files are skipped.

Auth
----
Uses your cached HuggingFace login, or set HF_TOKEN in the environment:
    export HF_TOKEN=hf_your_token

Usage
-----
    python scripts/push_dataset_to_hf.py            # append new files only
    python scripts/push_dataset_to_hf.py --dry-run  # show what would upload
    python scripts/push_dataset_to_hf.py --sync     # also re-upload changed files
"""

import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "bible_parallel_text_datasets")

HF_REPO_ID   = os.environ.get("GHANA_CORPUS_REPO", "ghananlpcommunity/ghana-corpus")
HF_REPO_TYPE = "dataset"

# Files in DATA_ROOT that should never be uploaded (run-state, not data).
SKIP = {"progress.json", "progress.json.tmp", "testament_status.json"}
MIN_BYTES = 64   # skip empty / header-only CSVs


def collect_local() -> list[str]:
    """Repo-relative paths of every uploadable local data CSV."""
    rels = []
    for root, _dirs, files in os.walk(DATA_ROOT):
        for name in files:
            if name in SKIP or not name.endswith(".csv"):
                continue
            local = os.path.join(root, name)
            if os.path.getsize(local) < MIN_BYTES:
                continue
            rel = os.path.relpath(local, DATA_ROOT).replace(os.sep, "/")
            rels.append(rel)
    return sorted(rels)


def remote_files(api: HfApi) -> set[str]:
    try:
        return set(api.list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE))
    except RepositoryNotFoundError:
        return set()


def main():
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    sync    = "--sync" in argv

    token = os.environ.get("HF_TOKEN")  # falls back to cached login if None
    api = HfApi(token=token)

    local = collect_local()
    if not local:
        sys.exit(f"No data files found under {DATA_ROOT}")

    remote = remote_files(api)
    new = [r for r in local if r not in remote]
    existing = [r for r in local if r in remote]

    print(f"Local data files : {len(local)}")
    print(f"Already on HF     : {len(existing)}")
    print(f"New (not on HF)   : {len(new)}")
    for r in new:
        print(f"   + {r}")

    # What to upload: new files always; everything if --sync (HF skips unchanged).
    to_upload = local if sync else new
    if not to_upload:
        print("\nHuggingFace is already up to date.")
        return

    if dry_run:
        print(f"\n[dry-run] would upload {len(to_upload)} file(s); nothing sent.")
        return

    print(f"\nEnsuring dataset repo {HF_REPO_ID} ...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                    exist_ok=True, private=False)

    print(f"Uploading {len(to_upload)} file(s) ...")
    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        folder_path=DATA_ROOT,
        allow_patterns=to_upload,
        commit_message=("Sync Ghana corpus data" if sync
                        else f"Add {len(to_upload)} new corpus file(s)"),
    )
    print(f"\nDone: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
