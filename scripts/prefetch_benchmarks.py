"""
Prefetch + verify the six Stage 3 benchmarks locally.

This script downloads each dataset to a project-local cache so the rest of
the pipeline never has to hit the network at runtime. It also reports
size + sample-item counts + a sample record per benchmark so we know what
shape each one comes in BEFORE writing the adapters.

Local cache layout:
    data/benchmarks/hf_cache/        — HuggingFace `datasets` cache
    data/benchmarks/longmemeval/     — LongMemEval JSON files (from GitHub)
    data/benchmarks/manifest.json    — what was fetched, when, sizes

Datasets:
    1. NarrativeQA   — deepmind/narrativeqa (HF)
    2. QASPER        — allenai/qasper (HF)
    3. CUAD          — theatticusproject/cuad-qa (HF)
    4. HotpotQA      — hotpotqa/hotpot_qa (HF, "distractor" config)
    5. FinanceBench  — PatronusAI/financebench (HF)
    6. LongMemEval   — xiaowu0162/LongMemEval (GitHub release)

Usage:
    python scripts/prefetch_benchmarks.py
    python scripts/prefetch_benchmarks.py --only hotpotqa qasper
    python scripts/prefetch_benchmarks.py --no-narrativeqa  # skip the big one
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"
HF_CACHE = DATA_DIR / "hf_cache"
LME_DIR = DATA_DIR / "longmemeval"

# Point HF at our local cache before importing datasets.
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)
os.environ["HF_HOME"] = str(HF_CACHE)


def _human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n_bytes)
    for u in units:
        if f < 1024:
            return f"{f:.1f} {u}"
        f /= 1024
    return f"{f:.1f} PB"


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _print_step(name: str) -> None:
    print()
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Per-benchmark fetchers
# ---------------------------------------------------------------------------


def fetch_hotpotqa() -> dict[str, Any]:
    """HotpotQA: Wikipedia multi-hop QA, 113K total, "distractor" config."""
    _print_step("HotpotQA (hotpotqa/hotpot_qa, config=distractor)")
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
    )
    elapsed = time.time() - t0
    splits = {name: len(split) for name, split in ds.items()}
    sample = ds["validation"][0]
    print(f"  splits = {splits}")
    print(f"  sample keys = {sorted(sample.keys())}")
    print(f"  sample question = {sample.get('question', '?')[:120]}")
    print(f"  sample answer   = {sample.get('answer', '?')[:120]}")
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "hotpotqa",
        "hf_id": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "splits": splits,
        "sample_keys": sorted(sample.keys()),
        "elapsed_s": elapsed,
    }


def fetch_qasper() -> dict[str, Any]:
    """QASPER: 5,049 QA over 1,585 NLP papers.

    The HF `allenai/qasper` is a loading-script dataset which `datasets>=3`
    no longer supports. We download the canonical tarballs directly from
    AI2's public S3 bucket and extract them — same data the HF script uses
    internally, just without the unsupported wrapper.
    """
    _print_step("QASPER (direct from AI2 S3)")
    QASPER_DIR = DATA_DIR / "qasper"
    QASPER_DIR.mkdir(parents=True, exist_ok=True)
    # URLs taken directly from allenai/qasper's qasper.py loading script.
    tarballs = [
        ("qasper-train-dev-v0.3.tgz",
         "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"),
        ("qasper-test-and-evaluator-v0.3.tgz",
         "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"),
    ]
    t0 = time.time()
    for fname, url in tarballs:
        tar_path = QASPER_DIR / fname
        if tar_path.exists() and tar_path.stat().st_size > 0:
            print(f"  archive already present: {fname} ({_human_size(tar_path.stat().st_size)})")
        else:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, tar_path)
            print(f"    saved {_human_size(tar_path.stat().st_size)}")
        # Extract.
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(QASPER_DIR)
        print(f"    extracted {fname}")

    # Find the three JSON files (might be nested in a subdir).
    json_files = list(QASPER_DIR.rglob("qasper-*-v0.3.json"))
    print(f"  json files found: {[p.name for p in json_files]}")

    splits: dict[str, int] = {}
    sample_info: dict[str, Any] = {}
    for jp in json_files:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # QASPER format: {paper_id: {title, abstract, full_text, qas: [...]}}
        n_papers = len(data)
        n_questions = sum(len(p.get("qas", [])) for p in data.values())
        split_key = "test" if "test" in jp.name else ("dev" if "dev" in jp.name else "train")
        splits[split_key] = n_papers
        if split_key == "dev" and not sample_info:
            first_pid = next(iter(data))
            first_paper = data[first_pid]
            sample_info = {
                "paper_id": first_pid,
                "title": first_paper.get("title", "?")[:80],
                "n_questions_first_paper": len(first_paper.get("qas", [])),
                "n_sections_first_paper": len(first_paper.get("full_text", [])),
                "n_questions_split_total": n_questions,
            }
            print(f"  dev sample paper: {sample_info['title']!r}")
            print(f"  dev sample first paper has {sample_info['n_questions_first_paper']} questions, "
                  f"{sample_info['n_sections_first_paper']} sections")
    elapsed = time.time() - t0
    print(f"  splits (n_papers) = {splits}")
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "qasper",
        "source": "AI2 S3 direct (qasper-dataset.s3.us-west-2.amazonaws.com)",
        "version": "v0.3",
        "splits_n_papers": splits,
        "sample": sample_info,
        "elapsed_s": elapsed,
    }


def fetch_cuad() -> dict[str, Any]:
    """CUAD: 510 commercial contracts, SQuAD-shape QA over 41 clause categories.

    The HF `theatticusproject/cuad-qa` is a loading-script dataset
    which `datasets>=3` no longer supports. We download from Zenodo
    (canonical, citeable archive) instead — the JSON is SQuAD v2 format.
    """
    _print_step("CUAD (direct from Zenodo)")
    CUAD_DIR = DATA_DIR / "cuad"
    CUAD_DIR.mkdir(parents=True, exist_ok=True)
    # Zenodo permanent archive for CUAD v1.
    zip_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
    zip_path = CUAD_DIR / "CUAD_v1.zip"
    t0 = time.time()
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"  archive already present: {zip_path} ({_human_size(zip_path.stat().st_size)})")
    else:
        print(f"  downloading {zip_url}")
        urllib.request.urlretrieve(zip_url, zip_path)
        print(f"    saved {_human_size(zip_path.stat().st_size)}")

    # Extract.
    extract_dir = CUAD_DIR / "CUAD_v1"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CUAD_DIR)
        print(f"  extracted into {CUAD_DIR}")
    else:
        print(f"  already extracted: {extract_dir}")

    # Parse the main JSON (SQuAD v2 format).
    main_json_candidates = list(CUAD_DIR.rglob("CUAD_v1.json"))
    if not main_json_candidates:
        # Try alternate name
        main_json_candidates = list(CUAD_DIR.rglob("*.json"))
    sample_info: dict[str, Any] = {}
    n_contracts = 0
    n_questions = 0
    if main_json_candidates:
        main_json = main_json_candidates[0]
        print(f"  parsing {main_json.name} ({_human_size(main_json.stat().st_size)})")
        with main_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # SQuAD format: {"version": ..., "data": [{"title", "paragraphs": [{"context", "qas": [{"question", "answers"}]}]}]}
        contracts = data.get("data", [])
        n_contracts = len(contracts)
        for c in contracts:
            for p in c.get("paragraphs", []):
                n_questions += len(p.get("qas", []))
        if contracts:
            first = contracts[0]
            first_para = first.get("paragraphs", [{}])[0]
            first_q = first_para.get("qas", [{}])[0]
            sample_info = {
                "title": first.get("title", "?")[:80],
                "context_chars": len(first_para.get("context", "")),
                "sample_question": first_q.get("question", "?")[:120],
            }
            print(f"  sample contract title: {sample_info['title']!r}")
            print(f"  sample context length: {sample_info['context_chars']} chars")
            print(f"  sample question: {sample_info['sample_question']!r}")
    elapsed = time.time() - t0
    print(f"  n_contracts = {n_contracts}, n_questions = {n_questions}")
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "cuad",
        "source": "zenodo.org direct ZIP",
        "version": "v1",
        "n_contracts": n_contracts,
        "n_questions": n_questions,
        "sample": sample_info,
        "elapsed_s": elapsed,
    }


def fetch_narrativeqa() -> dict[str, Any]:
    """NarrativeQA: 1,567 books + scripts with 46K QA pairs. LARGE — multi-GB."""
    _print_step("NarrativeQA (deepmind/narrativeqa) — multi-GB download")
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("deepmind/narrativeqa", cache_dir=str(HF_CACHE), trust_remote_code=True)
    elapsed = time.time() - t0
    splits = {name: len(split) for name, split in ds.items()}
    sample = ds["validation"][0]
    doc_text = sample.get("document", {}).get("text", "")
    print(f"  splits = {splits}")
    print(f"  sample question = {sample.get('question', {}).get('text', '?')[:120]}")
    print(f"  sample document length (chars) = {len(doc_text):,}")
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "narrativeqa",
        "hf_id": "deepmind/narrativeqa",
        "splits": splits,
        "sample_doc_chars": len(doc_text),
        "elapsed_s": elapsed,
    }


def fetch_financebench() -> dict[str, Any]:
    """FinanceBench: 10K QA over 361 US public filings (10-K / 10-Q / 8-K / earnings)."""
    _print_step("FinanceBench (PatronusAI/financebench)")
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("PatronusAI/financebench", cache_dir=str(HF_CACHE), trust_remote_code=True)
    elapsed = time.time() - t0
    splits = {name: len(split) for name, split in ds.items()}
    first_split = next(iter(ds.values()))
    sample = first_split[0]
    print(f"  splits = {splits}")
    print(f"  sample keys = {sorted(sample.keys())}")
    print(f"  sample question = {sample.get('question', '?')[:120]}")
    print(f"  sample answer   = {sample.get('answer', '?')[:120]}")
    if "evidence" in sample:
        ev = sample["evidence"]
        if isinstance(ev, list):
            ev_text = " ".join([str(e) for e in ev])
            print(f"  evidence (list, joined) length = {len(ev_text)} chars")
        else:
            print(f"  evidence (str) length = {len(str(ev))} chars")
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "financebench",
        "hf_id": "PatronusAI/financebench",
        "splits": splits,
        "sample_keys": sorted(sample.keys()),
        "elapsed_s": elapsed,
    }


def fetch_longmemeval() -> dict[str, Any]:
    """LongMemEval (Wu et al. 2024, ICLR 2025).

    The original `xiaowu0162/longmemeval` HF dataset is deprecated.
    The cleaned version `xiaowu0162/longmemeval-cleaned` exposes three
    JSON files, but `datasets.load_dataset` fails to auto-parse them
    due to a type mismatch in the `answer` column. We bypass that by
    using `huggingface_hub.hf_hub_download` to fetch the raw JSON
    files directly and parsing them ourselves.

    Files (sizes from the HF repo, totaling ~3 GB):
      - longmemeval_oracle.json   (~few MB, oracle gold conversations)
      - longmemeval_s_cleaned.json (~1.5 GB)
      - longmemeval_m_cleaned.json (~1.5 GB)
    """
    _print_step("LongMemEval (xiaowu0162/longmemeval-cleaned, raw JSON via hf_hub_download)")
    LME_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from huggingface_hub import hf_hub_download

    json_files = [
        "longmemeval_oracle.json",
        "longmemeval_s_cleaned.json",
        "longmemeval_m_cleaned.json",
    ]
    splits_info: dict[str, Any] = {}
    sample_info: dict[str, Any] = {}
    fetched_any = False
    for fname in json_files:
        target = LME_DIR / fname
        if target.exists() and target.stat().st_size > 1000:
            print(f"  already cached: {fname} ({_human_size(target.stat().st_size)})")
        else:
            print(f"  downloading {fname} via hf_hub_download ...")
            try:
                path = hf_hub_download(
                    repo_id="xiaowu0162/longmemeval-cleaned",
                    filename=fname,
                    repo_type="dataset",
                    cache_dir=str(HF_CACHE),
                )
                shutil.copy(path, target)
                print(f"    saved {_human_size(target.stat().st_size)}")
            except Exception as e:
                print(f"    [WARN] {fname} fetch failed: {e!r}")
                splits_info[fname] = {"error": repr(e)}
                continue
        # Parse the JSON to get item count and sample structure.
        try:
            with target.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                n = len(data)
                splits_info[fname] = {"n_items": n, "size_bytes": target.stat().st_size}
                print(f"    {fname}: {n} items, {_human_size(target.stat().st_size)}")
                if data and not sample_info:
                    sample = data[0]
                    sample_info["file"] = fname
                    if isinstance(sample, dict):
                        sample_info["sample_keys"] = sorted(sample.keys())
                        if "question" in sample:
                            sample_info["sample_question"] = str(sample["question"])[:160]
                        if "haystack_sessions" in sample:
                            hs = sample["haystack_sessions"]
                            if isinstance(hs, list):
                                sample_info["sample_n_sessions"] = len(hs)
                                if hs and isinstance(hs[0], list):
                                    sample_info["sample_session_msgs"] = len(hs[0])
                        if "answer" in sample:
                            sample_info["sample_answer"] = str(sample["answer"])[:160]
                        print(f"    sample keys = {sample_info['sample_keys']}")
                        if "sample_question" in sample_info:
                            print(f"    sample question = {sample_info['sample_question']!r}")
                        if "sample_n_sessions" in sample_info:
                            print(f"    sample has {sample_info['sample_n_sessions']} sessions")
            elif isinstance(data, dict):
                splits_info[fname] = {"type": "dict", "top_keys": sorted(data.keys())[:20]}
                print(f"    {fname}: top-level dict, keys={splits_info[fname]['top_keys'][:5]}...")
            fetched_any = True
        except Exception as e:
            print(f"    [WARN] failed to parse {fname}: {e!r}")
            splits_info[fname] = {"error_parse": repr(e), "size_bytes": target.stat().st_size}

    if not fetched_any:
        raise RuntimeError("LongMemEval: all 3 JSON files failed to fetch or parse")

    elapsed = time.time() - t0
    print(f"  elapsed = {elapsed:.1f}s")
    return {
        "name": "longmemeval",
        "hf_id": "xiaowu0162/longmemeval-cleaned",
        "source": "huggingface_hub raw JSON (bypasses load_dataset)",
        "files": splits_info,
        "sample": sample_info,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FETCHERS = {
    "hotpotqa": fetch_hotpotqa,
    "qasper": fetch_qasper,
    "cuad": fetch_cuad,
    "narrativeqa": fetch_narrativeqa,
    "financebench": fetch_financebench,
    "longmemeval": fetch_longmemeval,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=f"Subset of benchmarks to fetch. Choices: {list(FETCHERS)}",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Benchmarks to skip.",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Cache directory: {DATA_DIR}")
    print(f"HF_DATASETS_CACHE = {HF_CACHE}")

    plan = list(args.only) if args.only else list(FETCHERS)
    plan = [n for n in plan if n not in args.skip and n in FETCHERS]
    print(f"Will fetch: {plan}")

    results: dict[str, Any] = {}
    failed: dict[str, str] = {}

    t_total = time.time()
    for name in plan:
        try:
            results[name] = FETCHERS[name]()
        except Exception as e:
            traceback.print_exc()
            print(f"\n  [FAIL] {name}: {e!r}")
            failed[name] = repr(e)
    total_elapsed = time.time() - t_total

    # Summary
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Cache size on disk: {_human_size(_dir_size(DATA_DIR))}")
    print()
    for name in FETCHERS:
        if name in results:
            print(f"  [OK]   {name}")
        elif name in failed:
            print(f"  [FAIL] {name}  -- {failed[name][:80]}")
        else:
            print(f"  [SKIP] {name}")

    # Save manifest — merge with prior runs so --only never loses past results.
    manifest_path = DATA_DIR / "manifest.json"
    merged_results: dict[str, Any] = {}
    merged_failed: dict[str, str] = {}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
            merged_results.update(prior.get("results", {}))
            merged_failed.update(prior.get("failed", {}))
        except Exception:
            pass
    # New OK results override prior failures for the same benchmark.
    for n, r in results.items():
        merged_results[n] = r
        merged_failed.pop(n, None)
    for n, e in failed.items():
        if n not in results:  # only record failure if not also a success this run
            merged_failed[n] = e
    manifest = {
        "last_fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_size_bytes": _dir_size(DATA_DIR),
        "cache_size_human": _human_size(_dir_size(DATA_DIR)),
        "last_run_elapsed_s": total_elapsed,
        "last_run_plan": plan,
        "results": merged_results,
        "failed": merged_failed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest saved to {manifest_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
