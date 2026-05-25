"""Phase 1.9 — FB semantic-corpus batch_calib (end-of-corpus 150 questions).

All [ANS] expected. Semantic (TF-IDF) graph at end retains less than V5
graph but more than FlatMemory under heavy dilution.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__batch_calib__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Most ANS predictions refuse → 0.0; exceptions get explicit scores
SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.75, "[ANS] doc0; gold=$1577; pred=$1,501 (4.8% off — within tolerance, close)."),
    1: (1.0, "[ANS] doc1; gold=$8.70B; pred=8.738B exact."),
    2: (0.0, "[ANS] doc2; Y/N flip — pred='Yes capital-intensive'."),
    3: (1.0, "[ANS] doc3; matches 3M op margin drivers (litigation, PFAS, Russia, divestiture)."),
    4: (0.5, "[ANS] doc4; pred='Consumer segment' (no %) partial."),
    15: (1.0, "[ANS] doc15; gold=0; pred=0 exact."),
    43: (0.0, "[ANS] doc43; gold='Customer deposits'; pred='total liabilities' wrong."),
    74: (0.25, "[ANS] doc74; gold=$59,268; pred=$52,694 (11% off)."),
    90: (1.0, "[ANS] doc90; Consumer Health Aug 30 exact match."),
    96: (1.0, "[ANS] doc96; JPM gross margin not relevant — exact paraphrase."),
    119: (0.25, "[ANS] doc119; gold=$4.6B; pred=$4.2B (8.7% off)."),
    120: (0.5, "[ANS] doc120; partial geographies (US/Canada/LatAm/Europe — missing Africa/ME/Asia/Australia)."),
    121: (0.0, "[ANS] doc121; pred='Yes Combat Arms litigation' — Y/N flip + fabrication (Combat Arms is 3M)."),
    122: (0.0, "[ANS] doc122; gold=$411M; pred='0' confident-wrong."),
    124: (0.25, "[ANS] doc124; gold=16.5%; pred=20.1% (22% off — outside tolerance)."),
    125: (1.0, "[ANS] doc125; 'not approved' matches 'defeated' paraphrase."),
    126: (0.25, "[ANS] doc126; gold=$400M; pred='$1.5B' wrong specific."),
    127: (0.25, "[ANS] doc127; gold=$8.4B; pred='$5.0B' wrong specific."),
    129: (0.25, "[ANS] doc129; gold=1pp; pred='2 percentage points' — wrong magnitude."),
}

DEFAULT_RATIONALE = "[ANS] semantic graph does not retain this doc's content at end-of-corpus. PRED refuses — penalised."

ENTRY_SUFFIXES: list[str] = [f"doc{i}_qa0" for i in range(150)]


def main() -> None:
    assert len(ENTRY_SUFFIXES) == 150
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    added = skipped = 0
    total = 0.0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for i, suffix in enumerate(ENTRY_SUFFIXES):
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing:
                skipped += 1
                continue
            score, rationale = SPECIAL.get(i, (0.0, DEFAULT_RATIONALE))
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL},
                               ensure_ascii=False) + "\n")
            added += 1
            total += score
            existing.add(qid)
    print(f"batch_calib added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
