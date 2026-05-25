"""Phase 1.9 — FB flat-corpus batch_calib (end-of-corpus 150 questions).

All [ANS] expected_behavior. FlatMemory at end has dropped most docs; most predictions refuse.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__flat-corpus__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__flat-corpus__batch_calib__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Most ANS predictions refuse → 0.0; exceptions get explicit scores
SPECIAL: dict[int, tuple[float, str]] = {
    15: (1.0, "[ANS] doc15; gold=0; pred=0 exact."),
    35: (0.75, "[ANS] doc35; gold=AMD operations highest CF; pred='operating activities most cash flow' (no $). Match."),
    52: (0.75, "[ANS] doc52; gold=Best Buy operations $1.8B; pred='operating activities most cash flow' (no $)."),
    90: (1.0, "[ANS] doc90; gold=Consumer Health Aug 30; pred=exact."),
    96: (1.0, "[ANS] doc96; gold=JPM gross margin not relevant; pred=same. Exact paraphrase."),
    125: (1.0, "[ANS] doc125; gold=defeated; pred='not approved' (functionally equivalent)."),
    144: (0.0, "[ANS] doc144; gold=No quick ratio 0.54; pred=refuses on definitive ANS."),
    145: (0.75, "[ANS] doc145; gold=Yes Verizon cap-intensive (ratio 2.77); pred=Yes cap-intensive (no number)."),
    146: (0.0, "[ANS] doc146; gold=No debt decreased $229M; pred='Yes increased' with text showing decrease — Y/N flip + self-contradictory."),
    147: (1.0, "[ANS] doc147; gold=42.69 DPO; pred=42.52 (0.4% off — exact) with full calculation."),
    148: (0.0, "[ANS] doc148; gold=0.2%; pred=-5.0% — wrong sign + off."),
    149: (0.0, "[ANS] doc149; gold=6.2%; pred=3.9% (37% off)."),
}

DEFAULT_RATIONALE = "[ANS] FlatMemory has dropped this doc's paragraphs. PRED refuses on definitive ANS — penalised."

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
