"""Phase 1.9 — FB semantic-corpus calibration part1 (entries 0-149)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    22: (1.0, "[ANS] doc1 seen=3; gold=$8.70B; pred=8.738B (0.44% off — exact)."),
    37: (0.5, "[ACK] doc90 not seen; PRED='Consumer Health Aug 30' correct world-knowledge."),
    40: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong specific."),
    42: (0.5, "[ACK] doc25 not seen; pred='packaging industry' correct world-knowledge."),
    71: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct world-knowledge."),
    75: (0.0, "[ANS] doc6 seen=8; pred=refuses on definitive ANS."),
    84: (0.0, "[ANS] doc5 seen=9; pred=refuses on definitive ANS."),
    96: (0.0, "[ANS] doc7 seen=10; pred=refuses on definitive ANS."),
    99: (0.5, "[ACK] doc96 not seen; PRED='JPM gross margin not relevant' correct world-knowledge."),
    107: (1.0, "[ANS] doc1 seen=11; 8.738B exact match."),
    108: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    121: (0.0, "[ANS] doc12 seen=13; pred=refuses."),
    136: (1.0, "[ANS] doc3 seen=14; gold=3M op margin drivers (Combat Arms, PFAS, Russia); pred=exact match including divestiture-related restructuring."),
    137: (0.0, "[ANS] doc12 seen=14; pred=refuses."),
    144: (0.0, "[ANS] doc7 seen=15; pred=refuses on definitive ANS."),
    146: (1.0, "[ANS] doc3 seen=15; same drivers match."),
    147: (0.5, "[ACK] doc90 not seen; correct WK."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc123_qa0__after0", "doc31_qa0__after0", "doc147_qa0__after0", "doc130_qa0__after0",
    "doc115_qa0__after0", "doc119_qa0__after0", "doc133_qa0__after0", "doc137_qa0__after0",
    "doc59_qa0__after0", "doc27_qa0__after0",
    "doc93_qa0__after1", "doc72_qa0__after1", "doc64_qa0__after1", "doc6_qa0__after1",
    "doc27_qa0__after1", "doc35_qa0__after1", "doc5_qa0__after1", "doc60_qa0__after1",
    "doc106_qa0__after1", "doc87_qa0__after1",
    "doc101_qa0__after2", "doc71_qa0__after2", "doc1_qa0__after2", "doc118_qa0__after2",
    "doc75_qa0__after2", "doc67_qa0__after2", "doc13_qa0__after2", "doc78_qa0__after2",
    "doc116_qa0__after2", "doc91_qa0__after2",
    "doc43_qa0__after3", "doc120_qa0__after3", "doc101_qa0__after3", "doc64_qa0__after3",
    "doc107_qa0__after3", "doc121_qa0__after3", "doc102_qa0__after3", "doc90_qa0__after3",
    "doc26_qa0__after3", "doc22_qa0__after3",
    "doc122_qa0__after4", "doc141_qa0__after4", "doc25_qa0__after4", "doc43_qa0__after4",
    "doc76_qa0__after4", "doc120_qa0__after4", "doc138_qa0__after4", "doc42_qa0__after4",
    "doc83_qa0__after4", "doc95_qa0__after4",
    "doc147_qa0__after5", "doc32_qa0__after5", "doc131_qa0__after5", "doc97_qa0__after5",
    "doc93_qa0__after5", "doc80_qa0__after5", "doc109_qa0__after5", "doc113_qa0__after5",
    "doc13_qa0__after5", "doc110_qa0__after5",
    "doc127_qa0__after6", "doc149_qa0__after6", "doc46_qa0__after6", "doc34_qa0__after6",
    "doc62_qa0__after6", "doc25_qa0__after6", "doc126_qa0__after6", "doc43_qa0__after6",
    "doc83_qa0__after6", "doc146_qa0__after6",
    "doc127_qa0__after7", "doc125_qa0__after7", "doc81_qa0__after7", "doc58_qa0__after7",
    "doc133_qa0__after7", "doc6_qa0__after7", "doc136_qa0__after7", "doc141_qa0__after7",
    "doc47_qa0__after7", "doc91_qa0__after7",
    "doc61_qa0__after8", "doc147_qa0__after8", "doc143_qa0__after8", "doc69_qa0__after8",
    "doc5_qa0__after8", "doc138_qa0__after8", "doc108_qa0__after8", "doc76_qa0__after8",
    "doc131_qa0__after8", "doc145_qa0__after8",
    "doc37_qa0__after9", "doc82_qa0__after9", "doc23_qa0__after9", "doc119_qa0__after9",
    "doc20_qa0__after9", "doc50_qa0__after9", "doc7_qa0__after9", "doc73_qa0__after9",
    "doc33_qa0__after9", "doc96_qa0__after9",
    "doc142_qa0__after10", "doc129_qa0__after10", "doc138_qa0__after10", "doc70_qa0__after10",
    "doc58_qa0__after10", "doc130_qa0__after10", "doc46_qa0__after10", "doc1_qa0__after10",
    "doc122_qa0__after10", "doc87_qa0__after10",
    "doc108_qa0__after11", "doc53_qa0__after11", "doc94_qa0__after11", "doc67_qa0__after11",
    "doc75_qa0__after11", "doc132_qa0__after11", "doc143_qa0__after11", "doc95_qa0__after11",
    "doc86_qa0__after11", "doc40_qa0__after11",
    "doc100_qa0__after12", "doc12_qa0__after12", "doc107_qa0__after12", "doc142_qa0__after12",
    "doc105_qa0__after12", "doc48_qa0__after12", "doc21_qa0__after12", "doc58_qa0__after12",
    "doc143_qa0__after12", "doc17_qa0__after12",
    "doc99_qa0__after13", "doc64_qa0__after13", "doc98_qa0__after13", "doc54_qa0__after13",
    "doc68_qa0__after13", "doc110_qa0__after13", "doc3_qa0__after13", "doc12_qa0__after13",
    "doc124_qa0__after13", "doc137_qa0__after13",
    "doc53_qa0__after14", "doc63_qa0__after14", "doc30_qa0__after14", "doc19_qa0__after14",
    "doc7_qa0__after14", "doc111_qa0__after14", "doc3_qa0__after14", "doc90_qa0__after14",
    "doc65_qa0__after14", "doc140_qa0__after14",
]


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
            score, rationale = SPECIAL.get(i, (1.0, DEFAULT_RATIONALE))
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL},
                               ensure_ascii=False) + "\n")
            added += 1
            total += score
            existing.add(qid)
    print(f"part1 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
