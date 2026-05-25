"""Phase 1.9 — FB semantic-corpus calibration part2 (entries 150-299)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    25: (0.0, "[ANS] doc2 seen=18; gold=NO; pred=Yes cap-intensive $1,749M — Y/N flip."),
    28: (0.25, "[ACK] doc74 not seen; pred='$49,164M' confident-wrong specific (gold $59,268)."),
    35: (0.5, "[ANS] doc4 seen=19; gold=consumer segment shrunk 0.9%; pred=Consumer segment only (no %)."),
    37: (1.0, "[ANS] doc3 seen=19; matches 3M op margin drivers."),
    38: (0.0, "[ANS] doc14 seen=19; pred=refuses."),
    51: (0.25, "[ACK] doc74 not seen; same '$49,164M' wrong."),
    55: (0.0, "[ANS] doc19 seen=21; pred=refuses."),
    59: (0.0, "[ANS] doc18 seen=21; pred=refuses."),
    60: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    63: (0.0, "[ANS] doc11 seen=22; pred=refuses."),
    86: (0.0, "[ANS] doc11 seen=24; pred=refuses."),
    89: (1.0, "[ANS] doc15 seen=24; gold=0; pred=0 exact."),
    90: (0.5, "[ACK] doc125 not seen; correct WK 'not approved'."),
    92: (1.0, "[ANS] doc1 seen=25; gold=$8.70B; pred=8.738B exact (semantic retains)."),
    105: (0.0, "[ANS] doc11 seen=26; pred=refuses."),
    108: (0.0, "[ANS] doc2 seen=26; same Y/N flip."),
    119: (0.25, "[ACK] doc74 not seen; same wrong."),
    125: (0.0, "[ANS] doc20 seen=28; pred=refuses."),
    128: (1.0, "[ANS] doc0 seen=28; gold=$1577; pred=1,577 exact (TF-IDF retains)."),
    149: (0.0, "[ANS] doc18 seen=30; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc80_qa0__after15", "doc81_qa0__after15", "doc26_qa0__after15", "doc46_qa0__after15",
    "doc127_qa0__after15", "doc23_qa0__after15", "doc36_qa0__after15", "doc130_qa0__after15",
    "doc48_qa0__after15", "doc34_qa0__after15",
    "doc71_qa0__after16", "doc115_qa0__after16", "doc138_qa0__after16", "doc86_qa0__after16",
    "doc136_qa0__after16", "doc145_qa0__after16", "doc89_qa0__after16", "doc105_qa0__after16",
    "doc116_qa0__after16", "doc23_qa0__after16",
    "doc103_qa0__after17", "doc73_qa0__after17", "doc124_qa0__after17", "doc18_qa0__after17",
    "doc115_qa0__after17", "doc2_qa0__after17", "doc64_qa0__after17", "doc85_qa0__after17",
    "doc74_qa0__after17", "doc33_qa0__after17",
    "doc37_qa0__after18", "doc39_qa0__after18", "doc139_qa0__after18", "doc34_qa0__after18",
    "doc109_qa0__after18", "doc4_qa0__after18", "doc49_qa0__after18", "doc3_qa0__after18",
    "doc14_qa0__after18", "doc97_qa0__after18",
    "doc136_qa0__after19", "doc113_qa0__after19", "doc57_qa0__after19", "doc59_qa0__after19",
    "doc75_qa0__after19", "doc36_qa0__after19", "doc110_qa0__after19", "doc51_qa0__after19",
    "doc119_qa0__after19", "doc138_qa0__after19",
    "doc105_qa0__after20", "doc74_qa0__after20", "doc84_qa0__after20", "doc36_qa0__after20",
    "doc83_qa0__after20", "doc19_qa0__after20", "doc140_qa0__after20", "doc61_qa0__after20",
    "doc111_qa0__after20", "doc18_qa0__after20",
    "doc122_qa0__after21", "doc113_qa0__after21", "doc91_qa0__after21", "doc11_qa0__after21",
    "doc110_qa0__after21", "doc140_qa0__after21", "doc63_qa0__after21", "doc48_qa0__after21",
    "doc87_qa0__after21", "doc68_qa0__after21",
    "doc120_qa0__after22", "doc114_qa0__after22", "doc99_qa0__after22", "doc80_qa0__after22",
    "doc45_qa0__after22", "doc68_qa0__after22", "doc53_qa0__after22", "doc84_qa0__after22",
    "doc43_qa0__after22", "doc61_qa0__after22",
    "doc48_qa0__after23", "doc66_qa0__after23", "doc63_qa0__after23", "doc113_qa0__after23",
    "doc117_qa0__after23", "doc41_qa0__after23", "doc11_qa0__after23", "doc128_qa0__after23",
    "doc119_qa0__after23", "doc15_qa0__after23",
    "doc125_qa0__after24", "doc26_qa0__after24", "doc1_qa0__after24", "doc32_qa0__after24",
    "doc61_qa0__after24", "doc126_qa0__after24", "doc134_qa0__after24", "doc53_qa0__after24",
    "doc120_qa0__after24", "doc135_qa0__after24",
    "doc59_qa0__after25", "doc139_qa0__after25", "doc134_qa0__after25", "doc83_qa0__after25",
    "doc31_qa0__after25", "doc11_qa0__after25", "doc26_qa0__after25", "doc94_qa0__after25",
    "doc2_qa0__after25", "doc49_qa0__after25",
    "doc36_qa0__after26", "doc131_qa0__after26", "doc115_qa0__after26", "doc85_qa0__after26",
    "doc118_qa0__after26", "doc77_qa0__after26", "doc110_qa0__after26", "doc63_qa0__after26",
    "doc40_qa0__after26", "doc74_qa0__after26",
    "doc102_qa0__after27", "doc124_qa0__after27", "doc39_qa0__after27", "doc105_qa0__after27",
    "doc132_qa0__after27", "doc20_qa0__after27", "doc106_qa0__after27", "doc80_qa0__after27",
    "doc0_qa0__after27", "doc104_qa0__after27",
    "doc89_qa0__after28", "doc63_qa0__after28", "doc41_qa0__after28", "doc29_qa0__after28",
    "doc124_qa0__after28", "doc109_qa0__after28", "doc106_qa0__after28", "doc39_qa0__after28",
    "doc56_qa0__after28", "doc70_qa0__after28",
    "doc147_qa0__after29", "doc135_qa0__after29", "doc124_qa0__after29", "doc97_qa0__after29",
    "doc58_qa0__after29", "doc91_qa0__after29", "doc138_qa0__after29", "doc108_qa0__after29",
    "doc71_qa0__after29", "doc18_qa0__after29",
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
    print(f"part2 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
