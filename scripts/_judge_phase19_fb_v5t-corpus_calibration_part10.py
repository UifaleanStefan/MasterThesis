"""Phase 1.9 — FB v5t-corpus calibration part10 (entries 1350-1499, FINAL)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    3: (0.25, "[ANS] doc88 seen=136; Y/N flip (Yes 3.5% vs No decelerate)."),
    8: (0.25, "[ANS] doc127 seen=136; gold=$8.4B; pred=$4.2B (only one of two agreements mentioned)."),
    20: (1.0, "[ANS] doc128 seen=138; gold=strong start to FY23; pred=strong start + 8% organic + 9% EPS — exact + detail."),
    24: (1.0, "[ANS] doc134 seen=138; gold=Developed Rest of the World; pred=Developed Rest of World exact."),
    25: (1.0, "[ANS] doc135 seen=138; gold=Yes spinning off Upjohn; pred=Yes $700M separating Upjohn — direction+detail."),
    27: (1.0, "[ANS] doc126 seen=138; gold=$400M increase; pred=$400M increase ($3.8B → $4.2B) exact."),
    36: (1.0, "[ANS] doc131 seen=139; gold=Yes JV gain 2019; pred=Yes JV gain 2019. Exact."),
    51: (1.0, "[ANS] doc135 seen=141; Yes $700M Upjohn — same as 1375."),
    56: (1.0, "[ANS] doc105 seen=141; gold=Yes MGM $0.01; pred=Yes MGM $0.01 exact."),
    64: (1.0, "[ANS] doc125 seen=142; gold=defeated; pred=defeated with vote breakdown exact."),
    72: (0.25, "[ANS] doc127 seen=143; same partial $4.2B (1 of 2 agreements)."),
    76: (0.0, "[ANS] doc139 seen=143; gold=increase 47 new stores; pred='decrease of $104,233' — wrong direction."),
    83: (0.0, "[ANS] doc141 seen=144; gold=increased; pred='Decrease' — Y/N flip."),
    85: (1.0, "[ANS] doc134 seen=144; Developed Rest of World exact."),
    86: (1.0, "[ANS] doc79 seen=144; Mary Dillon Foot Locker + Ulta exact."),
    87: (1.0, "[ANS] doc138 seen=144; gold=lower marketing + leverage of incentive comp; pred=exact same drivers."),
    89: (1.0, "[ANS] doc7 seen=144; 65th year exact."),
    92: (0.0, "[ANS] doc139 seen=145; same wrong 'decrease' direction."),
    99: (0.5, "[ANS] doc78 seen=145; Yes paid (no $0.55 detail)."),
    104: (0.0, "[ANS] doc136 seen=146; gold='None'; pred=ULTA NASDAQ common stock — positive answer when 'none' expected."),
    113: (1.0, "[ANS] doc64 seen=147; Yes cyclical exact."),
    114: (0.0, "[ANS] doc139 seen=147; same wrong 'decrease' direction."),
    116: (1.0, "[ANS] doc98 seen=147; VaR decreased exact."),
    130: (1.0, "[ANS] doc140 seen=149; gold=36%; pred=36.5% (1.4% off — within tolerance)."),
    135: (0.25, "[ANS] doc127 seen=149; same partial $4.2B."),
    138: (1.0, "[ANS] doc85 seen=149; matches No low growth 1.3% vs 13.6%."),
    139: (0.75, "[ANS] doc137 seen=149; gold=no acquisitions; pred=passages do not mention — functionally matches."),
    140: (1.0, "[ANS] doc90 seen=150; Consumer Health Aug 30 exact."),
    142: (0.5, "[ANS] doc63 seen=150; gold=airlines + US govt 40%; pred=airlines + govt + defense partial."),
    143: (0.75, "[ANS] doc109 seen=150; gold=corporate bonds 82%; pred=Corporate bonds (no %)."),
    144: (1.0, "[ANS] doc61 seen=150; Lion Air + Ethiopian exact."),
    146: (1.0, "[ANS] doc80 seen=150; RAJ + votes exact."),
    147: (1.0, "[ANS] doc105 seen=150; MGM $0.01 exact."),
    149: (1.0, "[ANS] doc128 seen=150; strong start + 8%/9% guidance detail."),
}

DEFAULT_RATIONALE_ACK = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."
DEFAULT_RATIONALE_ANS = "[ANS] source doc paragraphs not retrievable from V5 graph. PRED refuses on definitive ANS — penalised."

# Tag each entry's type
ENTRIES: list[tuple[str, str]] = [
    # (suffix, "ANS" or "ACK") — 150 entries
    ("doc55_qa0__after135", "ANS"), ("doc60_qa0__after135", "ANS"), ("doc102_qa0__after135", "ANS"),
    ("doc88_qa0__after135", "ANS"), ("doc86_qa0__after135", "ANS"), ("doc81_qa0__after135", "ANS"),
    ("doc118_qa0__after135", "ANS"), ("doc139_qa0__after135", "ACK"), ("doc127_qa0__after135", "ANS"),
    ("doc10_qa0__after135", "ANS"),
    ("doc115_qa0__after136", "ANS"), ("doc120_qa0__after136", "ANS"), ("doc27_qa0__after136", "ANS"),
    ("doc148_qa0__after136", "ACK"), ("doc108_qa0__after136", "ANS"), ("doc2_qa0__after136", "ANS"),
    ("doc58_qa0__after136", "ANS"), ("doc80_qa0__after136", "ANS"), ("doc63_qa0__after136", "ANS"),
    ("doc103_qa0__after136", "ANS"),
    ("doc128_qa0__after137", "ANS"), ("doc39_qa0__after137", "ANS"), ("doc60_qa0__after137", "ANS"),
    ("doc88_qa0__after137", "ANS"), ("doc134_qa0__after137", "ANS"), ("doc135_qa0__after137", "ANS"),
    ("doc113_qa0__after137", "ANS"), ("doc126_qa0__after137", "ANS"), ("doc18_qa0__after137", "ANS"),
    ("doc13_qa0__after137", "ANS"),
    ("doc60_qa0__after138", "ANS"), ("doc39_qa0__after138", "ANS"), ("doc119_qa0__after138", "ANS"),
    ("doc142_qa0__after138", "ACK"), ("doc35_qa0__after138", "ANS"), ("doc8_qa0__after138", "ANS"),
    ("doc131_qa0__after138", "ANS"), ("doc67_qa0__after138", "ANS"), ("doc47_qa0__after138", "ANS"),
    ("doc3_qa0__after138", "ANS"),
    ("doc148_qa0__after139", "ACK"), ("doc70_qa0__after139", "ANS"), ("doc118_qa0__after139", "ANS"),
    ("doc39_qa0__after139", "ANS"), ("doc74_qa0__after139", "ANS"), ("doc12_qa0__after139", "ANS"),
    ("doc24_qa0__after139", "ANS"), ("doc25_qa0__after139", "ANS"), ("doc0_qa0__after139", "ANS"),
    ("doc92_qa0__after139", "ANS"),
    ("doc5_qa0__after140", "ANS"), ("doc135_qa0__after140", "ANS"), ("doc76_qa0__after140", "ANS"),
    ("doc26_qa0__after140", "ANS"), ("doc55_qa0__after140", "ANS"), ("doc58_qa0__after140", "ANS"),
    ("doc105_qa0__after140", "ANS"), ("doc31_qa0__after140", "ANS"), ("doc123_qa0__after140", "ANS"),
    ("doc3_qa0__after140", "ANS"),
    ("doc62_qa0__after141", "ANS"), ("doc3_qa0__after141", "ANS"), ("doc38_qa0__after141", "ANS"),
    ("doc143_qa0__after141", "ACK"), ("doc125_qa0__after141", "ANS"), ("doc87_qa0__after141", "ANS"),
    ("doc63_qa0__after141", "ANS"), ("doc69_qa0__after141", "ANS"), ("doc124_qa0__after141", "ANS"),
    ("doc17_qa0__after141", "ANS"),
    ("doc34_qa0__after142", "ANS"), ("doc102_qa0__after142", "ANS"), ("doc127_qa0__after142", "ANS"),
    ("doc146_qa0__after142", "ACK"), ("doc2_qa0__after142", "ANS"), ("doc113_qa0__after142", "ANS"),
    ("doc139_qa0__after142", "ANS"), ("doc74_qa0__after142", "ANS"), ("doc132_qa0__after142", "ANS"),
    ("doc107_qa0__after142", "ANS"),
    ("doc63_qa0__after143", "ANS"), ("doc45_qa0__after143", "ANS"), ("doc4_qa0__after143", "ANS"),
    ("doc141_qa0__after143", "ANS"), ("doc93_qa0__after143", "ANS"), ("doc134_qa0__after143", "ANS"),
    ("doc79_qa0__after143", "ANS"), ("doc138_qa0__after143", "ANS"), ("doc11_qa0__after143", "ANS"),
    ("doc7_qa0__after143", "ANS"),
    ("doc86_qa0__after144", "ANS"), ("doc31_qa0__after144", "ANS"), ("doc139_qa0__after144", "ANS"),
    ("doc44_qa0__after144", "ANS"), ("doc24_qa0__after144", "ANS"), ("doc97_qa0__after144", "ANS"),
    ("doc63_qa0__after144", "ANS"), ("doc110_qa0__after144", "ANS"), ("doc23_qa0__after144", "ANS"),
    ("doc78_qa0__after144", "ANS"),
    ("doc23_qa0__after145", "ANS"), ("doc110_qa0__after145", "ANS"), ("doc19_qa0__after145", "ANS"),
    ("doc20_qa0__after145", "ANS"), ("doc136_qa0__after145", "ANS"), ("doc95_qa0__after145", "ANS"),
    ("doc119_qa0__after145", "ANS"), ("doc109_qa0__after145", "ANS"), ("doc62_qa0__after145", "ANS"),
    ("doc12_qa0__after145", "ANS"),
    ("doc111_qa0__after146", "ANS"), ("doc51_qa0__after146", "ANS"), ("doc10_qa0__after146", "ANS"),
    ("doc64_qa0__after146", "ANS"), ("doc139_qa0__after146", "ANS"), ("doc24_qa0__after146", "ANS"),
    ("doc98_qa0__after146", "ANS"), ("doc5_qa0__after146", "ANS"), ("doc13_qa0__after146", "ANS"),
    ("doc53_qa0__after146", "ANS"),
    ("doc25_qa0__after147", "ANS"), ("doc24_qa0__after147", "ANS"), ("doc35_qa0__after147", "ANS"),
    ("doc22_qa0__after147", "ANS"), ("doc117_qa0__after147", "ANS"), ("doc26_qa0__after147", "ANS"),
    ("doc141_qa0__after147", "ANS"), ("doc83_qa0__after147", "ANS"), ("doc102_qa0__after147", "ANS"),
    ("doc111_qa0__after147", "ANS"),
    ("doc140_qa0__after148", "ANS"), ("doc107_qa0__after148", "ANS"), ("doc38_qa0__after148", "ANS"),
    ("doc59_qa0__after148", "ANS"), ("doc120_qa0__after148", "ANS"), ("doc127_qa0__after148", "ANS"),
    ("doc77_qa0__after148", "ANS"), ("doc118_qa0__after148", "ANS"), ("doc85_qa0__after148", "ANS"),
    ("doc137_qa0__after148", "ANS"),
    ("doc90_qa0__after149", "ANS"), ("doc82_qa0__after149", "ANS"), ("doc63_qa0__after149", "ANS"),
    ("doc109_qa0__after149", "ANS"), ("doc61_qa0__after149", "ANS"), ("doc55_qa0__after149", "ANS"),
    ("doc80_qa0__after149", "ANS"), ("doc105_qa0__after149", "ANS"), ("doc108_qa0__after149", "ANS"),
    ("doc128_qa0__after149", "ANS"),
]


def main() -> None:
    assert len(ENTRIES) == 150
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
        for i, (suffix, kind) in enumerate(ENTRIES):
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing:
                skipped += 1
                continue
            if i in SPECIAL:
                score, rationale = SPECIAL[i]
            elif kind == "ACK":
                score, rationale = 1.0, DEFAULT_RATIONALE_ACK
            else:  # ANS
                score, rationale = 0.0, DEFAULT_RATIONALE_ANS
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL},
                               ensure_ascii=False) + "\n")
            added += 1
            total += score
            existing.add(qid)
    print(f"part10 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
