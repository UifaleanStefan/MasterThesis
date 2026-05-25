"""Phase 1.9 — FB v5t-corpus calibration part3 (entries 300-449)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc12 seen=31; pred=refuses."),
    5: (0.0, "[ANS] doc0 seen=31; pred=refuses."),
    7: (0.0, "[ANS] doc5 seen=31; pred=refuses on definitive ANS."),
    9: (0.5, "[ACK] doc90 not seen; pred=Consumer Health Aug 30 correct WK."),
    12: (1.0, "[ANS] doc21 seen=32; gold=$1616M; pred=$1,615.9M exact (V5 graph retains)."),
    13: (0.25, "[ACK] doc63 not seen; pred partial WK."),
    17: (0.0, "[ANS] doc18 seen=32; pred=refuses."),
    21: (0.0, "[ANS] doc18 seen=33; pred=refuses."),
    22: (1.0, "[ANS] doc7 seen=33; gold=Yes 65 years; pred=Yes 65th year. Exact."),
    33: (0.25, "[ACK] doc34 not seen; PRED dismisses metric as not useful (wrong framing)."),
    35: (1.0, "[ANS] doc15 seen=34; gold=0; pred=0 exact."),
    36: (0.5, "[ACK] doc90 not seen; correct WK."),
    39: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    41: (0.25, "[ANS] doc26 seen=35; gold=NO declined 0.8%; pred 'not useful framing' + 'cannot determine' partial mismatch."),
    46: (0.75, "[ANS] doc25 seen=35; gold=Amcor packaging leader; pred=packaging industry. Less specific but correct."),
    47: (1.0, "[ANS] doc34 seen=35; gold=Xilinx amortization; pred=exact."),
    49: (0.0, "[ANS] doc29 seen=35; gold=flat; pred=-5% wrong direction."),
    68: (0.0, "[ANS] doc31 seen=37; pred=refuses on definitive ANS."),
    69: (0.0, "[ANS] doc3 seen=37; pred=refuses."),
    72: (0.0, "[ANS] doc11 seen=38; pred=refuses."),
    73: (0.0, "[ANS] doc10 seen=38; pred=refuses."),
    74: (0.5, "[ACK] doc90 not seen; correct WK."),
    80: (0.5, "[ACK] doc90 not seen; correct WK."),
    84: (0.0, "[ANS] doc1 seen=39; pred=refuses."),
    85: (0.5, "[ANS] doc27 seen=39; gold=87% employee; pred=lists employee+fixed asset+other costs but no %."),
    87: (0.0, "[ANS] doc24 seen=39; pred=refuses on definitive ANS."),
    95: (0.0, "[ANS] doc8 seen=40; pred=refuses."),
    96: (1.0, "[ANS] doc33 seen=40; matches gold (Data Center, Gaming, Embedded)."),
    99: (0.0, "[ANS] doc2 seen=40; pred=refuses on definitive ANS."),
    100: (0.0, "[ANS] doc16 seen=41; pred=refuses."),
    107: (0.0, "[ANS] doc11 seen=41; pred=refuses."),
    118: (1.0, "[ANS] doc21 seen=42; gold=$1616M; pred=$1,615.9M exact (V5 graph retains)."),
    124: (1.0, "[ANS] doc36 seen=43; gold=Data Center; pred=Data Center segment exact."),
    130: (0.75, "[ANS] doc25 seen=44; same packaging industry match."),
    136: (0.5, "[ANS] doc27 seen=44; same partial (no %)."),
    138: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    139: (0.0, "[ANS] doc24 seen=44; pred=refuses."),
    141: (0.75, "[ANS] doc35 seen=45; gold=AMD operations highest; pred='Operating activities most cash flow' (no $)."),
    142: (0.0, "[ANS] doc17 seen=45; pred=refuses."),
    143: (0.0, "[ANS] doc30 seen=45; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc12_qa0__after30", "doc98_qa0__after30", "doc47_qa0__after30", "doc97_qa0__after30",
    "doc52_qa0__after30", "doc0_qa0__after30", "doc60_qa0__after30", "doc5_qa0__after30",
    "doc42_qa0__after30", "doc90_qa0__after30",
    "doc124_qa0__after31", "doc91_qa0__after31", "doc21_qa0__after31", "doc63_qa0__after31",
    "doc120_qa0__after31", "doc67_qa0__after31", "doc139_qa0__after31", "doc18_qa0__after31",
    "doc135_qa0__after31", "doc141_qa0__after31",
    "doc117_qa0__after32", "doc18_qa0__after32", "doc7_qa0__after32", "doc115_qa0__after32",
    "doc47_qa0__after32", "doc106_qa0__after32", "doc87_qa0__after32", "doc56_qa0__after32",
    "doc77_qa0__after32", "doc112_qa0__after32",
    "doc135_qa0__after33", "doc144_qa0__after33", "doc18_qa0__after33", "doc34_qa0__after33",
    "doc72_qa0__after33", "doc15_qa0__after33", "doc90_qa0__after33", "doc89_qa0__after33",
    "doc64_qa0__after33", "doc125_qa0__after33",
    "doc130_qa0__after34", "doc26_qa0__after34", "doc68_qa0__after34", "doc40_qa0__after34",
    "doc129_qa0__after34", "doc144_qa0__after34", "doc25_qa0__after34", "doc34_qa0__after34",
    "doc131_qa0__after34", "doc29_qa0__after34",
    "doc136_qa0__after35", "doc93_qa0__after35", "doc146_qa0__after35", "doc149_qa0__after35",
    "doc42_qa0__after35", "doc85_qa0__after35", "doc98_qa0__after35", "doc92_qa0__after35",
    "doc78_qa0__after35", "doc100_qa0__after35",
    "doc88_qa0__after36", "doc69_qa0__after36", "doc120_qa0__after36", "doc112_qa0__after36",
    "doc133_qa0__after36", "doc136_qa0__after36", "doc145_qa0__after36", "doc131_qa0__after36",
    "doc31_qa0__after36", "doc3_qa0__after36",
    "doc52_qa0__after37", "doc70_qa0__after37", "doc11_qa0__after37", "doc10_qa0__after37",
    "doc90_qa0__after37", "doc54_qa0__after37", "doc50_qa0__after37", "doc107_qa0__after37",
    "doc129_qa0__after37", "doc108_qa0__after37",
    "doc90_qa0__after38", "doc138_qa0__after38", "doc43_qa0__after38", "doc71_qa0__after38",
    "doc1_qa0__after38", "doc27_qa0__after38", "doc140_qa0__after38", "doc24_qa0__after38",
    "doc135_qa0__after38", "doc88_qa0__after38",
    "doc115_qa0__after39", "doc92_qa0__after39", "doc146_qa0__after39", "doc76_qa0__after39",
    "doc80_qa0__after39", "doc8_qa0__after39", "doc33_qa0__after39", "doc95_qa0__after39",
    "doc46_qa0__after39", "doc2_qa0__after39",
    "doc16_qa0__after40", "doc93_qa0__after40", "doc128_qa0__after40", "doc110_qa0__after40",
    "doc59_qa0__after40", "doc54_qa0__after40", "doc135_qa0__after40", "doc11_qa0__after40",
    "doc53_qa0__after40", "doc57_qa0__after40",
    "doc85_qa0__after41", "doc88_qa0__after41", "doc53_qa0__after41", "doc61_qa0__after41",
    "doc46_qa0__after41", "doc124_qa0__after41", "doc84_qa0__after41", "doc134_qa0__after41",
    "doc21_qa0__after41", "doc87_qa0__after41",
    "doc106_qa0__after42", "doc124_qa0__after42", "doc98_qa0__after42", "doc56_qa0__after42",
    "doc36_qa0__after42", "doc51_qa0__after42", "doc111_qa0__after42", "doc60_qa0__after42",
    "doc148_qa0__after42", "doc50_qa0__after42",
    "doc25_qa0__after43", "doc114_qa0__after43", "doc133_qa0__after43", "doc141_qa0__after43",
    "doc55_qa0__after43", "doc85_qa0__after43", "doc27_qa0__after43", "doc94_qa0__after43",
    "doc122_qa0__after43", "doc24_qa0__after43",
    "doc76_qa0__after44", "doc35_qa0__after44", "doc17_qa0__after44", "doc30_qa0__after44",
    "doc66_qa0__after44", "doc101_qa0__after44", "doc95_qa0__after44", "doc67_qa0__after44",
    "doc53_qa0__after44", "doc141_qa0__after44",
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
    print(f"part3 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
