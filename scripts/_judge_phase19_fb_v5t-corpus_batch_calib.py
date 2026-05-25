"""Phase 1.9 — FB v5t-corpus batch_calib (end-of-corpus 150 questions).

All [ANS] expected. V5 graph at end retains more than FlatMemory but less than
fully-tuned configs.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__batch_calib__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Most ANS predictions refuse → 0.0; exceptions get explicit scores
SPECIAL: dict[int, tuple[float, str]] = {
    4: (0.0, "[ANS] doc4; gold=consumer segment shrunk 0.9%; pred='Corporate and other segment' — wrong segment."),
    7: (1.0, "[ANS] doc7; gold=Yes 65 years; pred=Yes 65th year exact."),
    15: (1.0, "[ANS] doc15; gold=0; pred=0 exact."),
    44: (1.0, "[ANS] doc44; gold=Yes; pred=Yes Card Member retention exact."),
    52: (0.75, "[ANS] doc52; gold=Best Buy operations $1.8B; pred='operating activities most cash flow' (no $)."),
    54: (1.0, "[ANS] doc54; gold=982→969; pred=982→969 exact."),
    60: (0.5, "[ANS] doc60; only Commercial Airplanes (gold has 3 categories)."),
    61: (1.0, "[ANS] doc61; Lion Air + Ethiopian exact."),
    63: (0.5, "[ANS] doc63; partial WK (airlines + govt + defense)."),
    64: (1.0, "[ANS] doc64; Yes cyclical exact."),
    65: (0.5, "[ANS] doc65; 787 + 737 (no 777X)."),
    74: (0.25, "[ANS] doc74; gold=$59268; pred=$53,837 (9% off — outside tolerance)."),
    78: (0.5, "[ANS] doc78; Yes paid (no $0.55)."),
    79: (1.0, "[ANS] doc79; Mary Dillon Ulta exact."),
    80: (1.0, "[ANS] doc80; RAJ + 16,105,005 votes exact."),
    85: (1.0, "[ANS] doc85; No low growth 1.3% vs 13.6% exact."),
    87: (0.25, "[ANS] doc87; refuses with meta-claim about JnJ diverse ops — partial."),
    89: (0.0, "[ANS] doc89; gold=US 3.0% vs Intl -0.6%; pred='US 1.3%' wrong specifics."),
    90: (1.0, "[ANS] doc90; Consumer Health Aug 30 exact."),
    92: (0.0, "[ANS] doc92; gold=$13.2B Kenvue proceeds; pred='$3.7B' (72% off)."),
    96: (1.0, "[ANS] doc96; JPM gross margin not relevant exact paraphrase."),
    97: (0.75, "[ANS] doc97; gold=Corporate & Investment Bank ($3725M); pred=Corporate & Investment Bank (no $)."),
    98: (1.0, "[ANS] doc98; Yes VaR decreased $7M exact."),
    105: (1.0, "[ANS] doc105; MGM $0.01 exact."),
    109: (0.75, "[ANS] doc109; Corporate bonds (no %)."),
    114: (1.0, "[ANS] doc114; gold=55.1%; pred=56.2% (2% off — within tolerance)."),
    116: (0.25, "[ANS] doc116; gold=3.46; pred=4.29 (24% off)."),
    121: (0.75, "[ANS] doc121; gold=No not material legal battles; pred='party to litigation but no material adverse effect' — functionally matches."),
    122: (0.0, "[ANS] doc122; gold=$411M; pred='0' confident-wrong."),
    125: (1.0, "[ANS] doc125; defeated with vote breakdown exact."),
    126: (1.0, "[ANS] doc126; $400M increase $3.8B → $4.2B exact."),
    127: (0.25, "[ANS] doc127; pred=$4.2B (1 of 2 agreements; gold $8.4B total)."),
    128: (1.0, "[ANS] doc128; strong start + 8%/9% guidance details."),
    129: (1.0, "[ANS] doc129; 1pp from 8% to 9% exact."),
    130: (0.25, "[ANS] doc130; gold=Yes Pfizer PPNE positive; pred=Yes ($9159M→$21979M) but uses net income (wrong metric, right direction)."),
    131: (0.25, "[ANS] doc131; gold=Yes JV gain 2019; pred=Yes JV gain but specifies 2021 (wrong year)."),
    132: (0.0, "[ANS] doc132; pred=refuses (gold Trillium, Array, Therachon)."),
    133: (0.0, "[ANS] doc133; gold=$77.78M; pred='$700M' (800% off)."),
    134: (1.0, "[ANS] doc134; Developed Rest of World exact."),
    135: (1.0, "[ANS] doc135; Yes Pfizer separating Upjohn ($700M cost) — matches direction."),
    136: (1.0, "[ANS] doc136; gold=There are none; pred='None' exact."),
    137: (0.75, "[ANS] doc137; gold=Ulta no acquisitions FY23/FY22; pred='no major acquisitions reported' matches."),
    138: (1.0, "[ANS] doc138; gold=lower marketing + leverage of incentive comp; pred=exact + deleverage details."),
    139: (0.0, "[ANS] doc139; gold=increase 47 new stores; pred='decrease $104,233' — wrong direction."),
    140: (1.0, "[ANS] doc140; gold=36%; pred=36.5% (1.4% off — within tolerance)."),
    141: (0.0, "[ANS] doc141; gold=increased; pred='Decrease' — Y/N flip."),
    142: (0.75, "[ANS] doc142; gold=Cross currency swaps $32,502M; pred=Cross currency swaps (no $)."),
    143: (0.5, "[ANS] doc143; gold=$1097M pension + $862M health; pred=$1097M (pension only). Partial."),
    144: (0.0, "[ANS] doc144; gold=No Verizon quick ratio 0.54; pred=refuses on definitive ANS."),
    145: (0.75, "[ANS] doc145; gold=Yes Verizon cap-intensive (ratio 2.77); pred=Yes ($307,689M PP&E) — different metric, matches direction."),
    146: (0.0, "[ANS] doc146; gold=No decreased $229M; pred='Yes increased $7,443→$9,963' — Y/N flip (and likely wrong source figures)."),
    147: (0.25, "[ANS] doc147; gold=42.69 Walmart DPO; pred=36.36 (15% off)."),
    148: (0.0, "[ANS] doc148; gold=0.2% (change); pred=4.1% (level, wrong scope)."),
    149: (0.0, "[ANS] doc149; gold=6.2%; pred=4.0% (35% off)."),
}

DEFAULT_RATIONALE = "[ANS] V5 graph does not retain this doc's content. PRED refuses on definitive ANS — penalised."

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
