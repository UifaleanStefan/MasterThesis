"""Phase 1.9 — HotpotQA v4t-tuned online Protocol A judge (10 entries).

Cell: hotpotqa__v4t-tuned__online__seed42
Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  5-point rubric
Mean: 0.9000
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage3" / "judge_queue" / "hotpotqa__v4t-tuned__online__seed42" / "results.jsonl"
RESULTS.parent.mkdir(parents=True, exist_ok=True)

qid_prefix = "hotpotqa__v4t-tuned__online__"

JUDGMENTS: list[tuple[str, float, str]] = [
    # (qid_suffix, score, rationale)
    (
        "doc0_qa0__seed42",
        1.0,
        "Gold: yes. Predicted: 'Yes, both Scott Derrickson and Ed Wood were American.' Correctly answers the yes/no "
        "question and provides accurate nationality detail. Both are indeed American filmmakers. Full credit.",
    ),
    (
        "doc1_qa0__seed42",
        0.25,
        "Gold: Chief of Protocol. Predicted: 'Shirley Temple held the position of United States ambassador to Ghana "
        "and to Czechoslovakia.' The actress ID (Shirley Temple) is correct, but the government position is wrong. "
        "Shirley Temple served as US Chief of Protocol (1976-77), not as ambassador to Ghana/Czechoslovakia "
        "(those were separate later positions). The gold answer 'Chief of Protocol' is not mentioned. "
        "Wrong specific answer with no acknowledgment of uncertainty.",
    ),
    (
        "doc2_qa0__seed42",
        1.0,
        "Gold: Animorphs. Predicted: 'Animorphs'. Exact match. Full credit.",
    ),
    (
        "doc3_qa0__seed42",
        1.0,
        "Gold: no. Predicted: 'No, the Laleli Mosque is located in Laleli, Fatih, while the Esma Sultan Mansion is "
        "located in Ortaköy.' Correctly answers no and provides accurate supporting geographic detail. "
        "Laleli (Fatih) and Ortaköy are indeed different neighborhoods in Istanbul. Full credit.",
    ),
    (
        "doc4_qa0__seed42",
        1.0,
        "Gold: Greenwich Village, New York City. Predicted: 'Greenwich Village'. The core answer is correct — "
        "Greenwich Village is the neighborhood the question asks for. The gold appends 'New York City' as context "
        "but the question 'based in what New York city?' is answered by 'Greenwich Village' (which is in NYC by "
        "definition). Essentially complete and correct. Full credit.",
    ),
    (
        "doc5_qa0__seed42",
        1.0,
        "Gold: YG Entertainment. Predicted: '2014 S/S is the debut album of the South Korean boy group WINNER, "
        "which was formed by YG Entertainment.' Explicitly states 'formed by YG Entertainment' — correct and "
        "complete. Additional details (WINNER) are accurate. Full credit.",
    ),
    (
        "doc6_qa0__seed42",
        1.0,
        "Gold: Eenasul Fateh. Predicted: 'Eenasul Fateh'. Exact match. Full credit.",
    ),
    (
        "doc7_qa0__seed42",
        0.75,
        "Gold: 3,677 seated. Predicted: 'The Androscoggin Bank Colisée has a capacity of 4,000 (3,677 seated).' "
        "The correct seated capacity (3,677) is explicitly mentioned in parentheses. However, the primary figure "
        "given is '4,000' which is the total capacity, not the seated count. The question specifically asks 'can "
        "seat how many people' — a grader reading quickly would see '4,000' as the answer, which is incorrect. "
        "Correct information is present but presented in a way that could mislead. Partial credit.",
    ),
    (
        "doc8_qa0__seed42",
        1.0,
        "Gold: Terry Richardson. Predicted: 'Terry Richardson is older.' Correct answer — directly states Terry "
        "Richardson is older. Full credit.",
    ),
    (
        "doc9_qa0__seed42",
        1.0,
        "Gold: yes. Predicted: 'Yes, both Local H and For Against are from the United States.' Correctly answers "
        "yes and provides accurate supporting context. Both bands are indeed US-based. Full credit.",
    ),
]


def main() -> None:
    written = 0
    for qid_suffix, score, rationale in JUDGMENTS:
        assert score in {0.0, 0.25, 0.5, 0.75, 1.0}, f"Invalid score {score}"
        record = {
            "qid": qid_prefix + qid_suffix,
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
        }
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    print(f"Wrote {written} judgments to {RESULTS.relative_to(ROOT)}")
    scores = [s for _, s, _ in JUDGMENTS]
    print(f"Mean judge score: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
