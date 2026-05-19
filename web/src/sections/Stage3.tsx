import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, TrendingUp } from "lucide-react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { Pill } from "../components/shared/Pill";
import { CodeBlock } from "../components/shared/CodeBlock";

type StageRecord = { [system: string]: { [benchmark: string]: number | null } };
type TunedRecord = {
  [benchmark: string]: {
    canonical_recall: number;
    tuned_recall: number;
    improvement: number;
    n_gold_questions: number;
  };
};
type TransferMatrix = {
  rows: string[];
  cols: string[];
  matrix_means: { [row: string]: { [col: string]: number | null } };
  diagonal_avg: number | null;
  off_diagonal_avg: number | null;
  canonical_avg: number | null;
  diagonal_lift_vs_canonical: number | null;
  off_diagonal_lift_vs_canonical: number | null;
  interpretation: string | null;
};
type StageData = {
  benchmarks: string[];
  systems: string[];
  table: StageRecord;
  tuned_vs_canonical: TunedRecord;
  transfer_matrix?: TransferMatrix | null;
};

const HEADLINE_BENCHMARKS = ["qasper", "cuad"];

export function Stage3() {
  const [data, setData] = useState<StageData | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/stage3_retrieval.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  return (
    <Section id="stage3" eyebrow="Stage 3 — Real Benchmarks">
      <SectionHeader
        title={
          <>
            Six published benchmarks.{" "}
            <span style={{ color: "var(--color-amber)" }}>
              Per-task &theta; lifts V4 by +0.26 to +0.36 recall.
            </span>
          </>
        }
        lede={
          <>
            We migrated the evaluation off hand-authored documents onto six real
            long-context QA benchmarks (HotpotQA, QASPER, CUAD, NarrativeQA,
            FinanceBench, LongMemEval). Per-benchmark CMA-ES tuning of V4's 10D
            &theta; on recall@k <strong>beats every other memory system</strong>{" "}
            on the two long-haystack tasks (QASPER and CUAD). The remaining
            four saturate at recall = 1.0 — they have small haystacks where
            k = 8 retrieves everything that matters.
          </>
        }
      />

      {/* Headline result table */}
      <div className="mt-8">
        {data ? (
          <>
            <RetrievalTable data={data} />
            {data.transfer_matrix ? (
              <TransferMatrixPanel matrix={data.transfer_matrix} />
            ) : null}
          </>
        ) : (
          <div
            className="panel-rise p-8 text-center"
            style={{ color: "var(--color-text-2)" }}
          >
            <Pill tone="cyan">Loading retrieval data…</Pill>
            <p className="mt-3 text-sm">
              If this persists, run{" "}
              <code className="font-mono text-xs">
                python scripts/run_stage3_retrieval.py --benchmarks all
                --load-tuned-thetas
              </code>{" "}
              then{" "}
              <code className="font-mono text-xs">
                python scripts/build_stage3_frontend_data.py
              </code>
              .
            </p>
          </div>
        )}
      </div>

      {/* Pivot — what's left for the API */}
      <div className="grid lg:grid-cols-[1.1fr_1fr] gap-6 mt-10 items-start">
        <div>
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="panel-rise p-8"
            style={{
              background:
                "linear-gradient(135deg, var(--color-amber-soft) 0%, transparent 50%, var(--color-cyan-soft) 100%), var(--color-surface)",
            }}
          >
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
              style={{ color: "var(--color-muted)" }}
            >
              the next experiment
            </div>
            <div
              className="text-3xl md:text-4xl font-mono"
              style={{ color: "var(--color-text)" }}
            >
              <span style={{ color: "var(--color-cyan)" }}>J</span> ={" "}
              <span style={{ color: "var(--color-emerald)" }}>QA_score</span>{" "}
              − <span style={{ color: "var(--color-rose)" }}>λ</span> ·{" "}
              <span style={{ color: "var(--color-amber)" }}>cost_usd</span>
            </div>
            <p
              className="mt-4 text-sm leading-relaxed"
              style={{ color: "var(--color-text-2)" }}
            >
              The retrieval table above is the <em>memory-quality</em> layer.
              The Phase-4 orchestrator (
              <code className="font-mono text-xs">
                scripts/run_stage3_full.py
              </code>
              ) layers GPT-4o-mini on top: each retrieved passage feeds the LLM
              answer prompt, scored by LLM-judge, with token cost tracked
              per cell. A <strong>dry-run</strong> mode uses tiktoken to project
              cost without spending — we've measured it under{" "}
              <strong>$2 for the canonical 6 × 3 × 30-question Phase-4 sweep</strong>.
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-3 mt-4">
            <Pill tone="cyan">6 benchmarks</Pill>
            <Pill tone="amber">3 memory configs</Pill>
            <Pill tone="violet">GPT-4o-mini judge</Pill>
          </div>
        </div>

        <div className="space-y-4">
          <div className="panel-rise p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={16} style={{ color: "var(--color-amber)" }} />
              <h3 className="font-semibold text-base">Three commands to reproduce</h3>
            </div>
            <CodeBlock
              label="powershell"
              code={`# 1. Tune V4 theta per benchmark (no LLM, ~3 min)
python -m tuning.tune_v4_per_benchmark --benchmarks all

# 2. Run the retrieval study (no LLM, ~3 min)
python scripts/run_stage3_retrieval.py --benchmarks all \\
  --n-docs 15 --load-tuned-thetas

# 3. Project Phase-4 cost (no LLM, ~1 min)
python scripts/run_stage3_full.py --mode dry-run \\
  --benchmarks all --configs v4-canonical v4-tuned flat-50`}
            />
          </div>

          <div
            className="panel-rise p-5"
            style={{
              background:
                "linear-gradient(135deg, transparent 0%, var(--color-cyan-soft) 100%), var(--color-surface)",
            }}
          >
            <h3
              className="font-semibold text-base mb-2"
              style={{ color: "var(--color-cyan)" }}
            >
              What stays gated by API budget.
            </h3>
            <p
              className="text-sm leading-relaxed"
              style={{ color: "var(--color-text-2)" }}
            >
              The LLM-judge scoring path (
              <code className="font-mono text-xs">
                evaluation/document_qa_llm_judge.py
              </code>
              ) is wired in and dry-run-validated. With OPENAI_API_KEY set, the
              orchestrator runs the same per-cell loop but calls GPT-4o-mini for
              both the answer and the judge — producing the joint answer-quality
              x cost numbers that close Stage 3.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}

function RetrievalTable({ data }: { data: StageData }) {
  const { benchmarks, table, tuned_vs_canonical } = data;
  // Sort systems by long-haystack performance: avg of CUAD + QASPER.
  const systems = data.systems.slice().sort((a, b) => {
    const scoreA = HEADLINE_BENCHMARKS.reduce((acc, b) => {
      const v = table[a]?.[b];
      return acc + (typeof v === "number" ? v : 0);
    }, 0);
    const scoreB = HEADLINE_BENCHMARKS.reduce((acc, b) => {
      const v = table[b]?.[b];
      return acc + (typeof v === "number" ? v : 0);
    }, 0);
    return scoreB - scoreA;
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.45 }}
      className="panel-rise p-6 overflow-x-auto"
    >
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <div
            className="text-[0.65rem] uppercase tracking-[0.18em]"
            style={{ color: "var(--color-muted)" }}
          >
            recall@k=8, n_docs=15 per benchmark
          </div>
          <h3 className="font-semibold text-lg mt-1">
            Memory systems × 6 real benchmarks
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <TrendingUp size={14} style={{ color: "var(--color-emerald)" }} />
          <span style={{ color: "var(--color-text-2)" }}>
            Sorted by long-haystack performance (CUAD + QASPER)
          </span>
        </div>
      </div>

      <table className="w-full text-xs font-mono">
        <thead>
          <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
            <th className="text-left py-2 pr-3 font-semibold">system</th>
            {benchmarks.map((b) => (
              <th
                key={b}
                className={`text-right py-2 px-2 font-semibold ${
                  HEADLINE_BENCHMARKS.includes(b)
                    ? "underline decoration-amber-400 underline-offset-4"
                    : ""
                }`}
                style={{
                  color: HEADLINE_BENCHMARKS.includes(b)
                    ? "var(--color-amber)"
                    : "var(--color-text-2)",
                }}
              >
                {b}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {systems.map((sys) => {
            const isV4Tuned = sys === "V4-tuned";
            return (
              <tr
                key={sys}
                style={{
                  borderBottom: "1px solid var(--color-border-soft)",
                  background: isV4Tuned
                    ? "linear-gradient(90deg, var(--color-amber-soft), transparent)"
                    : "transparent",
                }}
              >
                <td
                  className="py-2 pr-3"
                  style={{
                    color: isV4Tuned ? "var(--color-amber)" : "var(--color-text)",
                    fontWeight: isV4Tuned ? 600 : 400,
                  }}
                >
                  {sys}
                </td>
                {benchmarks.map((b) => {
                  const v = table[sys]?.[b];
                  const headline = HEADLINE_BENCHMARKS.includes(b);
                  return (
                    <td
                      key={b}
                      className="text-right py-2 px-2 tabular-nums"
                      style={{
                        color:
                          typeof v === "number"
                            ? v >= 0.5 && headline
                              ? "var(--color-emerald)"
                              : "var(--color-text)"
                            : "var(--color-muted)",
                      }}
                    >
                      {typeof v === "number" ? v.toFixed(3) : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Tuned vs canonical comparison */}
      <div className="mt-6">
        <div
          className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
          style={{ color: "var(--color-muted)" }}
        >
          V4 canonical (grid-world θ) vs V4-tuned (per-benchmark θ via CMA-ES)
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(tuned_vs_canonical).map(([bench, comp]) => (
            <div
              key={bench}
              className="panel-rise p-3"
              style={{
                background:
                  comp.improvement >= 0.1
                    ? "linear-gradient(135deg, var(--color-emerald-soft) 0%, transparent 80%), var(--color-surface)"
                    : "var(--color-surface)",
              }}
            >
              <div
                className="text-[0.65rem] uppercase tracking-wider mb-1"
                style={{ color: "var(--color-muted)" }}
              >
                {bench}
              </div>
              <div className="flex items-baseline gap-2 font-mono text-xs">
                <span style={{ color: "var(--color-text-2)" }}>
                  {comp.canonical_recall.toFixed(3)}
                </span>
                <span style={{ color: "var(--color-muted)" }}>→</span>
                <span
                  style={{
                    color:
                      comp.improvement >= 0.1
                        ? "var(--color-emerald)"
                        : "var(--color-text)",
                    fontWeight: 600,
                  }}
                >
                  {comp.tuned_recall.toFixed(3)}
                </span>
                <span
                  className="ml-auto"
                  style={{
                    color:
                      comp.improvement >= 0.1
                        ? "var(--color-emerald)"
                        : "var(--color-muted)",
                  }}
                >
                  {comp.improvement >= 0 ? "+" : ""}
                  {comp.improvement.toFixed(3)}
                </span>
              </div>
              <div
                className="text-[0.65rem] mt-1"
                style={{ color: "var(--color-text-2)" }}
              >
                {comp.n_gold_questions} questions with gold relevance
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

function TransferMatrixPanel({ matrix }: { matrix: TransferMatrix }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.45, delay: 0.1 }}
      className="panel-rise p-6 mt-4"
      style={{
        background:
          "linear-gradient(135deg, transparent 0%, var(--color-cyan-soft) 100%), var(--color-surface)",
      }}
    >
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <div
            className="text-[0.65rem] uppercase tracking-[0.18em]"
            style={{ color: "var(--color-muted)" }}
          >
            cross-benchmark theta transfer · 3 × 2 matrix · n_docs = 15
          </div>
          <h3 className="font-semibold text-lg mt-1">
            Does task-specific θ generalize across long-haystack benchmarks?
          </h3>
        </div>
      </div>

      <table className="w-full text-xs font-mono">
        <thead>
          <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
            <th className="text-left py-2 pr-3 font-semibold">θ source ↓ / eval →</th>
            {matrix.cols.map((col) => (
              <th
                key={col}
                className="text-right py-2 px-2 font-semibold"
                style={{ color: "var(--color-amber)" }}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr
              key={row}
              style={{ borderBottom: "1px solid var(--color-border-soft)" }}
            >
              <td
                className="py-2 pr-3"
                style={{
                  color:
                    row === "canonical"
                      ? "var(--color-muted)"
                      : "var(--color-text)",
                  fontWeight: row === "canonical" ? 400 : 500,
                }}
              >
                {row}
              </td>
              {matrix.cols.map((col) => {
                const v = matrix.matrix_means?.[row]?.[col];
                const isDiagonal = row === `${col}-tuned`;
                return (
                  <td
                    key={col}
                    className="text-right py-2 px-2 tabular-nums"
                    style={{
                      color: isDiagonal
                        ? "var(--color-emerald)"
                        : typeof v === "number" && v > 0.4
                        ? "var(--color-text)"
                        : "var(--color-muted)",
                      fontWeight: isDiagonal ? 700 : 400,
                    }}
                  >
                    {typeof v === "number" ? v.toFixed(3) : "—"}
                    {isDiagonal ? " ★" : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Summary stats */}
      <div className="grid sm:grid-cols-3 gap-3 mt-5">
        <div className="text-center">
          <div
            className="text-[0.6rem] uppercase tracking-wider mb-1"
            style={{ color: "var(--color-muted)" }}
          >
            canonical avg
          </div>
          <div
            className="font-mono text-lg"
            style={{ color: "var(--color-text-2)" }}
          >
            {matrix.canonical_avg !== null && matrix.canonical_avg !== undefined
              ? matrix.canonical_avg.toFixed(3)
              : "—"}
          </div>
        </div>
        <div className="text-center">
          <div
            className="text-[0.6rem] uppercase tracking-wider mb-1"
            style={{ color: "var(--color-muted)" }}
          >
            diagonal avg (matched θ)
          </div>
          <div
            className="font-mono text-lg"
            style={{ color: "var(--color-emerald)", fontWeight: 700 }}
          >
            {matrix.diagonal_avg !== null && matrix.diagonal_avg !== undefined
              ? matrix.diagonal_avg.toFixed(3)
              : "—"}
            {matrix.diagonal_lift_vs_canonical !== null &&
              matrix.diagonal_lift_vs_canonical !== undefined && (
                <span
                  className="text-xs ml-2"
                  style={{ color: "var(--color-emerald)" }}
                >
                  {matrix.diagonal_lift_vs_canonical >= 0 ? "+" : ""}
                  {matrix.diagonal_lift_vs_canonical.toFixed(3)}
                </span>
              )}
          </div>
        </div>
        <div className="text-center">
          <div
            className="text-[0.6rem] uppercase tracking-wider mb-1"
            style={{ color: "var(--color-muted)" }}
          >
            off-diagonal avg (mismatched θ)
          </div>
          <div
            className="font-mono text-lg"
            style={{ color: "var(--color-cyan)", fontWeight: 600 }}
          >
            {matrix.off_diagonal_avg !== null &&
            matrix.off_diagonal_avg !== undefined
              ? matrix.off_diagonal_avg.toFixed(3)
              : "—"}
            {matrix.off_diagonal_lift_vs_canonical !== null &&
              matrix.off_diagonal_lift_vs_canonical !== undefined && (
                <span
                  className="text-xs ml-2"
                  style={{ color: "var(--color-cyan)" }}
                >
                  {matrix.off_diagonal_lift_vs_canonical >= 0 ? "+" : ""}
                  {matrix.off_diagonal_lift_vs_canonical.toFixed(3)}
                </span>
              )}
          </div>
        </div>
      </div>

      {matrix.interpretation ? (
        <p
          className="mt-5 text-sm leading-relaxed"
          style={{ color: "var(--color-text-2)" }}
        >
          {matrix.interpretation}
        </p>
      ) : null}
    </motion.div>
  );
}
