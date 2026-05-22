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

type Phase4Cell = {
  mean_judge: number | null;
  mean_recall: number | null;
  judge_95ci: [number, number] | null;
  n_questions_pooled: number | null;
};
type PairedTest = {
  n_pairs: number;
  t: number | null;
  p_two_sided: number | null;
  mean_diff: number | null;
  lift: number | null;
  tuned_mean: number | null;
  canonical_mean: number | null;
  // Phase 1.7 additions:
  p_holm_t?: number | null;
  significant_holm_t?: boolean;
  p_holm_wilcoxon?: number | null;
  significant_holm_wilcoxon?: boolean;
  wilcoxon?: {
    W: number;
    p_two_sided: number;
    significant: boolean;
  } | null;
  cohens_d?: {
    d: number;
    magnitude: string;
  } | null;
  lift_cluster_ci?: {
    point_estimate: number;
    ci_lower: number;
    ci_upper: number;
  } | null;
};
type Phase4Data = {
  benchmarks: string[];
  configs: string[];
  seeds: number[];
  summary: { [benchmark: string]: { [config: string]: Phase4Cell } };
  paired_ttests: { [benchmark: string]: PairedTest };
  total_cost_usd: number;
};

const HEADLINE_BENCHMARKS = ["qasper", "cuad"];

export function Stage3() {
  const [data, setData] = useState<StageData | null>(null);
  const [phase4, setPhase4] = useState<Phase4Data | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/stage3_retrieval.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
    fetch(`${import.meta.env.BASE_URL}data/stage3_phase4.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setPhase4)
      .catch(() => setPhase4(null));
  }, []);

  return (
    <Section id="stage3" eyebrow="Stage 3 — Real Benchmarks">
      <SectionHeader
        title={
          <>
            Six published benchmarks.{" "}
            <span style={{ color: "var(--color-amber)" }}>
              Per-task &theta; lifts V4 on CUAD with Holm-corrected p&nbsp;&lt;&nbsp;0.01.
            </span>
          </>
        }
        lede={
          <>
            We migrated the evaluation off hand-authored documents onto six real
            long-context QA benchmarks (HotpotQA, QASPER, CUAD, NarrativeQA,
            FinanceBench, LongMemEval). Per-benchmark CMA-ES tuning of V4's 10D
            &theta; lifts CUAD answer quality by +0.067 judge points
            (<strong>p<sub>Holm</sub> = 0.0033, d = +0.191</strong>) and gives
            comparable lift on QASPER (not Holm-significant). Phase 1.7
            added BM25 and tuned-AttentionMemory baselines under identical
            CMA-ES budgets — on CUAD, AttentionMemory-tuned actually beats
            V4-tuned by +0.053, narrowing the claim from{" "}
            <em>"this architecture wins"</em> to{" "}
            <em>"tuning wins; architecture is benchmark-dependent."</em>
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
            {phase4 ? <Phase4Panel data={phase4} /> : null}
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
              per cell. <strong>Phase 4 + 1.7 complete</strong>: total spend
              ~$5 across 11,000 LLM-judged questions (3 seeds × 100 q × 5
              configs × 6 benchmarks, plus held-out test split and k-sweep
              multi-seed). The Pareto frontier is in §5.6 of the chapter.
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-3 mt-4">
            <Pill tone="cyan">6 benchmarks</Pill>
            <Pill tone="amber">5 memory configs</Pill>
            <Pill tone="violet">3 seeds · Holm-corrected</Pill>
          </div>
        </div>

        <div className="space-y-4">
          <div className="panel-rise p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={16} style={{ color: "var(--color-amber)" }} />
              <h3 className="font-semibold text-base">Reproduction recipe</h3>
            </div>
            <CodeBlock
              label="powershell"
              code={`# 1. Tune V4 theta per benchmark (no LLM, ~3 min)
python -m tuning.tune_v4_per_benchmark --benchmarks all

# 2. Tune AttentionMemory tau with identical CMA-ES budget
python -m tuning.tune_attention_per_benchmark --benchmarks qasper cuad

# 3. Held-out V4 tuning (TRAIN/TEST split)
python -m tuning.tune_v4_per_benchmark --benchmarks qasper cuad \\
  --held-out-split --split-seed 42 --out-suffix heldout

# 4. Multi-seed Phase 4 eval (requires OPENAI_API_KEY)
python scripts/run_stage3_full.py --mode full --benchmarks all \\
  --configs v4-canonical v4-tuned flat-50 bm25 attention-tuned \\
  --n-questions 100 --seeds 42 7 100

# 5. Aggregate with Holm-Bonferroni + cluster-bootstrap CI
python scripts/aggregate_stage3_results.py --seeds 42 7 100`}
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
              The narrowed claim, Phase 1.7.
            </h3>
            <p
              className="text-sm leading-relaxed"
              style={{ color: "var(--color-text-2)" }}
            >
              Original claim: <em>"V4-tuned dominates the long-haystack
              regime."</em> Adversarial review surfaced 17 critique points;
              the held-out test (B), BM25 baseline (C), tuned-AttentionMemory
              baseline (D), and Holm correction (A) narrow this to:{" "}
              <strong>
                task-tuned parameterized memory beats canonical V4 with
                Holm-corrected significance on CUAD (p&lt;0.01); which tuned
                architecture wins among V4 / BM25 / AttentionMemory is
                benchmark-dependent.
              </strong>{" "}
              Full Limitations bundle in chapter §6.6.
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
  const systemScore = (sys: string): number =>
    HEADLINE_BENCHMARKS.reduce((acc, bench) => {
      const v = table[sys]?.[bench];
      return acc + (typeof v === "number" ? v : 0);
    }, 0);
  const systems = data.systems.slice().sort((a, b) => systemScore(b) - systemScore(a));

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
            cross-benchmark theta transfer · 5 × 2 matrix · n_docs = 15
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
                // Diagonal if theta-source benchmark matches eval benchmark.
                // Phase 1.7: matches both "<bench>-tuned" and "<bench>-heldout".
                const isDiagonal =
                  row === `${col}-tuned` || row === `${col}-heldout`;
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

function Phase4Panel({ data }: { data: Phase4Data }) {
  const sigMarker = (p: number | null | undefined): string =>
    p === null || p === undefined
      ? ""
      : p < 0.001
      ? "***"
      : p < 0.01
      ? "**"
      : p < 0.05
      ? "*"
      : "";
  // Winners-by-benchmark restricted to configs with full multi-seed coverage.
  // Single-seed configs (BM25/AttentionMemory on most benchmarks) are still
  // shown in the table but not bolded as "winners" — see § note below.
  const winnersByBench: Record<string, string> = {};
  data.benchmarks.forEach((b) => {
    let best: { cfg: string; judge: number; n: number } | null = null;
    data.configs.forEach((c) => {
      const cell = data.summary[b]?.[c];
      const j = cell?.mean_judge;
      const n = cell?.n_questions_pooled ?? 0;
      // Require >=200 pooled questions for "winner" eligibility — i.e. true
      // multi-seed (>=2 seeds × 100 q). Single-seed cells excluded from
      // winner highlight because their CI is too wide for a fair comparison.
      if (typeof j === "number" && n >= 200 && (best === null || j > best.judge)) {
        best = { cfg: c, judge: j, n };
      }
    });
    if (best !== null) {
      const w: { cfg: string; judge: number; n: number } = best;
      winnersByBench[b] = w.cfg;
    }
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.45, delay: 0.2 }}
      className="panel-rise p-6 mt-4"
      style={{
        background:
          "linear-gradient(135deg, transparent 0%, var(--color-emerald-soft) 100%), var(--color-surface)",
      }}
    >
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <div
            className="text-[0.65rem] uppercase tracking-[0.18em]"
            style={{ color: "var(--color-muted)" }}
          >
            Phase 4 + 1.7 · LLM answer quality · gpt-4o-mini ·{" "}
            {data.seeds.length} seed(s) × 100 q · Holm-Bonferroni corrected
          </div>
          <h3 className="font-semibold text-lg mt-1">
            LLM-Judge Score per Memory Configuration
          </h3>
        </div>
        <div className="text-xs" style={{ color: "var(--color-text-2)" }}>
          Total API spend:{" "}
          <span style={{ color: "var(--color-emerald)", fontWeight: 600 }}>
            ${data.total_cost_usd.toFixed(4)}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
              <th className="text-left py-2 pr-3 font-semibold">benchmark</th>
              {data.configs.map((c) => (
                <th
                  key={c}
                  className="text-right py-2 px-2 font-semibold"
                  style={{
                    color:
                      c === "v4-tuned"
                        ? "var(--color-amber)"
                        : "var(--color-text-2)",
                  }}
                >
                  {c}
                </th>
              ))}
              <th
                className="text-right py-2 px-2 font-semibold"
                style={{ color: "var(--color-amber)" }}
              >
                V4t − V4c · lift · p_Holm · d
              </th>
            </tr>
          </thead>
          <tbody>
            {data.benchmarks.map((b) => {
              const t = data.paired_ttests?.[b];
              return (
                <tr
                  key={b}
                  style={{ borderBottom: "1px solid var(--color-border-soft)" }}
                >
                  <td className="py-2 pr-3 font-medium">{b}</td>
                  {data.configs.map((c) => {
                    const cell = data.summary[b]?.[c];
                    const j = cell?.mean_judge;
                    const ci = cell?.judge_95ci;
                    const n = cell?.n_questions_pooled ?? 0;
                    const isWinner = winnersByBench[b] === c;
                    const isSingleSeed = n > 0 && n < 200;
                    return (
                      <td
                        key={c}
                        className="text-right py-2 px-2 tabular-nums"
                        style={{
                          color: isWinner
                            ? "var(--color-emerald)"
                            : isSingleSeed
                            ? "var(--color-text-2)"
                            : "var(--color-text)",
                          fontWeight: isWinner ? 700 : 400,
                          fontStyle: isSingleSeed ? "italic" : "normal",
                        }}
                      >
                        {typeof j === "number" ? j.toFixed(3) : "—"}
                        {ci ? (
                          <div
                            className="text-[0.6rem]"
                            style={{
                              color: "var(--color-muted)",
                              fontWeight: 400,
                              fontStyle: "normal",
                            }}
                          >
                            [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
                            {isSingleSeed ? " · 1 seed" : ""}
                          </div>
                        ) : null}
                      </td>
                    );
                  })}
                  <td className="text-right py-2 px-2 tabular-nums">
                    {t && typeof t.lift === "number" ? (
                      <>
                        <span
                          style={{
                            color:
                              t.significant_holm_t
                                ? "var(--color-emerald)"
                                : t.lift > 0
                                ? "var(--color-text)"
                                : t.lift < 0
                                ? "var(--color-rose)"
                                : "var(--color-muted)",
                            fontWeight: t.significant_holm_t ? 700 : 600,
                          }}
                        >
                          {t.lift >= 0 ? "+" : ""}
                          {t.lift.toFixed(3)} {sigMarker(t.p_holm_t)}
                        </span>
                        <div
                          className="text-[0.6rem]"
                          style={{ color: "var(--color-muted)" }}
                        >
                          p_Holm=
                          {typeof t.p_holm_t === "number"
                            ? t.p_holm_t.toFixed(3)
                            : "—"}
                          {t.cohens_d ? (
                            <>
                              {" · d="}
                              {t.cohens_d.d >= 0 ? "+" : ""}
                              {t.cohens_d.d.toFixed(2)}
                            </>
                          ) : null}
                        </div>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-xs" style={{ color: "var(--color-text-2)" }}>
        Bold-green = best mean judge per benchmark (restricted to configs with
        ≥2 seeds; <em>italicized</em> cells are single-seed and excluded from
        the winner highlight). 95% CIs are cluster-bootstrap (resampled by
        per-document cluster) over pooled per-question judge. Significance
        markers <code>*</code> p_Holm&lt;0.05, <code>**</code> p_Holm&lt;0.01,{" "}
        <code>***</code> p_Holm&lt;0.001 — apply Holm-Bonferroni step-down
        correction across the 5 benchmarks tested. The right column reports
        the V4-tuned − V4-canonical paired contrast with Cohen's d effect size.
      </p>
    </motion.div>
  );
}
