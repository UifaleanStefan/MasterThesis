/**
 * FinanceBenchCorpus — Stage 3 Phase 2 end-to-end corpus QA results.
 *
 * Four interactive panels: judge table, theta contrast (canonical /
 * per-doc tuned / corpus-tuned), cross-document bleed scatter, and
 * question-category heatmap. Shared config filter; clicking any cell
 * or scatter dot opens a drill-down modal with a worked example.
 */

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { TrendingUp, X, Filter } from "lucide-react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { Pill } from "../components/shared/Pill";
import { ChartFrame } from "../components/shared/ChartFrame";
import { cn } from "../lib/format";
import { CrossDocScatter } from "../components/viz/CrossDocScatter";
import { QuestionTypeHeatmap } from "../components/viz/QuestionTypeHeatmap";
import { CalibrationTrajectory } from "../components/viz/CalibrationTrajectory";
import type {
  FinanceBenchConfig,
  FinanceBenchCorpusData,
  FinanceBenchMode,
  DrilldownExample,
  ScatterPoint,
} from "../data/financeBenchTypes";

export function FinanceBenchCorpus() {
  const [data, setData] = useState<FinanceBenchCorpusData | null>(null);
  const [visibleConfigs, setVisibleConfigs] = useState<Set<string>>(new Set());
  const [hoveredConfig, setHoveredConfig] = useState<string | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<FinanceBenchMode>("online");
  const [drilldown, setDrilldown] = useState<DrilldownExample | null>(null);
  const [drilldownContext, setDrilldownContext] = useState<{ cfg?: string; mode?: string } | null>(
    null,
  );
  const [sortKey, setSortKey] = useState<"online" | "batch" | "delta">("online");

  useEffect(() => {
    fetch("/data/stage3_finbench_corpus.json")
      .then((r) => r.json())
      .then((d: FinanceBenchCorpusData) => {
        setData(d);
        setVisibleConfigs(new Set(d.configs.map((c) => c.name)));
      })
      .catch(() => setData(null));
  }, []);

  const sortedConfigs = useMemo(() => {
    if (!data) return [];
    const arr = data.configs.slice();
    arr.sort((a, b) => {
      if (sortKey === "delta") {
        const ad =
          data.judge_table[a.name].online.mean_judge - data.judge_table[a.name].batch.mean_judge;
        const bd =
          data.judge_table[b.name].online.mean_judge - data.judge_table[b.name].batch.mean_judge;
        return ad - bd;
      }
      const av = data.judge_table[a.name][sortKey].mean_judge;
      const bv = data.judge_table[b.name][sortKey].mean_judge;
      return bv - av;
    });
    return arr;
  }, [data, sortKey]);

  if (!data) {
    return (
      <Section id="finbench-corpus" eyebrow="Stage 3 Phase 2 · FinanceBench end-to-end">
        <div style={{ color: "var(--color-muted)" }}>loading FinanceBench corpus data…</div>
      </Section>
    );
  }

  function toggleConfig(name: string) {
    setVisibleConfigs((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function openDrilldown(example: DrilldownExample, cfg?: string, mode?: string) {
    setDrilldown(example);
    setDrilldownContext({ cfg, mode });
  }

  return (
    <Section
      id="finbench-corpus"
      eyebrow="Stage 3 Phase 2 · FinanceBench end-to-end"
      variant="raised"
    >
      <SectionHeader
        title={
          <>
            One memory, <span style={{ color: "var(--color-cyan)" }}>150 questions</span>,{" "}
            <span style={{ color: "var(--color-amber)" }}>12 cells</span>
          </>
        }
        lede={
          <>
            A single V4ₜ memory instance ingests all 150 FinanceBench documents in sequence,
            then answers every question end-to-end with gpt-4o-mini. Six configurations × two
            query modes × 150 questions = 1,800 Claude-judged predictions. The headline:
            <strong className="ml-1" style={{ color: "var(--color-text)" }}>
              corpus-tuned θ amplifies the four-shift finding from §6.5
            </strong>{" "}
            and surfaces a fifth — <code>w_graph</code> activates for the first time.
          </>
        }
      />

      <div className="flex flex-wrap items-center gap-3 mb-8">
        <Pill tone="cyan">n=150 / cell</Pill>
        <Pill tone="amber">k=8 retrieval</Pill>
        <Pill tone="emerald">manual Claude judge</Pill>
        <Pill tone="muted">gpt-4o-mini answerer</Pill>
        <Pill tone="muted">total ≈ $1.50</Pill>
      </div>

      <ConfigFilter
        configs={data.configs}
        visible={visibleConfigs}
        onToggle={toggleConfig}
        onSelectAll={() => setVisibleConfigs(new Set(data.configs.map((c) => c.name)))}
        onSelectV4={() =>
          setVisibleConfigs(new Set(data.configs.filter((c) => c.family.startsWith("v4")).map((c) => c.name)))
        }
        onSelectWinners={() =>
          setVisibleConfigs(
            new Set([
              "v4t-corpus-tuned",
              "attention-corpus-tuned",
              "rag-corpus",
              "dump-all",
            ]),
          )
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        <ChartFrame
          title="Judge table"
          subtitle="Mean Claude-judge score · 95% bootstrap CI"
          controls={
            <select
              value={sortKey}
              onChange={(e) =>
                setSortKey(e.target.value as "online" | "batch" | "delta")
              }
              className="text-xs px-2 py-1 rounded border border-[var(--color-border)] bg-transparent"
              style={{ color: "var(--color-text-2)" }}
            >
              <option value="online">sort: online ↓</option>
              <option value="batch">sort: batch ↓</option>
              <option value="delta">sort: Δ (on−ba) ↑</option>
            </select>
          }
        >
          <JudgeTable
            configs={sortedConfigs}
            data={data}
            visibleConfigs={visibleConfigs}
            hoveredConfig={hoveredConfig}
            onHoverConfig={setHoveredConfig}
            onCellClick={(cfgName, mode) => {
              const ex =
                data.drilldown_examples.always_hard.find((e) => e.predictions[`${cfgName}__${mode}`]) ??
                data.drilldown_examples.always_easy.find((e) => e.predictions[`${cfgName}__${mode}`]) ??
                data.drilldown_examples.corpus_tuned_wins_vs_canonical[0];
              if (ex) openDrilldown(ex, cfgName, mode);
            }}
          />
        </ChartFrame>

        <ChartFrame
          title="θ contrast"
          subtitle="Five parameters · three regimes"
        >
          <ThetaContrast theta_contrast={data.theta_contrast} />
        </ChartFrame>
      </div>

      <div className="mt-6">
        <ChartFrame
          title="Cross-document bleed vs judge score"
          subtitle="Each dot is one (config, mode) cell. X = fraction of retrieved steps from own doc; Y = mean judge."
        >
          <CrossDocScatter
            points={data.cross_doc_scatter}
            visibleConfigs={visibleConfigs}
            highlight={hoveredConfig ? { config: hoveredConfig, mode: "online" } : null}
            onPointClick={(p: ScatterPoint) => {
              const winners = data.drilldown_examples.corpus_tuned_wins_vs_canonical;
              const losses = data.drilldown_examples.corpus_tuned_regressions_vs_canonical;
              const hard = data.drilldown_examples.always_hard;
              const pool = p.mean_judge < 0.2 ? hard : p.mean_judge > 0.7 ? winners : losses;
              const pick = pool.find((e) => e.predictions[`${p.config}__${p.mode}`]) ?? pool[0];
              if (pick) openDrilldown(pick, p.config, p.mode);
            }}
          />
        </ChartFrame>
      </div>

      <div className="mt-6">
        <ChartFrame
          title="Question category × config"
          subtitle="Where the LLM ceiling (multi-formula calc) bites vs. where retrieval matters."
        >
          <QuestionTypeHeatmap
            configs={data.configs}
            question_types={data.question_types}
            mode={heatmapMode}
            onModeChange={setHeatmapMode}
            visibleConfigs={visibleConfigs}
            onCellClick={(cfgName, mode, category) => {
              const pool = [
                ...data.drilldown_examples.always_hard,
                ...data.drilldown_examples.corpus_tuned_wins_vs_canonical,
                ...data.drilldown_examples.corpus_tuned_regressions_vs_canonical,
                ...data.drilldown_examples.always_easy,
              ];
              const pick = pool.find(
                (e) => e.category === category && e.predictions[`${cfgName}__${mode}`],
              );
              if (pick) openDrilldown(pick, cfgName, mode);
            }}
          />
        </ChartFrame>
      </div>

      {data.calibration_data ? (
        <div className="mt-6">
          <ChartFrame
            title="Protocol B calibration trajectory"
            subtitle="Per-decile mean across the 1500 calibration questions (10 configs × 150 questions/decile). Selective + tuned configs hold; canonical θ / windowed / TF-IDF / dump-all collapse as the corpus dilutes memory."
          >
            <CalibrationTrajectory
              calibration={data.calibration_data}
              configs={data.configs}
              visibleConfigs={visibleConfigs}
            />
          </ChartFrame>
        </div>
      ) : null}

      <div
        className="mt-6 text-[0.7rem] font-mono"
        style={{ color: "var(--color-muted)" }}
      >
        manifest: git {data._manifest.git_sha} · {data._manifest.timestamp_utc} · model{" "}
        {data._manifest.model} · k={data._manifest.k} · n={data._manifest.n_questions}/cell ·{" "}
        manual Claude judge per {data._manifest.judge_protocol}
      </div>

      <AnimatePresence>
        {drilldown ? (
          <DrillDownModal
            example={drilldown}
            triggerCfg={drilldownContext?.cfg}
            triggerMode={drilldownContext?.mode}
            configs={data.configs}
            onClose={() => {
              setDrilldown(null);
              setDrilldownContext(null);
            }}
          />
        ) : null}
      </AnimatePresence>
    </Section>
  );
}

/* ============================================================================
 * ConfigFilter
 * ========================================================================= */

interface ConfigFilterProps {
  configs: FinanceBenchConfig[];
  visible: Set<string>;
  onToggle: (name: string) => void;
  onSelectAll: () => void;
  onSelectV4: () => void;
  onSelectWinners: () => void;
}

function ConfigFilter({
  configs,
  visible,
  onToggle,
  onSelectAll,
  onSelectV4,
  onSelectWinners,
}: ConfigFilterProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 panel-rise px-4 py-3">
      <Filter size={14} style={{ color: "var(--color-muted)" }} />
      <span
        className="text-[0.65rem] uppercase tracking-[0.18em] mr-2"
        style={{ color: "var(--color-muted)" }}
      >
        configs
      </span>
      {configs.map((c) => {
        const isOn = visible.has(c.name);
        return (
          <button
            key={c.name}
            onClick={() => onToggle(c.name)}
            className={cn(
              "px-3 py-1 rounded-full text-xs border transition",
              isOn ? "" : "opacity-45",
            )}
            style={{
              borderColor: isOn ? c.color : "var(--color-border)",
              color: isOn ? c.color : "var(--color-muted)",
              background: isOn ? `${c.color}1a` : "transparent",
              transition: "opacity 180ms cubic-bezier(0.2, 0, 0, 1), background 180ms cubic-bezier(0.2, 0, 0, 1)",
            }}
          >
            <span
              className="inline-block w-2 h-2 rounded-full mr-2 align-middle"
              style={{ background: c.color }}
            />
            {c.label}
          </button>
        );
      })}
      <div className="ml-auto flex items-center gap-2 text-xs">
        <button
          className="px-2 py-1 rounded border border-[var(--color-border)] hover:border-[var(--color-cyan)]"
          style={{ color: "var(--color-text-2)" }}
          onClick={onSelectAll}
        >
          all
        </button>
        <button
          className="px-2 py-1 rounded border border-[var(--color-border)] hover:border-[var(--color-cyan)]"
          style={{ color: "var(--color-text-2)" }}
          onClick={onSelectV4}
        >
          V4 only
        </button>
        <button
          className="px-2 py-1 rounded border border-[var(--color-border)] hover:border-[var(--color-cyan)]"
          style={{ color: "var(--color-text-2)" }}
          onClick={onSelectWinners}
        >
          winners
        </button>
      </div>
    </div>
  );
}

/* ============================================================================
 * JudgeTable
 * ========================================================================= */

interface JudgeTableProps {
  configs: FinanceBenchConfig[];
  data: FinanceBenchCorpusData;
  visibleConfigs: Set<string>;
  hoveredConfig: string | null;
  onHoverConfig: (cfg: string | null) => void;
  onCellClick: (cfg: string, mode: string) => void;
}

function JudgeTable({
  configs,
  data,
  visibleConfigs,
  hoveredConfig,
  onHoverConfig,
  onCellClick,
}: JudgeTableProps) {
  const shown = configs.filter((c) => visibleConfigs.has(c.name));

  // Best cell per column for highlighting — only consider populated cells (n > 0)
  const populated = (mode: "online" | "batch") =>
    shown.map((c) => data.judge_table[c.name][mode]).filter((cell) => cell.n > 0);
  const bestOnline = populated("online").length
    ? Math.max(...populated("online").map((c) => c.mean_judge))
    : -1;
  const bestBatch = populated("batch").length
    ? Math.max(...populated("batch").map((c) => c.mean_judge))
    : -1;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
            <th className="text-left py-2 pr-3 font-semibold" style={{ color: "var(--color-muted)" }}>
              config
            </th>
            <th className="text-right py-2 px-2 font-semibold" style={{ color: "var(--color-cyan)" }}>
              online judge
            </th>
            <th className="text-right py-2 px-2 font-semibold" style={{ color: "var(--color-text-2)" }}>
              online R
            </th>
            <th className="text-right py-2 px-2 font-semibold" style={{ color: "var(--color-cyan)" }}>
              batch judge
            </th>
            <th className="text-right py-2 px-2 font-semibold" style={{ color: "var(--color-text-2)" }}>
              batch R
            </th>
            <th className="text-right py-2 pl-2 font-semibold" style={{ color: "var(--color-amber)" }}>
              Δ
            </th>
          </tr>
        </thead>
        <tbody>
          {shown.map((c) => {
            const onl = data.judge_table[c.name].online;
            const bat = data.judge_table[c.name].batch;
            const haveOnl = onl.n > 0;
            const haveBat = bat.n > 0;
            const delta = haveOnl && haveBat ? onl.mean_judge - bat.mean_judge : null;
            const isHovered = hoveredConfig === c.name;

            return (
              <tr
                key={c.name}
                onMouseEnter={() => onHoverConfig(c.name)}
                onMouseLeave={() => onHoverConfig(null)}
                style={{
                  borderBottom: "1px solid var(--color-border-soft)",
                  background: isHovered
                    ? `linear-gradient(90deg, ${c.color}1f, transparent)`
                    : "transparent",
                  transition: "background 180ms cubic-bezier(0.2, 0, 0, 1)",
                }}
              >
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-2">
                    <span
                      className="inline-block w-1.5 h-4 rounded-sm"
                      style={{ background: c.color }}
                    />
                    <span style={{ color: "var(--color-text)" }}>{c.label}</span>
                  </div>
                </td>
                {haveOnl ? (
                  <Cell
                    cell={onl}
                    best={onl.mean_judge >= bestOnline - 0.005}
                    onClick={() => onCellClick(c.name, "online")}
                  />
                ) : (
                  <EmptyCell />
                )}
                <td
                  className="text-right py-2 px-2 tabular-nums"
                  style={{ color: "var(--color-text-2)" }}
                >
                  {haveOnl ? onl.mean_recall.toFixed(2) : "—"}
                </td>
                {haveBat ? (
                  <Cell
                    cell={bat}
                    best={bat.mean_judge >= bestBatch - 0.005}
                    onClick={() => onCellClick(c.name, "batch")}
                  />
                ) : (
                  <EmptyCell />
                )}
                <td
                  className="text-right py-2 px-2 tabular-nums"
                  style={{ color: "var(--color-text-2)" }}
                >
                  {haveBat ? bat.mean_recall.toFixed(2) : "—"}
                </td>
                <td
                  className="text-right py-2 pl-2 tabular-nums"
                  style={{
                    color:
                      delta === null
                        ? "var(--color-muted)"
                        : Math.abs(delta) < 0.05
                          ? "var(--color-text-2)"
                          : delta > 0
                            ? "var(--color-amber)"
                            : "var(--color-emerald)",
                  }}
                >
                  {delta === null ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div
        className="mt-3 text-[0.65rem] flex items-center gap-2"
        style={{ color: "var(--color-muted)" }}
      >
        <TrendingUp size={11} />
        <span>
          highlighted cell = best in its column. Click any judge cell for a worked example.
        </span>
      </div>
    </div>
  );
}

interface CellProps {
  cell: { mean_judge: number; ci_lower: number; ci_upper: number };
  best: boolean;
  onClick: () => void;
}

function Cell({ cell, best, onClick }: CellProps) {
  return (
    <td
      onClick={onClick}
      className="text-right py-2 px-2 tabular-nums cursor-pointer"
      style={{
        color: best ? "var(--color-amber)" : "var(--color-text)",
        fontWeight: best ? 600 : 400,
        background: best ? "var(--color-amber-soft)" : "transparent",
        transition: "background 180ms cubic-bezier(0.2, 0, 0, 1)",
      }}
      title={`95% CI [${cell.ci_lower.toFixed(2)}, ${cell.ci_upper.toFixed(2)}]`}
    >
      {cell.mean_judge.toFixed(3)}
      <span className="ml-1 text-[0.55rem]" style={{ color: "var(--color-muted)" }}>
        ±{((cell.ci_upper - cell.ci_lower) / 2).toFixed(2)}
      </span>
    </td>
  );
}

function EmptyCell() {
  return (
    <td
      className="text-right py-2 px-2 tabular-nums"
      style={{ color: "var(--color-muted)" }}
      title="No data — this config was evaluated calibration-only (Phase 1.9 extension; no Protocol A run)."
    >
      —
    </td>
  );
}

/* ============================================================================
 * ThetaContrast
 * ========================================================================= */

interface ThetaContrastProps {
  theta_contrast: FinanceBenchCorpusData["theta_contrast"];
}

function ThetaContrast({ theta_contrast }: ThetaContrastProps) {
  const { params, rows, annotations } = theta_contrast;

  // Compute per-column max for height normalization
  const maxByCol = params.map((_, ci) =>
    Math.max(...rows.map((r) => Math.abs(r.values[ci]))),
  );

  const PARAM_COLOR: Record<string, string> = {
    theta_store: "var(--color-cyan)",
    theta_decay: "var(--color-cyan)",
    w_graph: "var(--color-emerald)",
    w_embed: "var(--color-amber)",
    w_recency: "var(--color-rose)",
  };

  return (
    <div className="space-y-5">
      <div
        className="grid items-end gap-3"
        style={{ gridTemplateColumns: "140px repeat(5, 1fr)" }}
      >
        {/* header */}
        <div />
        {params.map((p) => (
          <div
            key={p}
            className="text-[0.65rem] uppercase tracking-[0.16em] text-center"
            style={{ color: PARAM_COLOR[p] ?? "var(--color-muted)" }}
          >
            {p}
          </div>
        ))}

        {/* rows */}
        {rows.map((row, ri) => (
          <RowBars
            key={row.name}
            row={row}
            params={params}
            paramColor={PARAM_COLOR}
            maxByCol={maxByCol}
            ri={ri}
          />
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {annotations.map((a, i) => {
          const tone =
            a.includes("graph comes alive") ? "emerald" :
            a.includes("recency collapses") ? "rose" :
            a.includes("embeddings dominate") ? "amber" : "cyan";
          return (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 4 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.32, delay: i * 0.08, ease: [0.2, 0, 0, 1] }}
            >
              <Pill tone={tone as "emerald" | "rose" | "amber" | "cyan"}>{a}</Pill>
            </motion.span>
          );
        })}
      </div>
    </div>
  );
}

interface RowBarsProps {
  row: { name: string; values: number[] };
  params: string[];
  paramColor: Record<string, string>;
  maxByCol: number[];
  ri: number;
}

function RowBars({ row, params, paramColor, maxByCol, ri }: RowBarsProps) {
  const isWinner = row.name.includes("winner");
  return (
    <>
      <div
        className="text-xs font-medium leading-tight pr-2 self-end pb-1"
        style={{
          color: isWinner ? "var(--color-cyan)" : "var(--color-text-2)",
          fontWeight: isWinner ? 600 : 500,
        }}
      >
        {row.name}
      </div>
      {params.map((p, ci) => {
        const v = row.values[ci];
        const norm = maxByCol[ci] > 0 ? Math.abs(v) / maxByCol[ci] : 0;
        const heightPct = 12 + norm * 76;
        return (
          <div key={p} className="flex flex-col items-center gap-1">
            <motion.div
              initial={{ scaleY: 0 }}
              whileInView={{ scaleY: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.05 * ci + 0.18 * ri, ease: [0.2, 0, 0, 1] }}
              className="w-full max-w-[60px] rounded-t"
              style={{
                height: `${heightPct}px`,
                background: `linear-gradient(180deg, ${paramColor[p] ?? "var(--color-cyan)"}cc, ${paramColor[p] ?? "var(--color-cyan)"}55)`,
                transformOrigin: "bottom",
                outline: isWinner ? `1px solid ${paramColor[p] ?? "var(--color-cyan)"}` : "none",
              }}
              title={`${p} = ${v.toFixed(3)}`}
            />
            <div
              className="text-[0.65rem] font-mono tabular-nums"
              style={{ color: "var(--color-text-2)" }}
            >
              {v.toFixed(3)}
            </div>
          </div>
        );
      })}
    </>
  );
}

/* ============================================================================
 * DrillDownModal
 * ========================================================================= */

interface DrillDownModalProps {
  example: DrilldownExample;
  triggerCfg?: string;
  triggerMode?: string;
  configs: FinanceBenchConfig[];
  onClose: () => void;
}

function DrillDownModal({
  example,
  triggerCfg,
  triggerMode,
  configs,
  onClose,
}: DrillDownModalProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const cellKeys = Object.keys(example.predictions);
  const orderedCells = configs.flatMap((c) =>
    (["online", "batch"] as FinanceBenchMode[]).map((m) => ({
      key: `${c.name}__${m}`,
      label: c.label,
      mode: m,
      color: c.color,
      cfgName: c.name,
    })),
  ).filter((cell) => cellKeys.includes(cell.key));

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
      style={{ background: "rgba(2, 6, 23, 0.78)", backdropFilter: "blur(6px)" }}
      onClick={onClose}
    >
      <motion.div
        className="panel-rise w-full max-w-4xl max-h-[85vh] overflow-y-auto"
        initial={{ y: 12, opacity: 0, scale: 0.98 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        exit={{ y: 8, opacity: 0, scale: 0.99 }}
        transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-[var(--color-border)]">
          <div className="min-w-0">
            <div className="text-[0.6rem] uppercase tracking-[0.18em] mb-1" style={{ color: "var(--color-muted)" }}>
              FinanceBench · doc #{example.doc_idx} · {example.category}
            </div>
            <h3 className="text-base md:text-lg font-semibold leading-tight">
              {example.question}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded hover:bg-[var(--color-border)] transition"
            style={{ transition: "background 180ms cubic-bezier(0.2, 0, 0, 1)" }}
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-4 space-y-4">
          <div>
            <div
              className="text-[0.6rem] uppercase tracking-[0.18em] mb-1"
              style={{ color: "var(--color-muted)" }}
            >
              gold answer
            </div>
            <div
              className="font-mono text-xs px-3 py-2 rounded"
              style={{
                background: "var(--color-emerald-soft)",
                color: "var(--color-text)",
                border: "1px solid var(--color-emerald)33",
              }}
            >
              {example.gold_answer}
            </div>
          </div>

          <div>
            <div
              className="text-[0.6rem] uppercase tracking-[0.18em] mb-2"
              style={{ color: "var(--color-muted)" }}
            >
              predictions across {orderedCells.length} cells
            </div>
            <div className="space-y-2">
              {orderedCells.map((cell) => {
                const pred = example.predictions[cell.key];
                if (!pred) return null;
                const isTrigger =
                  triggerCfg === cell.cfgName && triggerMode === cell.mode;
                return (
                  <div
                    key={cell.key}
                    className="grid items-start gap-3 px-3 py-2 rounded"
                    style={{
                      gridTemplateColumns: "160px 1fr 60px",
                      background: isTrigger ? `${cell.color}1a` : "rgba(255,255,255,0.02)",
                      border: isTrigger ? `1px solid ${cell.color}80` : "1px solid var(--color-border-soft)",
                    }}
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span
                        className="inline-block w-1.5 h-3 rounded-sm"
                        style={{ background: cell.color }}
                      />
                      <span style={{ color: cell.color }}>{cell.label}</span>
                      <span style={{ color: "var(--color-muted)" }}>· {cell.mode}</span>
                    </div>
                    <div
                      className="text-xs font-mono leading-relaxed"
                      style={{ color: "var(--color-text)" }}
                    >
                      {pred.predicted}
                    </div>
                    <div
                      className="text-right text-xs tabular-nums font-semibold"
                      style={{
                        color:
                          pred.judge_score >= 0.75
                            ? "var(--color-emerald)"
                            : pred.judge_score >= 0.25
                              ? "var(--color-amber)"
                              : "var(--color-rose)",
                      }}
                    >
                      {pred.judge_score.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
            ESC to close · click outside to close. Predictions truncated at 400 chars.
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
