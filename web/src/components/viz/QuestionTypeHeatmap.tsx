/**
 * QuestionTypeHeatmap — 5 question categories × 6 configs.
 *
 * Cell color encodes mean judge score for questions in that category
 * (heuristically classified by regex). Cell label shows the score
 * and a small "n=N" badge with the bucket size.
 *
 * Mode toggle (online / batch) switches the data. The optional view
 * toggle swaps the cell metric.
 */

import { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/format";
import type {
  FinanceBenchConfig,
  FinanceBenchMode,
  QuestionTypes,
} from "../../data/financeBenchTypes";

interface QuestionTypeHeatmapProps {
  configs: FinanceBenchConfig[];
  question_types: QuestionTypes;
  mode: FinanceBenchMode;
  onModeChange: (m: FinanceBenchMode) => void;
  visibleConfigs?: Set<string>;
  onCellClick?: (cfgName: string, mode: FinanceBenchMode, category: string) => void;
}

const CATEGORY_BLURBS: Record<string, string> = {
  "multi-formula calc":
    "Requires combining 2+ line items (ratio / margin / CAGR / coverage).",
  "qualitative judgement":
    "Subjective phrasing — 'is X capital-intensive', 'does X improve'.",
  "list extraction":
    "Asks for an enumeration of names, items, or labeled categories.",
  "simple extraction":
    "Single FY value lookup ('what is the FY2022 revenue').",
  "other":
    "Anything not matched by the four regex patterns above.",
};

function judgeColor(v: number): string {
  // 0 → faded slate; 0.5 → amber; 1 → emerald
  if (v >= 0.75) return "rgba(16, 185, 129, 0.78)";
  if (v >= 0.55) return "rgba(52, 211, 153, 0.55)";
  if (v >= 0.35) return "rgba(245, 158, 11, 0.50)";
  if (v >= 0.15) return "rgba(244, 63, 94, 0.42)";
  return "rgba(100, 116, 139, 0.30)";
}

export function QuestionTypeHeatmap({
  configs,
  question_types,
  mode,
  onModeChange,
  visibleConfigs,
  onCellClick,
}: QuestionTypeHeatmapProps) {
  const [hovered, setHovered] = useState<{ cfg: string; cat: string } | null>(null);

  const shownConfigs = visibleConfigs
    ? configs.filter((c) => visibleConfigs.has(c.name))
    : configs;
  const cats = question_types.categories;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="text-[0.65rem] uppercase tracking-[0.18em]" style={{ color: "var(--color-muted)" }}>
          mean judge by question category · {mode} mode
        </div>
        <div className="inline-flex rounded-full border border-[var(--color-border)] p-1 text-xs">
          {(["online", "batch"] as FinanceBenchMode[]).map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={cn(
                "px-3 py-1 rounded-full transition",
                mode === m
                  ? "bg-[var(--color-cyan-soft)] text-[var(--color-cyan)]"
                  : "text-[var(--color-text-2)] hover:text-[var(--color-text)]",
              )}
              style={{
                transition: "background 180ms cubic-bezier(0.2, 0, 0, 1), color 180ms cubic-bezier(0.2, 0, 0, 1)",
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[680px]">
          <div
            className="grid items-stretch"
            style={{
              gridTemplateColumns: `180px repeat(${shownConfigs.length}, minmax(96px, 1fr))`,
              gap: 4,
            }}
          >
            <div />
            {shownConfigs.map((c) => (
              <div
                key={c.name}
                className="text-[0.65rem] uppercase tracking-[0.16em] py-2 text-center"
                style={{ color: c.color }}
                title={c.label}
              >
                {c.label.split(" ")[0]}
              </div>
            ))}

            {cats.map((cat) => (
              <RowFragment
                key={cat}
                cat={cat}
                shownConfigs={shownConfigs}
                question_types={question_types}
                mode={mode}
                hovered={hovered}
                setHovered={setHovered}
                onCellClick={onCellClick}
              />
            ))}
          </div>
        </div>
      </div>

      {hovered ? (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18, ease: [0.2, 0, 0, 1] }}
          className="panel-rise px-4 py-3 text-xs"
        >
          <div className="font-mono mb-1">
            <span style={{ color: shownConfigs.find((c) => c.name === hovered.cfg)?.color }}>
              {hovered.cfg}
            </span>{" "}
            <span style={{ color: "var(--color-muted)" }}>·</span>{" "}
            <span style={{ color: "var(--color-text-2)" }}>{hovered.cat}</span>
          </div>
          <div style={{ color: "var(--color-text-2)" }}>
            {CATEGORY_BLURBS[hovered.cat] ?? ""}
          </div>
        </motion.div>
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        {cats.map((cat) => {
          const counts = question_types.counts;
          const idx = cats.indexOf(cat);
          return (
            <div
              key={cat}
              className="panel-rise px-4 py-3"
              style={{ borderColor: "var(--color-border-soft)" }}
            >
              <div className="font-semibold mb-1" style={{ color: "var(--color-text)" }}>
                {cat}
              </div>
              <div style={{ color: "var(--color-text-2)" }}>
                hard {counts.always_hard[idx]} · easy {counts.always_easy[idx]} · mid {counts.mid[idx]}
              </div>
              <div className="mt-1" style={{ color: "var(--color-muted)" }}>
                {CATEGORY_BLURBS[cat]}
              </div>
            </div>
          );
        })}
      </div>

      <div className="text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
        {question_types.category_method}.
      </div>
    </div>
  );
}

interface RowFragmentProps {
  cat: string;
  shownConfigs: FinanceBenchConfig[];
  question_types: QuestionTypes;
  mode: FinanceBenchMode;
  hovered: { cfg: string; cat: string } | null;
  setHovered: (h: { cfg: string; cat: string } | null) => void;
  onCellClick?: (cfgName: string, mode: FinanceBenchMode, category: string) => void;
}

function RowFragment({
  cat,
  shownConfigs,
  question_types,
  mode,
  hovered,
  setHovered,
  onCellClick,
}: RowFragmentProps) {
  return (
    <>
      <div
        className="text-xs py-3 pr-2 text-right font-medium leading-tight"
        style={{ color: "var(--color-text-2)" }}
      >
        {cat}
      </div>
      {shownConfigs.map((c) => {
        const cellKey = `${c.name}__${mode}`;
        const score = question_types.per_cell_mean_judge[cellKey]?.[cat] ?? 0;
        const n = question_types.per_cell_n_by_cat[cellKey]?.[cat] ?? 0;
        const isHovered = hovered?.cfg === c.name && hovered?.cat === cat;

        return (
          <button
            key={c.name + cat}
            onMouseEnter={() => setHovered({ cfg: c.name, cat })}
            onMouseLeave={() => setHovered(null)}
            onClick={() => onCellClick?.(c.name, mode, cat)}
            className="flex flex-col items-center justify-center py-3 rounded-md transition-all cursor-pointer relative"
            style={{
              background: judgeColor(score),
              outline: isHovered ? `1.5px solid ${c.color}` : "1px solid rgba(255,255,255,0.05)",
              transform: isHovered ? "scale(1.04)" : "scale(1)",
              transition: "transform 180ms cubic-bezier(0.2, 0, 0, 1), outline 180ms cubic-bezier(0.2, 0, 0, 1)",
              minHeight: 56,
            }}
          >
            <div
              className="font-mono text-sm font-semibold tabular-nums"
              style={{ color: score >= 0.55 ? "rgba(2, 6, 23, 0.85)" : "var(--color-text)" }}
            >
              {score.toFixed(2)}
            </div>
            <div
              className="text-[0.6rem]"
              style={{ color: score >= 0.55 ? "rgba(2, 6, 23, 0.65)" : "var(--color-muted)" }}
            >
              n={n}
            </div>
          </button>
        );
      })}
    </>
  );
}
