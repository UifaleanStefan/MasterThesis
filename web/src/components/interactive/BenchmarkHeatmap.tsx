/**
 * BenchmarkHeatmap — 12 systems x 4 environments, color-mapped reward.
 *
 * Each cell renders the mean_reward and (when available) retrieval precision
 * as a tiny ring. Hovering / clicking a cell pops a detail panel showing
 * full stats. V4 is highlighted with a cyan outline.
 *
 * Color scale: cool dark blue at 0 -> bright cyan/teal at the env's max.
 */

import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { BenchmarkData } from "../../data/types";
import { fmt } from "../../lib/format";
import { useEmbedding } from "./EmbeddingToggle";
import { TFIDF_BENCHMARK_TOP } from "../../data/tfidf_constants";

interface BenchmarkHeatmapProps {
  data: BenchmarkData;
  metric?: "mean_reward" | "retrieval_precision" | "mean_memory_size" | "efficiency";
}

const ENV_ORDER = [
  "Key-Door",
  "Goal-Room",
  "MultiHop-KeyDoor",
  "MegaQuestRoom",
] as const;

export function BenchmarkHeatmap({
  data,
  metric = "mean_reward",
}: BenchmarkHeatmapProps) {
  const { backend } = useEmbedding();

  // Build the matrix of (env, system) -> value, plus the union of systems.
  const { systems, envs } = useMemo(() => {
    const sysSet = new Set<string>();
    const envs: string[] = [];
    for (const env of ENV_ORDER) {
      const block = data?.[env];
      if (!block) continue;
      envs.push(env);
      for (const s of Object.keys(block)) {
        if (s.startsWith("_")) continue;
        sysSet.add(s);
      }
    }
    return { systems: Array.from(sysSet), envs };
  }, [data]);

  // Show TF-IDF era top-of-ranking inline when toggle is set.
  const tfidfMap = useMemo(() => {
    const m = new Map<string, { reward: number; precision: number }>();
    for (const e of TFIDF_BENCHMARK_TOP) {
      m.set(e.system, { reward: e.reward, precision: e.precision });
    }
    return m;
  }, []);

  const useTfidf = backend === "tfidf";

  function cellValue(env: string, sys: string): { reward: number | null; precision: number | null; raw: unknown } {
    if (useTfidf && env === "MultiHop-KeyDoor" && tfidfMap.has(sys)) {
      const e = tfidfMap.get(sys)!;
      return { reward: e.reward, precision: e.precision, raw: e };
    }
    const block = (data as Record<string, Record<string, unknown>>)?.[env];
    const cell = block?.[sys] as Record<string, unknown> | undefined;
    if (!cell || (cell as { error?: unknown }).error) {
      return { reward: null, precision: null, raw: cell };
    }
    return {
      reward: typeof cell.mean_reward === "number" ? cell.mean_reward : null,
      precision:
        typeof cell.retrieval_precision === "number"
          ? (cell.retrieval_precision as number)
          : null,
      raw: cell,
    };
  }

  // Compute per-env max reward to normalize the color scale.
  const maxByEnv = useMemo(() => {
    const result: Record<string, number> = {};
    for (const env of envs) {
      let m = 0;
      for (const sys of systems) {
        const v = cellValue(env, sys);
        const candidate =
          metric === "mean_reward" ? v.reward
          : metric === "retrieval_precision" ? v.precision
          : metric === "mean_memory_size" ? (v.raw as { mean_memory_size?: number } | null)?.mean_memory_size ?? 0
          : (v.raw as { efficiency?: number } | null)?.efficiency ?? 0;
        if (typeof candidate === "number" && candidate > m) m = candidate;
      }
      result[env] = m || 1;
    }
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, systems, envs, useTfidf, metric]);

  const [active, setActive] = useState<{ env: string; sys: string } | null>(null);

  // Sort systems by their MultiHop reward (descending) for readability.
  const sortedSystems = useMemo(() => {
    const sortBy = (sys: string) => {
      const v = cellValue("MultiHop-KeyDoor", sys);
      return v.reward ?? -1;
    };
    return [...systems].sort((a, b) => sortBy(b) - sortBy(a));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [systems, useTfidf]);

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[760px]">
        <div
          className="grid"
          style={{ gridTemplateColumns: `200px repeat(${envs.length}, 1fr)` }}
        >
          {/* Header row */}
          <div />
          {envs.map((env) => (
            <div
              key={env}
              className="text-[0.65rem] uppercase tracking-[0.16em] py-2 text-center"
              style={{ color: "var(--color-muted)" }}
            >
              {env.replace("-KeyDoor", "")}
            </div>
          ))}

          {sortedSystems.map((sys) => (
            <RowFragment
              key={sys}
              sys={sys}
              envs={envs}
              cellValue={cellValue}
              metric={metric}
              maxByEnv={maxByEnv}
              isV4={sys === "GraphMemoryV4"}
              onPick={(env) => setActive({ env, sys })}
              active={active}
            />
          ))}
        </div>
      </div>

      {active ? (
        <DetailPanel
          env={active.env}
          sys={active.sys}
          cell={cellValue(active.env, active.sys)}
          onClose={() => setActive(null)}
        />
      ) : null}

      <div className="flex items-center gap-3 mt-4 text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
        <span>color = {metric === "mean_reward" ? "reward" : metric === "retrieval_precision" ? "precision" : metric === "mean_memory_size" ? "mem size" : "efficiency"}</span>
        <span>·</span>
        <span>ring = retrieval precision</span>
        <span>·</span>
        <span style={{ color: "var(--color-cyan)" }}>cyan outline = V4</span>
      </div>
    </div>
  );
}

interface RowProps {
  sys: string;
  envs: string[];
  cellValue: (
    env: string,
    sys: string,
  ) => { reward: number | null; precision: number | null; raw: unknown };
  metric: "mean_reward" | "retrieval_precision" | "mean_memory_size" | "efficiency";
  maxByEnv: Record<string, number>;
  isV4: boolean;
  onPick: (env: string) => void;
  active: { env: string; sys: string } | null;
}

function RowFragment({ sys, envs, cellValue, metric, maxByEnv, isV4, onPick, active }: RowProps) {
  return (
    <>
      <div
        className="py-1 pr-3 flex items-center justify-end text-xs font-mono"
        style={{ color: isV4 ? "var(--color-cyan)" : "var(--color-text-2)" }}
      >
        {sys}
      </div>
      {envs.map((env) => {
        const v = cellValue(env, sys);
        const mx = maxByEnv[env] || 1;
        let value: number | null = null;
        if (metric === "mean_reward") value = v.reward;
        else if (metric === "retrieval_precision") value = v.precision;
        else if (metric === "mean_memory_size") {
          const mm = (v.raw as { mean_memory_size?: number } | null)?.mean_memory_size;
          value = typeof mm === "number" ? mm : null;
        } else if (metric === "efficiency") {
          const ef = (v.raw as { efficiency?: number } | null)?.efficiency;
          value = typeof ef === "number" ? ef : null;
        }
        const t = value !== null ? Math.max(0, Math.min(1, value / mx)) : 0;
        const bg =
          value === null
            ? "rgba(255, 255, 255, 0.02)"
            : `hsl(${190 - t * 30} 90% ${10 + t * 35}%)`;
        const isActive = active?.env === env && active?.sys === sys;
        return (
          <button
            key={env}
            onClick={() => onPick(env)}
            className="relative h-12 m-1 rounded-lg flex items-center justify-center text-xs font-mono transition-all"
            style={{
              background: bg,
              border: isV4 || isActive ? "1.5px solid var(--color-cyan)" : "1px solid var(--color-border)",
              color: t > 0.45 ? "white" : "var(--color-text-2)",
              boxShadow: isV4 ? "0 0 12px rgba(34, 211, 238, 0.25)" : "none",
            }}
            aria-label={`${sys} on ${env}: ${fmt(value, 3)}`}
          >
            <span className="tabular-nums">{value === null ? "—" : fmt(value, 3)}</span>
            {/* Precision ring */}
            {v.precision !== null && metric !== "retrieval_precision" ? (
              <svg
                viewBox="0 0 24 24"
                className="absolute right-1 top-1 w-3 h-3 opacity-80"
                aria-hidden
              >
                <circle cx="12" cy="12" r="10" fill="none" stroke="rgba(255, 255, 255, 0.15)" strokeWidth="3" />
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  fill="none"
                  stroke="white"
                  strokeWidth="3"
                  strokeDasharray={`${v.precision * 62.83} 62.83`}
                  transform="rotate(-90 12 12)"
                />
              </svg>
            ) : null}
          </button>
        );
      })}
    </>
  );
}

function DetailPanel({
  env, sys, cell, onClose,
}: {
  env: string;
  sys: string;
  cell: { reward: number | null; precision: number | null; raw: unknown };
  onClose: () => void;
}) {
  const raw = cell.raw as Record<string, unknown> | null;
  return (
    <motion.div
      key={`${env}-${sys}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="panel-rise mt-6 p-5"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="text-[0.65rem] uppercase tracking-[0.18em]" style={{ color: "var(--color-muted)" }}>
            cell detail
          </div>
          <div className="text-lg font-semibold mt-1">
            <span className="font-mono" style={{ color: "var(--color-cyan)" }}>{sys}</span>
            <span className="px-2 text-base" style={{ color: "var(--color-muted)" }}>·</span>
            <span style={{ color: "var(--color-text-2)" }}>{env}</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-xs px-2 py-1 rounded hover:bg-[rgba(255,255,255,0.04)]"
          style={{ color: "var(--color-text-2)" }}
        >
          close
        </button>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="reward" value={fmt(cell.reward, 3)} tone="cyan" />
        <Stat
          label="precision"
          value={cell.precision !== null ? fmt(cell.precision, 3) : "N/A"}
          tone="emerald"
        />
        <Stat
          label="mem size"
          value={typeof (raw as { mean_memory_size?: number } | null)?.mean_memory_size === "number"
            ? fmt((raw as { mean_memory_size: number }).mean_memory_size, 1)
            : "—"}
          tone="violet"
        />
        <Stat
          label="tokens / ep"
          value={typeof (raw as { mean_tokens?: number } | null)?.mean_tokens === "number"
            ? fmt((raw as { mean_tokens: number }).mean_tokens, 0)
            : "—"}
          tone="amber"
        />
      </div>
    </motion.div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: "cyan" | "violet" | "amber" | "emerald" }) {
  const accent =
    tone === "cyan" ? "var(--color-cyan)" :
    tone === "violet" ? "var(--color-violet)" :
    tone === "amber" ? "var(--color-amber)" :
    "var(--color-emerald)";
  return (
    <div className="panel px-3 py-2">
      <div className="text-[0.6rem] uppercase tracking-[0.16em]" style={{ color: "var(--color-muted)" }}>
        {label}
      </div>
      <div className="mt-1 font-semibold tabular-nums" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}
