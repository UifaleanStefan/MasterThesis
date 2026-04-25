/**
 * AblationKnockout — interactive bar chart of V4 ablation results.
 *
 * Each row is one ablation config (e.g. no_novelty, no_erich). Clicking
 * the row "knocks out" that component and the bars sort/animate to show
 * the post-knockout reward. theta_novel knockout flashes red because it
 * costs 100% of the reward.
 */

import { motion, AnimatePresence } from "framer-motion";
import { useMemo, useState } from "react";
import type { AblationData, AblationConfigResult } from "../../data/types";
import { fmt } from "../../lib/format";

interface Row {
  name: string;
  description: string;
  reward: number;
  precision: number | null;
  memSize: number;
  degradation: number; // 0 = no degradation, 1 = 100%
}

const PINNED_ORDER: string[] = [
  "full",
  "no_recency",
  "no_surprise",
  "graph_only",
  "no_embed",
  "no_decay",
  "no_erich",
  "v1_equivalent",
  "store_all",
  "no_novelty",
];

export function AblationKnockout({ data }: { data: AblationData }) {
  const [active, setActive] = useState<string | null>(null);

  const rows: Row[] = useMemo(() => {
    if (!data?.results) return [];
    const fullR = data.results.full?.mean_reward ?? 0.001;
    const out: Row[] = [];
    for (const [name, raw] of Object.entries(data.results)) {
      const r = raw as AblationConfigResult;
      out.push({
        name,
        description: r.description ?? name,
        reward: r.mean_reward,
        precision: r.mean_precision,
        memSize: r.mean_memory_size,
        degradation: Math.max(0, (fullR - r.mean_reward) / fullR),
      });
    }
    out.sort((a, b) => {
      const ai = PINNED_ORDER.indexOf(a.name);
      const bi = PINNED_ORDER.indexOf(b.name);
      if (ai === -1 && bi === -1) return b.reward - a.reward;
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
    return out;
  }, [data]);

  const maxReward = useMemo(() => Math.max(0.01, ...rows.map((r) => r.reward)), [rows]);
  const activeRow = active ? rows.find((r) => r.name === active) : null;
  const fullRow = rows.find((r) => r.name === "full");

  return (
    <div className="grid lg:grid-cols-[1.5fr_1fr] gap-6">
      <div className="panel-rise p-5">
        <div className="text-[0.65rem] uppercase tracking-[0.18em] mb-3" style={{ color: "var(--color-muted)" }}>
          click to knock out a component
        </div>
        <div className="space-y-1.5">
          {rows.map((r) => {
            const t = r.reward / maxReward;
            const tone = colorFor(r);
            const isActive = active === r.name;
            const isFull = r.name === "full";
            return (
              <button
                key={r.name}
                onClick={() => setActive(isActive ? null : r.name)}
                className="w-full grid grid-cols-[150px_1fr_50px] items-center gap-3 py-2 px-2 rounded-md hover:bg-[rgba(255,255,255,0.02)] transition-colors text-left"
                style={{
                  outline: isActive ? `1px solid ${tone}` : "1px solid transparent",
                }}
              >
                <span
                  className="text-xs font-mono truncate"
                  style={{ color: isFull ? "var(--color-cyan)" : "var(--color-text-2)" }}
                >
                  {r.name}
                </span>
                <div className="relative h-5">
                  <div
                    className="absolute inset-y-0 left-0 right-0 rounded-full"
                    style={{
                      height: 6,
                      top: "50%",
                      transform: "translateY(-50%)",
                      background: "rgba(255, 255, 255, 0.04)",
                    }}
                  />
                  <motion.div
                    className="absolute rounded-full"
                    initial={false}
                    animate={{ width: `${t * 100}%` }}
                    transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] as const }}
                    style={{
                      height: 6,
                      top: "50%",
                      transform: "translateY(-50%)",
                      left: 0,
                      background: tone,
                      boxShadow: `0 0 6px ${tone}55`,
                    }}
                  />
                </div>
                <span
                  className="text-xs font-mono tabular-nums text-right"
                  style={{ color: tone }}
                >
                  {fmt(r.reward, 3)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="panel-rise p-6 flex flex-col">
        <div className="text-[0.65rem] uppercase tracking-[0.18em]" style={{ color: "var(--color-muted)" }}>
          {activeRow ? "knockout summary" : "what V4 needs"}
        </div>
        <AnimatePresence mode="wait">
          {activeRow ? (
            <motion.div
              key={activeRow.name}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="mt-2"
            >
              <div className="font-mono text-2xl font-semibold" style={{ color: colorFor(activeRow) }}>
                {activeRow.name}
              </div>
              <p className="mt-2 text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
                {activeRow.description}
              </p>

              <div className="grid grid-cols-3 gap-3 mt-5">
                <Mini label="reward" value={fmt(activeRow.reward, 3)} tone={colorFor(activeRow)} />
                <Mini
                  label="degradation"
                  value={`${(activeRow.degradation * 100).toFixed(0)}%`}
                  tone={
                    activeRow.degradation > 0.5
                      ? "var(--color-rose)"
                      : "var(--color-amber)"
                  }
                />
                <Mini
                  label="mem size"
                  value={fmt(activeRow.memSize, 0)}
                  tone="var(--color-violet)"
                />
              </div>

              {activeRow.degradation >= 0.99 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="mt-5 panel px-4 py-3 text-sm leading-relaxed"
                  style={{
                    borderColor: "var(--color-rose)",
                    background: "rgba(251, 113, 133, 0.07)",
                    color: "var(--color-text)",
                  }}
                >
                  <span style={{ color: "var(--color-rose)", fontWeight: 600 }}>
                    100% degradation.
                  </span>{" "}
                  Without this, the system stores nothing or stores everything. Memory becomes
                  uninformative; reward collapses.
                </motion.div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="mt-2"
            >
              <div className="font-semibold text-2xl">
                <span style={{ color: "var(--color-cyan)" }}>full V4</span> · reward{" "}
                {fmt(fullRow?.reward, 3)}
              </div>
              <p className="mt-3 text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
                Click any row to knock out one component of V4&apos;s θ vector and read the cost.
                <br />
                <span style={{ color: "var(--color-rose)" }}>theta_novel</span> is the load-bearing
                pillar — costs 100% of the reward. <span style={{ color: "var(--color-amber)" }}>theta_erich</span>{" "}
                is the second pillar at ~64%.
              </p>
              <div className="grid grid-cols-3 gap-3 mt-5">
                <Mini label="full reward" value={fmt(fullRow?.reward, 3)} tone="var(--color-cyan)" />
                <Mini label="precision" value={fmt(fullRow?.precision, 2)} tone="var(--color-emerald)" />
                <Mini label="mem size" value={fmt(fullRow?.memSize, 0)} tone="var(--color-violet)" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function colorFor(r: Row): string {
  if (r.name === "full") return "var(--color-cyan)";
  if (r.degradation >= 0.95) return "var(--color-rose)";
  if (r.degradation >= 0.5) return "var(--color-amber)";
  if (r.degradation >= 0.1) return "var(--color-violet)";
  return "var(--color-emerald)";
}

function Mini({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="panel px-3 py-2">
      <div className="text-[0.6rem] uppercase tracking-[0.14em]" style={{ color: "var(--color-muted)" }}>
        {label}
      </div>
      <div className="mt-0.5 font-mono font-semibold tabular-nums" style={{ color: tone }}>
        {value}
      </div>
    </div>
  );
}
