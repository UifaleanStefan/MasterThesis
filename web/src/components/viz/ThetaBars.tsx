/**
 * ThetaBars — animated SVG visualization of the 10D θ vector.
 *
 * Each dimension is one vertical bar. Heights animate when `theta` changes
 * (via framer-motion's <motion.rect>). Used both in the hero (auto-morphing
 * between TF-IDF and MiniLM optima) and in the Architecture / MiniLM Pivot
 * sections (driven by user interaction).
 */

import { motion } from "framer-motion";
import type { V4Theta } from "../../data/types";
import { cn } from "../../lib/format";

const DIMS: Array<{
  key: keyof V4Theta;
  label: string;
  short: string;
  group: "store" | "decay" | "retrieve";
  /** Display-time max so each dim's bar can fill its track usefully. */
  max: number;
}> = [
  { key: "theta_store",    label: "θ_store",    short: "store",    group: "store",    max: 1 },
  { key: "theta_novel",    label: "θ_novel",    short: "novel",    group: "store",    max: 1 },
  { key: "theta_erich",    label: "θ_erich",    short: "erich",    group: "store",    max: 1 },
  { key: "theta_surprise", label: "θ_surprise", short: "surprise", group: "store",    max: 1 },
  { key: "theta_entity",   label: "θ_entity",   short: "entity",   group: "store",    max: 1 },
  { key: "theta_temporal", label: "θ_temporal", short: "temporal", group: "store",    max: 1 },
  { key: "theta_decay",    label: "θ_decay",    short: "decay",    group: "decay",    max: 1 },
  { key: "w_graph",        label: "w_graph",    short: "graph",    group: "retrieve", max: 4 },
  { key: "w_embed",        label: "w_embed",    short: "embed",    group: "retrieve", max: 4 },
  { key: "w_recency",      label: "w_recency",  short: "recency",  group: "retrieve", max: 4 },
];

const GROUP_COLOR = {
  store: "var(--color-cyan)",
  decay: "var(--color-emerald)",
  retrieve: "var(--color-violet)",
};

interface ThetaBarsProps {
  theta: V4Theta;
  /** Optional secondary theta drawn as ghost bars behind the primary. */
  ghost?: V4Theta | null;
  /** Show value text under each bar. */
  showValues?: boolean;
  /** Show short labels under each bar. */
  showLabels?: boolean;
  /** Override bar color (defaults to per-group). */
  tone?: "cyan" | "violet" | "amber" | "emerald" | "rose" | null;
  /** Total svg height in px. */
  height?: number;
  className?: string;
}

const TONE_VAR = {
  cyan: "var(--color-cyan)",
  violet: "var(--color-violet)",
  amber: "var(--color-amber)",
  emerald: "var(--color-emerald)",
  rose: "var(--color-rose)",
};

export function ThetaBars({
  theta,
  ghost = null,
  showValues = false,
  showLabels = false,
  tone = null,
  height = 220,
  className,
}: ThetaBarsProps) {
  const padX = 12;
  const barW = 24;
  const gap = 14;
  const innerH = height - (showLabels ? 36 : 0) - (showValues ? 18 : 0);
  const labelY = innerH + 14;
  const valueY = innerH + (showLabels ? 30 : 14);

  const totalW = DIMS.length * barW + (DIMS.length - 1) * gap + padX * 2;

  return (
    <svg
      role="img"
      aria-label="10-dimensional theta vector visualization"
      className={cn("w-full select-none", className)}
      viewBox={`0 0 ${totalW} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {DIMS.map((d, i) => {
        const x = padX + i * (barW + gap);
        const v = clamp01(theta[d.key] / d.max);
        const ghostV = ghost ? clamp01(ghost[d.key] / d.max) : null;
        const color = tone ? TONE_VAR[tone] : GROUP_COLOR[d.group];
        const valBoxH = innerH * v;
        const ghostH = ghostV !== null ? innerH * ghostV : 0;

        return (
          <g key={d.key}>
            {/* Track */}
            <rect
              x={x}
              y={0}
              width={barW}
              height={innerH}
              rx={4}
              fill="rgba(255, 255, 255, 0.04)"
              stroke="rgba(255, 255, 255, 0.06)"
            />
            {/* Ghost (e.g. previous backend) */}
            {ghostV !== null && (
              <motion.rect
                x={x}
                y={innerH - ghostH}
                width={barW}
                height={ghostH}
                rx={4}
                fill="rgba(255, 255, 255, 0.08)"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              />
            )}
            {/* Active bar */}
            <motion.rect
              x={x}
              width={barW}
              rx={4}
              fill={color}
              initial={false}
              animate={{
                y: innerH - valBoxH,
                height: valBoxH,
              }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] as const }}
              style={{ filter: `drop-shadow(0 0 6px ${color}33)` }}
            />
            {showLabels && (
              <text
                x={x + barW / 2}
                y={labelY}
                textAnchor="middle"
                fontSize="9"
                fontFamily="var(--font-mono)"
                fill="var(--color-text-2)"
              >
                {d.short}
              </text>
            )}
            {showValues && (
              <motion.text
                x={x + barW / 2}
                y={valueY}
                textAnchor="middle"
                fontSize="9.5"
                fontFamily="var(--font-mono)"
                fill={color}
                key={`${d.key}-${theta[d.key]}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.4 }}
              >
                {(theta[d.key] as number).toFixed(2)}
              </motion.text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}
