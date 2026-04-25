/**
 * ThetaRadar — 10-axis radar chart for the V4 theta vector.
 *
 * Used in the MiniLM Pivot section to visualize how TF-IDF and MiniLM
 * optima differ in shape (not just in absolute values).
 */

import { motion } from "framer-motion";
import { useMemo } from "react";
import type { V4Theta } from "../../data/types";
import { cn } from "../../lib/format";

interface ThetaRadarProps {
  theta: V4Theta;
  /** Optional secondary theta drawn faded behind the primary. */
  ghost?: V4Theta | null;
  /** Color for the primary polygon (defaults to violet). */
  tone?: "cyan" | "violet" | "amber" | "emerald";
  /** Pixel size of the SVG (square). */
  size?: number;
  className?: string;
  /** Show axis labels. */
  showLabels?: boolean;
}

const DIMS: Array<{ key: keyof V4Theta; label: string; max: number }> = [
  { key: "theta_store",    label: "store",    max: 1 },
  { key: "theta_novel",    label: "novel",    max: 1 },
  { key: "theta_erich",    label: "erich",    max: 1 },
  { key: "theta_surprise", label: "surprise", max: 1 },
  { key: "theta_entity",   label: "entity",   max: 1 },
  { key: "theta_temporal", label: "temporal", max: 1 },
  { key: "theta_decay",    label: "decay",    max: 1 },
  { key: "w_graph",        label: "w_graph",   max: 4 },
  { key: "w_embed",        label: "w_embed",   max: 4 },
  { key: "w_recency",      label: "w_recency", max: 4 },
];

const TONE: Record<NonNullable<ThetaRadarProps["tone"]>, string> = {
  cyan: "var(--color-cyan)",
  violet: "var(--color-violet)",
  amber: "var(--color-amber)",
  emerald: "var(--color-emerald)",
};

export function ThetaRadar({
  theta,
  ghost = null,
  tone = "violet",
  size = 320,
  className,
  showLabels = true,
}: ThetaRadarProps) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.36;

  const positions = useMemo(() => {
    return DIMS.map((d, i) => {
      const angle = -Math.PI / 2 + (i / DIMS.length) * 2 * Math.PI;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      return { angle, cos, sin, key: d.key, label: d.label, max: d.max };
    });
  }, []);

  const points = positions.map((p) => {
    const v = Math.max(0, Math.min(1, theta[p.key] / p.max));
    return `${cx + p.cos * radius * v},${cy + p.sin * radius * v}`;
  });

  const ghostPoints = ghost
    ? positions.map((p) => {
        const v = Math.max(0, Math.min(1, ghost[p.key] / p.max));
        return `${cx + p.cos * radius * v},${cy + p.sin * radius * v}`;
      })
    : null;

  const accent = TONE[tone];

  return (
    <svg
      role="img"
      aria-label="10-axis theta radar"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className={cn("select-none", className)}
    >
      {/* Concentric grid rings */}
      {[0.25, 0.5, 0.75, 1].map((r) => (
        <circle
          key={r}
          cx={cx}
          cy={cy}
          r={radius * r}
          fill="none"
          stroke="rgba(255, 255, 255, 0.06)"
          strokeDasharray={r === 1 ? "" : "2 4"}
        />
      ))}
      {/* Spokes + labels */}
      {positions.map((p, i) => {
        const lx = cx + p.cos * (radius + 18);
        const ly = cy + p.sin * (radius + 18);
        return (
          <g key={i}>
            <line
              x1={cx}
              y1={cy}
              x2={cx + p.cos * radius}
              y2={cy + p.sin * radius}
              stroke="rgba(255, 255, 255, 0.06)"
            />
            {showLabels && (
              <text
                x={lx}
                y={ly}
                textAnchor={p.cos > 0.2 ? "start" : p.cos < -0.2 ? "end" : "middle"}
                dominantBaseline={p.sin > 0.5 ? "hanging" : p.sin < -0.5 ? "auto" : "middle"}
                fontSize="9"
                fontFamily="var(--font-mono)"
                fill="var(--color-text-2)"
              >
                {p.label}
              </text>
            )}
          </g>
        );
      })}

      {/* Ghost polygon */}
      {ghostPoints ? (
        <motion.polygon
          points={ghostPoints.join(" ")}
          fill="rgba(255, 255, 255, 0.06)"
          stroke="rgba(255, 255, 255, 0.18)"
          strokeWidth="1"
          strokeDasharray="4 4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        />
      ) : null}

      {/* Primary polygon */}
      <motion.polygon
        points={points.join(" ")}
        fill={`${accent}22`}
        stroke={accent}
        strokeWidth="1.5"
        initial={false}
        animate={{ points: points.join(" ") }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] as const }}
        style={{ filter: `drop-shadow(0 0 8px ${accent}55)` }}
      />

      {/* Vertices */}
      {positions.map((p, i) => {
        const v = Math.max(0, Math.min(1, theta[p.key] / p.max));
        return (
          <circle
            key={i}
            cx={cx + p.cos * radius * v}
            cy={cy + p.sin * radius * v}
            r="2.5"
            fill={accent}
          />
        );
      })}
    </svg>
  );
}
