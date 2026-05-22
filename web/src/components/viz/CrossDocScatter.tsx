/**
 * CrossDocScatter — judge score vs cross-document retrieval bleed.
 *
 * Each point is one (config, mode) cell from the FinanceBench corpus QA.
 * X axis = `in_doc_ratio` (fraction of retrieved steps from own document);
 * Y axis = mean judge score. The reference line at y=0.7 marks the
 * gpt-4o-mini ceiling observed across the best three configs.
 *
 * Online dots are circles, batch are diamonds — same color encodes
 * config family.
 */

import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ScatterPoint } from "../../data/financeBenchTypes";

interface CrossDocScatterProps {
  points: ScatterPoint[];
  height?: number;
  onPointClick?: (point: ScatterPoint) => void;
  visibleConfigs?: Set<string>;
  highlight?: { config: string; mode: string } | null;
}

const DIAMOND = "M 0,-8 L 8,0 L 0,8 L -8,0 Z";

export function CrossDocScatter({
  points,
  height = 380,
  onPointClick,
  visibleConfigs,
  highlight,
}: CrossDocScatterProps) {
  const filtered = visibleConfigs
    ? points.filter((p) => visibleConfigs.has(p.config))
    : points;

  const online = filtered.filter((p) => p.mode === "online");
  const batch = filtered.filter((p) => p.mode === "batch");

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 16, right: 24, bottom: 40, left: 32 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
        <XAxis
          type="number"
          dataKey="in_doc_ratio"
          name="in-doc retrieval ratio"
          domain={[0, "dataMax + 0.02"]}
          tickFormatter={(v: number) => v.toFixed(2)}
          stroke="var(--color-text-2)"
          tickLine={false}
          axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          label={{
            value: "in-doc retrieval ratio  (higher = less cross-doc bleed →)",
            position: "insideBottom",
            offset: -16,
            fill: "var(--color-muted)",
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          dataKey="mean_judge"
          name="judge"
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(1)}
          stroke="var(--color-text-2)"
          tickLine={false}
          axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          label={{
            value: "mean judge ↑",
            angle: -90,
            position: "insideLeft",
            offset: 12,
            fill: "var(--color-muted)",
            fontSize: 11,
          }}
        />
        <ReferenceLine
          y={0.7}
          stroke="var(--color-amber)"
          strokeDasharray="4 4"
          strokeOpacity={0.55}
          label={{
            value: "gpt-4o-mini ceiling ≈ 0.70",
            position: "insideTopRight",
            fill: "var(--color-amber)",
            fontSize: 10,
          }}
        />
        <Tooltip
          cursor={{ stroke: "rgba(255, 255, 255, 0.18)", strokeDasharray: "4 4" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as ScatterPoint;
            return (
              <div
                className="panel px-3 py-2 text-xs font-mono"
                style={{
                  background: "var(--color-surface)",
                  borderColor: "var(--color-border-strong)",
                }}
              >
                <div
                  className="font-semibold mb-1"
                  style={{ color: p.color }}
                >
                  {p.label} · {p.mode}
                </div>
                <div style={{ color: "var(--color-text-2)" }}>
                  judge {p.mean_judge.toFixed(3)}
                  {" "}[{p.ci_lower.toFixed(2)}, {p.ci_upper.toFixed(2)}]
                </div>
                <div style={{ color: "var(--color-text-2)" }}>
                  recall {p.mean_recall.toFixed(3)} · in-doc {p.in_doc_ratio.toFixed(3)}
                </div>
                <div className="mt-1" style={{ color: "var(--color-muted)" }}>
                  n={p.n} · click for examples
                </div>
              </div>
            );
          }}
        />
        <Scatter
          data={online}
          shape={((rawProps: unknown) => {
            const props = rawProps as { cx?: number; cy?: number; payload: ScatterPoint };
            const cx = props.cx ?? 0;
            const cy = props.cy ?? 0;
            const isHighlighted =
              highlight?.config === props.payload.config && highlight?.mode === "online";
            return (
              <circle
                cx={cx}
                cy={cy}
                r={isHighlighted ? 11 : 8}
                fill={props.payload.color}
                stroke="rgba(255,255,255,0.55)"
                strokeWidth={isHighlighted ? 2.2 : 1}
                style={{ cursor: onPointClick ? "pointer" : "default", transition: "r 180ms cubic-bezier(0.2, 0, 0, 1)" }}
                onClick={() => onPointClick?.(props.payload)}
              />
            );
          }) as never}
        />
        <Scatter
          data={batch}
          shape={((rawProps: unknown) => {
            const props = rawProps as { cx?: number; cy?: number; payload: ScatterPoint };
            const cx = props.cx ?? 0;
            const cy = props.cy ?? 0;
            const isHighlighted =
              highlight?.config === props.payload.config && highlight?.mode === "batch";
            return (
              <path
                d={DIAMOND}
                transform={`translate(${cx}, ${cy}) scale(${isHighlighted ? 1.3 : 1})`}
                fill={props.payload.color}
                fillOpacity={0.65}
                stroke="rgba(255,255,255,0.55)"
                strokeWidth={isHighlighted ? 2.2 : 1}
                style={{ cursor: onPointClick ? "pointer" : "default", transition: "transform 180ms cubic-bezier(0.2, 0, 0, 1)" }}
                onClick={() => onPointClick?.(props.payload)}
              />
            );
          }) as never}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
