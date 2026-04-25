/**
 * ParetoScatter — token-cost vs reward, all benchmark systems on
 * MultiHop-KeyDoor. V4 highlighted, others muted; hover reveals stats.
 *
 * Built with recharts for accessibility + lightweight interaction.
 */

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import type { BenchmarkData } from "../../data/types";
import { useMemo } from "react";

interface Point {
  system: string;
  tokens: number;
  reward: number;
  precision: number | null;
  size: number;
}

interface ParetoScatterProps {
  data: BenchmarkData;
  env?: string;
  height?: number;
}

const ACCENT_BY_SYS: Record<string, string> = {
  GraphMemoryV4: "var(--color-cyan)",
  GraphMemoryV5: "var(--color-emerald)",
  EpisodicSemantic: "var(--color-violet)",
  RAGMemory: "var(--color-amber)",
};

export function ParetoScatter({ data, env = "MultiHop-KeyDoor", height = 380 }: ParetoScatterProps) {
  const points = useMemo(() => {
    const block = (data as Record<string, Record<string, unknown>>)?.[env];
    if (!block) return [];
    const out: Point[] = [];
    for (const [sys, raw] of Object.entries(block)) {
      if (sys.startsWith("_")) continue;
      const row = raw as Record<string, unknown>;
      if (row.error) continue;
      const reward = typeof row.mean_reward === "number" ? row.mean_reward : null;
      const tokens = typeof row.mean_tokens === "number" ? row.mean_tokens : null;
      if (reward === null || tokens === null) continue;
      out.push({
        system: sys,
        tokens,
        reward,
        precision: typeof row.retrieval_precision === "number" ? (row.retrieval_precision as number) : null,
        size: typeof row.mean_memory_size === "number" ? (row.mean_memory_size as number) : 0,
      });
    }
    return out;
  }, [data, env]);

  if (points.length === 0) {
    return (
      <div
        className="flex items-center justify-center h-[280px]"
        style={{ color: "var(--color-muted)" }}
      >
        no data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 16, right: 16, bottom: 24, left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
        <XAxis
          type="number"
          dataKey="tokens"
          name="tokens / episode"
          stroke="var(--color-text-2)"
          tickLine={false}
          axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          label={{
            value: "tokens / episode  (cost proxy →)",
            position: "insideBottom",
            offset: -8,
            fill: "var(--color-muted)",
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          dataKey="reward"
          name="reward"
          stroke="var(--color-text-2)"
          tickLine={false}
          axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
          label={{
            value: "reward ↑",
            angle: -90,
            position: "insideLeft",
            offset: 12,
            fill: "var(--color-muted)",
            fontSize: 11,
          }}
        />
        <Tooltip
          cursor={{ stroke: "rgba(255, 255, 255, 0.18)", strokeDasharray: "4 4" }}
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border-strong)",
            borderRadius: 8,
            fontSize: 12,
          }}
          itemStyle={{ color: "var(--color-text)" }}
          labelStyle={{ color: "var(--color-text)" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as Point;
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
                  style={{ color: "var(--color-text)" }}
                >
                  {p.system}
                </div>
                <div style={{ color: "var(--color-text-2)" }}>
                  reward {p.reward.toFixed(3)} · tokens {Math.round(p.tokens)}
                  {p.precision !== null ? ` · prec ${p.precision.toFixed(2)}` : ""}
                  {p.size ? ` · mem ${Math.round(p.size)}` : ""}
                </div>
              </div>
            );
          }}
        />
        <Scatter data={points}>
          {points.map((p) => (
            <Cell
              key={p.system}
              fill={
                ACCENT_BY_SYS[p.system] ??
                (p.reward < 0.05 ? "rgba(255,255,255,0.2)" : "rgba(255, 255, 255, 0.55)")
              }
              stroke="rgba(255, 255, 255, 0.4)"
              strokeWidth={p.system === "GraphMemoryV4" ? 2 : 0.5}
              r={p.system === "GraphMemoryV4" ? 9 : 7}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
