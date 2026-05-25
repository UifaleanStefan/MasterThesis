/**
 * CalibrationTrajectory — per-decile Protocol B calibration mean (1500q / config).
 *
 * Shows how each memory configuration's judge score degrades as the corpus
 * dilutes the memory. Each line is one config; each x-tick is a decile of
 * the 1500 calibration questions (sorted by docs_seen). Selective+tuned
 * configs (V4ₜ corpus-tuned, attention, RAG, BM25) hold; canonical θ +
 * windowed/TF-IDF + dump-all collapse.
 */

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  CalibrationCell,
  FinanceBenchConfig,
} from "../../data/financeBenchTypes";

interface CalibrationTrajectoryProps {
  calibration: { [config: string]: CalibrationCell };
  configs: FinanceBenchConfig[];
  visibleConfigs?: Set<string>;
  height?: number;
}

export function CalibrationTrajectory({
  calibration,
  configs,
  visibleConfigs,
  height = 360,
}: CalibrationTrajectoryProps) {
  const shown = configs.filter(
    (c) =>
      (!visibleConfigs || visibleConfigs.has(c.name)) &&
      calibration[c.name]?.n > 0,
  );

  // Build chart data: one row per decile (1..10), one column per config.
  const data = Array.from({ length: 10 }, (_, i) => {
    const row: { decile: string; pct: string } & Record<string, number | string> = {
      decile: `D${i + 1}`,
      pct: `${(i + 1) * 10}%`,
    };
    for (const cfg of shown) {
      const d = calibration[cfg.name]?.deciles[i];
      if (d !== undefined) row[cfg.name] = d;
    }
    return row;
  });

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 12, right: 12, bottom: 20, left: 12 }}>
          <CartesianGrid stroke="var(--color-border-soft)" strokeDasharray="3 3" />
          <XAxis
            dataKey="pct"
            stroke="var(--color-muted)"
            tick={{ fontSize: 11, fontFamily: "monospace" }}
            label={{
              value: "calibration decile (corpus depth →)",
              position: "insideBottom",
              offset: -8,
              style: { fontSize: 11, fill: "var(--color-muted)" },
            }}
          />
          <YAxis
            domain={[0, 1]}
            stroke="var(--color-muted)"
            tick={{ fontSize: 11, fontFamily: "monospace" }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "mean Claude judge",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "var(--color-muted)" },
            }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--color-bg)",
              border: "1px solid var(--color-border)",
              fontSize: "11px",
              fontFamily: "monospace",
              borderRadius: 6,
            }}
            labelStyle={{ color: "var(--color-text-2)" }}
          />
          <Legend
            wrapperStyle={{ fontSize: 10, fontFamily: "monospace" }}
            iconSize={10}
          />
          {shown.map((cfg) => (
            <Line
              key={cfg.name}
              type="monotone"
              dataKey={cfg.name}
              name={cfg.label}
              stroke={cfg.color}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
