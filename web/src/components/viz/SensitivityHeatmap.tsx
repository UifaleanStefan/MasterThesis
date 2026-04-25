/**
 * SensitivityHeatmap — 2D reward landscape over the configured sensitivity
 * dims (theta_novel × w_recency by default). The learned θ position is
 * marked with a small ring.
 */

import type { SensitivityData } from "../../data/types";
import { fmt } from "../../lib/format";

interface SensitivityHeatmapProps {
  data: SensitivityData;
  size?: number;
}

export function SensitivityHeatmap({ data, size = 320 }: SensitivityHeatmapProps) {
  const rows = data.dim1_values; // theta_novel
  const cols = data.dim2_values; // w_recency
  const grid = data.reward_grid;

  let mn = Infinity;
  let mx = -Infinity;
  for (const row of grid) {
    for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (mx === mn) mx = mn + 0.001;

  const cellW = size / cols.length;
  const cellH = size / rows.length;

  // Find learned (dim1, dim2) point.
  const learnedR = data.learned_dim1;
  const learnedC = data.learned_dim2;
  const padR = (rows[rows.length - 1] - rows[0]) || 1;
  const padC = (cols[cols.length - 1] - cols[0]) || 1;
  const learnedX = ((learnedC - cols[0]) / padC) * size;
  const learnedY = (1 - (learnedR - rows[0]) / padR) * size;

  return (
    <figure>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {grid.map((row, i) => {
          const flippedI = rows.length - 1 - i; // y axis: high theta_novel at top
          return row.map((v, j) => {
            const t = (v - mn) / (mx - mn);
            const fill = `hsl(${190 - t * 30} 90% ${10 + t * 35}%)`;
            return (
              <rect
                key={`${i}-${j}`}
                x={j * cellW}
                y={flippedI * cellH}
                width={cellW + 1}
                height={cellH + 1}
                fill={fill}
              >
                <title>
                  {`theta_novel=${fmt(rows[i], 2)} w_recency=${fmt(cols[j], 2)} reward=${fmt(v, 3)}`}
                </title>
              </rect>
            );
          });
        })}

        {/* Learned point ring */}
        <g>
          <circle cx={learnedX} cy={learnedY} r={11} fill="none" stroke="white" strokeWidth={2} />
          <circle cx={learnedX} cy={learnedY} r={3} fill="white" />
        </g>

        {/* Axis labels */}
        <text x={4} y={14} fontSize="9" fontFamily="var(--font-mono)" fill="white">
          θ_novel ↑
        </text>
        <text x={size - 4} y={size - 4} fontSize="9" fontFamily="var(--font-mono)" fill="white" textAnchor="end">
          w_recency →
        </text>
      </svg>
      <figcaption className="mt-2 text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
        white ring marks the CMA-ES learned (θ_novel, w_recency).
      </figcaption>
    </figure>
  );
}
