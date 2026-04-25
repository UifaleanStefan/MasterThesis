/**
 * Bilinear interpolation on a 2D grid — used by the θ explorer to
 * estimate reward at arbitrary (theta_novel, w_recency) values from
 * the sensitivity grid.
 */

import { clamp } from "./format";

export interface Grid2D {
  /** Coordinates along axis 0 (rows). */
  rows: number[];
  /** Coordinates along axis 1 (cols). */
  cols: number[];
  /** values[i][j] at (rows[i], cols[j]). */
  values: number[][];
}

/**
 * Bilinear interpolation of a value at (r, c). Out-of-bounds queries are
 * clamped to the grid edges (no extrapolation).
 */
export function bilinear(grid: Grid2D, r: number, c: number): number {
  const rs = grid.rows;
  const cs = grid.cols;
  if (rs.length === 0 || cs.length === 0) return 0;

  const rClamped = clamp(r, rs[0], rs[rs.length - 1]);
  const cClamped = clamp(c, cs[0], cs[cs.length - 1]);

  // Find bracketing indices on each axis.
  const i0 = _bracket(rs, rClamped);
  const i1 = Math.min(i0 + 1, rs.length - 1);
  const j0 = _bracket(cs, cClamped);
  const j1 = Math.min(j0 + 1, cs.length - 1);

  const tr = rs[i1] === rs[i0] ? 0 : (rClamped - rs[i0]) / (rs[i1] - rs[i0]);
  const tc = cs[j1] === cs[j0] ? 0 : (cClamped - cs[j0]) / (cs[j1] - cs[j0]);

  const v00 = grid.values[i0][j0];
  const v01 = grid.values[i0][j1];
  const v10 = grid.values[i1][j0];
  const v11 = grid.values[i1][j1];

  const top = v00 * (1 - tc) + v01 * tc;
  const bot = v10 * (1 - tc) + v11 * tc;
  return top * (1 - tr) + bot * tr;
}

function _bracket(coords: number[], target: number): number {
  for (let i = 0; i < coords.length - 1; i++) {
    if (coords[i] <= target && target <= coords[i + 1]) return i;
  }
  return coords.length - 2;
}
