/**
 * The TF-IDF era V4 best theta — kept inline as a constant so the
 * EmbeddingToggle can flip between the two empirical optima on the page.
 *
 * Source: results/graphmemory_v4_cmaes_results.json prior to the April 2026
 * post-MiniLM re-run (commit 02e2041 era). Reward 0.178, precision 0.997.
 */

import type { V4Theta } from "./types";

export const TFIDF_V4_THETA: V4Theta = {
  theta_store: 0.293,
  theta_novel: 0.908,
  theta_erich: 0.198,
  theta_surprise: 0.785,
  theta_entity: 0.285,
  theta_temporal: 0.278,
  theta_decay: 0.668,
  w_graph: 0.0,
  w_embed: 1.079,
  w_recency: 3.777,
};

export const TFIDF_V4_REWARD = 0.178;
export const TFIDF_V4_PRECISION = 0.997;
export const TFIDF_V4_MEM_SIZE = 10;

/** Inline summary of TF-IDF era benchmark ranking for narrative overlays. */
export const TFIDF_BENCHMARK_TOP: Array<{ system: string; reward: number; precision: number }> = [
  { system: "GraphMemoryV4", reward: 0.178, precision: 0.997 },
  { system: "EpisodicSemantic", reward: 0.173, precision: 1.0 },
  { system: "WorkingMemory(7)", reward: 0.153, precision: 1.0 },
  { system: "AttentionMemory", reward: 0.153, precision: 1.0 },
  { system: "SemanticMemory", reward: 0.133, precision: 1.0 },
  { system: "HierarchicalMemory", reward: 0.127, precision: 1.0 },
  { system: "CausalMemory", reward: 0.1, precision: 1.0 },
  { system: "RAGMemory", reward: 0.053, precision: 0.482 },
  { system: "GraphMemoryV1", reward: 0.033, precision: 0.578 },
  { system: "FlatWindow(50)", reward: 0.0, precision: 0.028 },
  { system: "SummaryMemory", reward: 0.0, precision: 0.01 },
];
