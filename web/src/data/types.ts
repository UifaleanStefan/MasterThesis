/**
 * TypeScript types for every JSON shipped to /data/.
 *
 * These mirror the Python writers in run_*.py + evaluation/*.py + the
 * scripts/build_web_data.py slimmer. The "_manifest" sibling block is
 * present on every file (added by results.manifest.build_manifest).
 */

export interface Manifest {
  git_sha?: string;
  git_dirty?: boolean;
  timestamp_utc?: string;
  python_version?: string;
  platform?: string;
  numpy_version?: string;
  scipy_version?: string;
  embedding_backend?: string;
  seed?: number;
  experiment?: string;
  [key: string]: unknown;
}

/** /data/manifest.json — aggregated provenance across all data files. */
export interface AggregatedManifest {
  built_at_utc: string;
  embedding_backends: string[];
  git_shas: string[];
  latest_result_timestamp_utc: string | null;
  files_present: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark
// ─────────────────────────────────────────────────────────────────────────────

export interface SystemResult {
  mean_reward: number;
  std_reward?: number;
  ci_lower?: number;
  ci_upper?: number;
  mean_tokens?: number;
  mean_memory_size?: number;
  efficiency?: number;
  retrieval_precision?: number | null;
  n_episodes?: number;
  rewards?: number[];
  error?: string;
}

export type EnvName =
  | "Key-Door"
  | "Goal-Room"
  | "MultiHop-KeyDoor"
  | "MegaQuestRoom";

export type BenchmarkData = {
  [K in EnvName]?: { [system: string]: SystemResult };
} & {
  _manifest?: Manifest;
};

// ─────────────────────────────────────────────────────────────────────────────
// V4 CMA-ES (the headline experiment)
// ─────────────────────────────────────────────────────────────────────────────

export interface V4Theta {
  theta_store: number;
  theta_novel: number;
  theta_erich: number;
  theta_surprise: number;
  theta_entity: number;
  theta_temporal: number;
  theta_decay: number;
  w_graph: number;
  w_embed: number;
  w_recency: number;
}

export interface V4OptHistoryEntry {
  generation: number;
  best_fitness: number;
  mean: number[];
  sigma: number;
}

export interface V4Eval {
  mean_reward: number;
  std_reward: number;
  mean_precision: number;
  mean_memory_size: number;
  mean_tokens: number;
  efficiency: number;
  n_episodes: number;
}

export interface V4CmaesData {
  experiment: string;
  config: {
    n_generations: number;
    n_episodes_per_candidate: number;
    n_eval_episodes: number;
    k: number;
    sigma: number;
    seed: number;
  };
  v4: {
    best_theta_normalized: number[];
    best_params: V4Theta;
    opt_history: V4OptHistoryEntry[];
    eval: V4Eval;
  };
  v1_baseline: null | {
    best_theta: number[];
    opt_history: V4OptHistoryEntry[];
    eval: V4Eval;
  };
  elapsed_optimization_s: number;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ablation
// ─────────────────────────────────────────────────────────────────────────────

export interface AblationConfigResult {
  description: string;
  mean_reward: number;
  std_reward: number;
  mean_tokens: number;
  mean_memory_size: number;
  mean_precision: number | null;
  rewards?: number[];
  degradation?: number;
  w_graph?: number; // present on w_graph sweep entries
}

export interface AblationData {
  experiment: string;
  config: { n_episodes: number; k: number; seed_offset: number; environment: string };
  learned_params: V4Theta;
  results: { [name: string]: AblationConfigResult };
  elapsed_s: number;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transfer
// ─────────────────────────────────────────────────────────────────────────────

export interface TransferEnvResult {
  mean_reward: number;
  std_reward?: number;
  mean_precision?: number | null;
  mean_memory_size?: number;
  mean_tokens?: number;
  rewards?: number[];
}

export interface TransferData {
  experiment: string;
  config: { n_episodes: number; k: number; seed: number };
  multihop_v4_params: V4Theta;
  matrix: { [source: string]: { [target: string]: TransferEnvResult } };
  elapsed_s: number;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensitivity (2D landscape)
// ─────────────────────────────────────────────────────────────────────────────

export interface SensitivityData {
  experiment: string;
  config: {
    dim1: string;
    dim2: string;
    dim1_range: [number, number];
    dim2_range: [number, number];
    resolution: number;
    n_episodes_per_cell: number;
    environment: string;
  };
  dim1_values: number[];
  dim2_values: number[];
  reward_grid: number[][];
  precision_grid: number[][];
  best_params_dict: V4Theta;
  best_reward: number;
  learned_dim1: number;
  learned_dim2: number;
  analysis: {
    is_sharp_peak: boolean;
    top_10_mean: number;
    top_10_std: number;
    [k: string]: unknown;
  };
  elapsed_s: number;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural V2 (slimmed: no .mean[5,674])
// ─────────────────────────────────────────────────────────────────────────────

export interface NeuralV2HistoryEntry {
  generation: number;
  best_fitness: number;
  sigma: number;
}

export interface NeuralV2Data {
  experiment: string;
  config: {
    architecture: string;
    n_params: number;
    n_generations: number;
    n_episodes_per_candidate: number;
    n_eval_episodes: number;
    sigma: number;
    seed: number;
    train_env: string;
  };
  training: {
    best_fitness: number;
    elapsed_s: number;
    history: NeuralV2HistoryEntry[];
  };
  eval_multihop: V4Eval;
  eval_megaquest: V4Eval;
  v4_scalar_comparison: { mean_reward: number; mean_precision: number } | null;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pairwise significance
// ─────────────────────────────────────────────────────────────────────────────

export interface PairwiseEntry {
  point_estimate: number;
  ci_lower: number;
  ci_upper: number;
  ci_width: number;
  n: number;
  alpha: number;
}

export interface PairwiseComparison {
  ttest: {
    t_statistic: number;
    p_value: number;
    df: number;
    mean_diff: number;
    significant: boolean;
    n: number;
  };
  cohens_d: { d: number; magnitude: string; mean_a: number; mean_b: number };
  improvement: number;
  improvement_pct: number;
  conclusion: string;
  [systemLabel: string]: PairwiseEntry | unknown;
}

export interface PairwiseData {
  env: string;
  baseline: string;
  pairwise: { [system: string]: PairwiseComparison };
  skipped: string[];
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// DocumentQA recall
// ─────────────────────────────────────────────────────────────────────────────

export interface DocQAData {
  [system: string]: {
    mean_recall: number;
    std_recall: number;
    n_questions: number;
    per_question_recalls?: number[];
  } | Manifest | unknown;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiSession
// ─────────────────────────────────────────────────────────────────────────────

export interface MultiSessionData {
  results: {
    [system: string]: {
      mean_score: number;
      std_score: number;
      scores: number[];
      mean_memory_size: number;
      mean_n_steps: number;
      n_trials: number;
    };
  };
  elapsed_s: number;
  _manifest?: Manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// w_graph sweep (S3 ablation variant)
// ─────────────────────────────────────────────────────────────────────────────

export interface WGraphSweepData extends AblationData {
  /** Same shape as AblationData but each entry includes a `w_graph` field. */
}
