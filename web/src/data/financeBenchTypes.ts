/**
 * Type definitions for the FinanceBench corpus-mode QA frontend data
 * (web/public/data/stage3_finbench_corpus.json).
 */

export interface FinanceBenchConfig {
  name: string;
  label: string;
  family: string;
  color: string;
}

export type FinanceBenchMode = "online" | "batch";

export interface JudgeCell {
  mean_judge: number;
  ci_lower: number;
  ci_upper: number;
  mean_recall: number;
  n: number;
  cost_usd: number;
  in_doc_ratio: number;
}

export interface ThetaRow {
  name: string;
  values: number[];
}

export interface ThetaContrast {
  params: string[];
  rows: ThetaRow[];
  annotations: string[];
}

export interface ScatterPoint {
  config: string;
  label: string;
  family: string;
  color: string;
  mode: FinanceBenchMode;
  in_doc_ratio: number;
  mean_judge: number;
  mean_recall: number;
  ci_lower: number;
  ci_upper: number;
  n: number;
}

export interface QuestionTypes {
  categories: string[];
  category_method: string;
  counts: {
    always_hard: number[];
    always_easy: number[];
    mid: number[];
  };
  category_membership: { [qid: string]: string };
  per_cell_mean_judge: { [cellKey: string]: { [category: string]: number } };
  per_cell_n_by_cat: { [cellKey: string]: { [category: string]: number } };
  always_hard_n: number;
  always_easy_n: number;
  mid_n: number;
}

export interface DrilldownExample {
  doc_idx: number;
  question: string;
  gold_answer: string;
  category: string;
  predictions: {
    [cellKey: string]: {
      predicted: string;
      judge_score: number;
    };
  };
}

export interface FinanceBenchCorpusData {
  _manifest: {
    git_sha: string;
    timestamp_utc: string;
    n_questions: number;
    k: number;
    model: string;
    manual_judge: boolean;
    categorization_method: string;
    cross_doc_bleed_metric: string;
    judge_protocol: string;
  };
  configs: FinanceBenchConfig[];
  modes: FinanceBenchMode[];
  judge_table: { [config: string]: { [mode in FinanceBenchMode]: JudgeCell } };
  theta_contrast: ThetaContrast;
  cross_doc_scatter: ScatterPoint[];
  question_types: QuestionTypes;
  drilldown_examples: {
    always_hard: DrilldownExample[];
    always_easy: DrilldownExample[];
    corpus_tuned_wins_vs_canonical: DrilldownExample[];
    corpus_tuned_regressions_vs_canonical: DrilldownExample[];
  };
}
