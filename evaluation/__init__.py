"""Evaluation utilities for the memory system."""

from .run import run_evaluation, run_memory_comparison
from .ablation import run_ablation_study, print_ablation_report, get_ablation_configs
from .transfer import run_transfer_matrix, print_transfer_matrix, evaluate_theta_on_task
from .sensitivity import compute_sensitivity, run_multi_env_sensitivity, analyze_landscape
from .statistics import (
    bootstrap_ci, paired_ttest, cohens_d,
    full_comparison, print_comparison_report, run_all_comparisons,
)
from .benchmark import run_full_benchmark, print_benchmark_table, save_benchmark_results
from .document_qa_memory import (
    run_document_qa_memory_eval,
    print_document_qa_table,
    save_document_qa_results,
)
from .cost_tracker import CostTracker, compare_costs

__all__ = [
    "run_evaluation",
    "run_memory_comparison",
    "run_ablation_study",
    "print_ablation_report",
    "get_ablation_configs",
    "run_transfer_matrix",
    "print_transfer_matrix",
    "evaluate_theta_on_task",
    "compute_sensitivity",
    "run_multi_env_sensitivity",
    "analyze_landscape",
    "bootstrap_ci",
    "paired_ttest",
    "cohens_d",
    "full_comparison",
    "print_comparison_report",
    "run_all_comparisons",
    "run_full_benchmark",
    "print_benchmark_table",
    "save_benchmark_results",
    "run_document_qa_memory_eval",
    "print_document_qa_table",
    "save_document_qa_results",
    "CostTracker",
    "compare_costs",
]
