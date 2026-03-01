"""
Configuration system for thesis experiments.

All experiment parameters are defined in dataclasses (type-safe, IDE-friendly)
and serializable to/from YAML/JSON for full reproducibility.

Every experiment run reads a config, executes, and saves config + results together.
This means any past result can be exactly reproduced by re-running with its config.

Config hierarchy:
  ExperimentConfig
  ├── EnvironmentConfig
  ├── MemoryConfig
  ├── OptimizationConfig
  └── EvalConfig

YAML format example (experiments/multihop_cmaes.yaml):
  experiment:
    name: multihop_cmaes_v1
    seed: 42
  environment:
    name: MultiHopKeyDoor
    seed: 0
  memory:
    system: GraphMemory
    theta: [0.5, 0.1, 0.8]
  optimization:
    method: cmaes
    n_generations: 20
    n_candidates: 10
    n_episodes_per_candidate: 40
    sigma: 0.3
  eval:
    n_episodes: 100
    k: 8
    run_ablation: true
    run_transfer: false
    run_sensitivity: false
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class EnvironmentConfig:
    name: Literal[
        "ToyEnvironment", "GoalRoom", "HardKeyDoor",
        "MultiHopKeyDoor", "QuestRoom", "MegaQuestRoom",
        "TextWorld", "DocumentQA", "MultiSession"
    ] = "MultiHopKeyDoor"
    seed: int = 0
    # Environment-specific params
    difficulty: int = 5              # TextWorld difficulty
    document_name: str = "fantasy_lore"  # DocumentQA document
    n_sessions: int = 20             # MultiSession sessions


@dataclass
class MemoryConfig:
    system: Literal[
        "FlatMemory", "GraphMemory", "SemanticMemory", "SummaryMemory",
        "EpisodicSemanticMemory", "RAGMemory", "HierarchicalMemory",
        "WorkingMemory", "CausalMemory", "AttentionMemory", "NeuralController"
    ] = "GraphMemory"
    theta: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.8])
    window_size: int = 50            # FlatMemory
    capacity: int = 7                # WorkingMemory
    episodic_size: int = 30          # EpisodicSemanticMemory
    temperature: float = 0.5         # AttentionMemory


@dataclass
class OptimizationConfig:
    method: Literal["none", "es", "cmaes", "bayesian", "online", "meta"] = "cmaes"
    n_generations: int = 20
    n_candidates: int = 10           # ES/CMA-ES population size
    n_episodes_per_candidate: int = 40
    sigma: float = 0.3               # Initial step size
    # Bayesian
    n_random_init: int = 5
    n_trials: int = 20
    # Meta-learning
    n_outer: int = 10
    n_tasks_per_outer: int = 5
    n_inner_steps: int = 5
    meta_lr: float = 0.3
    # Online adaptation
    adapt_every: int = 10


@dataclass
class EvalConfig:
    n_episodes: int = 100
    k: int = 8
    run_ablation: bool = False
    n_ablation_episodes: int = 50
    run_transfer: bool = False
    n_transfer_episodes: int = 50
    run_sensitivity: bool = False
    sensitivity_resolution: int = 10
    sensitivity_episodes_per_cell: int = 20
    run_benchmark: bool = False
    run_statistical_tests: bool = True
    bootstrap_resamples: int = 1000
    generate_figures: bool = True


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    format_style: Literal["flat", "structured", "compressed"] = "structured"
    temperature: float = 0.0
    max_tokens: int = 10
    lambda_cost: float = 1000.0      # lambda in J = reward - lambda * cost_usd


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output_dir: str = "results"
    figures_dir: str = "docs/figures"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "experiment"),
            seed=d.get("seed", 42),
            environment=EnvironmentConfig(**d.get("environment", {})),
            memory=MemoryConfig(**d.get("memory", {})),
            optimization=OptimizationConfig(**d.get("optimization", {})),
            eval=EvalConfig(**d.get("eval", {})),
            llm=LLMConfig(**d.get("llm", {})),
            output_dir=d.get("output_dir", "results"),
            figures_dir=d.get("figures_dir", "docs/figures"),
            notes=d.get("notes", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        try:
            import yaml
            data = yaml.safe_load(Path(path).read_text())
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    def config_hash(self) -> str:
        """Short hash of config for deduplication."""
        import hashlib
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]


# ------------------------------------------------------------------
# Preset configs for common experiments
# ------------------------------------------------------------------

def make_multihop_cmaes_config(n_generations: int = 20, n_episodes: int = 40) -> ExperimentConfig:
    return ExperimentConfig(
        name="multihop_cmaes",
        environment=EnvironmentConfig(name="MultiHopKeyDoor"),
        memory=MemoryConfig(system="GraphMemory"),
        optimization=OptimizationConfig(method="cmaes", n_generations=n_generations,
                                        n_episodes_per_candidate=n_episodes),
        eval=EvalConfig(n_episodes=100, run_ablation=True, run_sensitivity=True),
    )


def make_documentary_qa_config(model: str = "gpt-4o-mini") -> ExperimentConfig:
    return ExperimentConfig(
        name="document_qa_llm",
        environment=EnvironmentConfig(name="DocumentQA", document_name="fantasy_lore"),
        memory=MemoryConfig(system="EpisodicSemanticMemory"),
        optimization=OptimizationConfig(method="bayesian", n_trials=15),
        eval=EvalConfig(n_episodes=20, generate_figures=True),
        llm=LLMConfig(enabled=True, model=model),
    )


def make_benchmark_config(n_episodes: int = 50) -> ExperimentConfig:
    return ExperimentConfig(
        name="full_benchmark",
        environment=EnvironmentConfig(name="MultiHopKeyDoor"),
        optimization=OptimizationConfig(method="none"),
        eval=EvalConfig(n_episodes=n_episodes, run_benchmark=True,
                        run_statistical_tests=True, generate_figures=True),
    )
