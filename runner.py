"""
CLI experiment runner — execute any experiment from a config file.

Usage:
    python runner.py --config experiments/multihop_cmaes.yaml
    python runner.py --config experiments/document_qa.json
    python runner.py --preset benchmark --n_episodes 50
    python runner.py --list-runs
    python runner.py --compare MultiHopKeyDoor

The runner:
  1. Loads the config (YAML or JSON).
  2. Instantiates environment, memory system, optimizer, and evaluator.
  3. Runs the experiment (optimization + evaluation).
  4. Saves results to the SQLite database.
  5. Generates figures if requested.
  6. Prints a summary report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thesis experiment runner — learnable memory construction"
    )
    parser.add_argument("--config", type=str, help="Path to experiment config (YAML or JSON)")
    parser.add_argument("--preset", type=str, choices=["benchmark", "multihop_cmaes", "docqa"],
                        help="Use a preset config")
    parser.add_argument("--n_episodes", type=int, default=None, help="Override n_episodes")
    parser.add_argument("--n_generations", type=int, default=None, help="Override n_generations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--list-runs", action="store_true", help="List recent experiment runs")
    parser.add_argument("--compare", type=str, metavar="ENV_NAME",
                        help="Compare all runs for an environment")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    from config import (ExperimentConfig, make_multihop_cmaes_config,
                        make_documentary_qa_config, make_benchmark_config)

    if args.config:
        path = Path(args.config)
        if path.suffix in (".yaml", ".yml"):
            cfg = ExperimentConfig.from_yaml(path)
        else:
            cfg = ExperimentConfig.from_json(path)
    elif args.preset == "benchmark":
        cfg = make_benchmark_config(n_episodes=args.n_episodes or 50)
    elif args.preset == "multihop_cmaes":
        cfg = make_multihop_cmaes_config(
            n_generations=args.n_generations or 20,
            n_episodes=args.n_episodes or 40,
        )
    elif args.preset == "docqa":
        cfg = make_documentary_qa_config()
    else:
        print("ERROR: Provide --config or --preset")
        sys.exit(1)

    # Apply overrides
    if args.n_episodes is not None:
        cfg.eval.n_episodes = args.n_episodes
    if args.n_generations is not None:
        cfg.optimization.n_generations = args.n_generations
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_figures:
        cfg.eval.generate_figures = False
    cfg.output_dir = args.output

    return cfg


def instantiate_env(cfg):
    from environment import (ToyEnvironment, GoalRoom, HardKeyDoor,
                              MultiHopKeyDoor, QuestRoom)
    from environment.mega_quest import MegaQuestRoom
    from environment.document_qa import DocumentQA
    from environment.multi_session import MultiSessionEnv

    env_map = {
        "ToyEnvironment": ToyEnvironment,
        "GoalRoom": GoalRoom,
        "HardKeyDoor": HardKeyDoor,
        "MultiHopKeyDoor": MultiHopKeyDoor,
        "QuestRoom": QuestRoom,
        "MegaQuestRoom": MegaQuestRoom,
        "DocumentQA": lambda seed: DocumentQA(cfg.environment.document_name, seed),
        "MultiSession": lambda seed: MultiSessionEnv(cfg.environment.n_sessions, seed=seed),
    }
    name = cfg.environment.name
    if name not in env_map:
        raise ValueError(f"Unknown environment: {name}")
    env_cls = env_map[name]
    if callable(env_cls) and not isinstance(env_cls, type):
        return env_cls(cfg.environment.seed)
    return env_cls(seed=cfg.environment.seed)


def instantiate_memory(cfg, theta: tuple | None = None):
    from memory.graph_memory import GraphMemory, MemoryParams
    from memory.flat_memory import FlatMemory
    from memory.semantic_memory import SemanticMemory
    from memory.summary_memory import SummaryMemory
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from memory.rag_memory import RAGMemory
    from memory.hierarchical_memory import HierarchicalMemory
    from memory.working_memory import WorkingMemory
    from memory.causal_memory import CausalMemory
    from memory.attention_memory import AttentionMemory

    t = theta or tuple(cfg.memory.theta)
    sys = cfg.memory.system

    mem_map = {
        "GraphMemory": lambda: GraphMemory(MemoryParams(*t, "learnable")),
        "FlatMemory": lambda: FlatMemory(cfg.memory.window_size),
        "SemanticMemory": lambda: SemanticMemory(),
        "SummaryMemory": lambda: SummaryMemory(),
        "EpisodicSemanticMemory": lambda: EpisodicSemanticMemory(cfg.memory.episodic_size),
        "RAGMemory": lambda: RAGMemory(),
        "HierarchicalMemory": lambda: HierarchicalMemory(),
        "WorkingMemory": lambda: WorkingMemory(cfg.memory.capacity),
        "CausalMemory": lambda: CausalMemory(),
        "AttentionMemory": lambda: AttentionMemory(cfg.memory.temperature),
    }
    if sys not in mem_map:
        raise ValueError(f"Unknown memory system: {sys}")
    return mem_map[sys]()


def run_optimization(cfg, env, policy):
    """Run the configured optimization method. Returns (best_theta, history)."""
    method = cfg.optimization.method
    opt_cfg = cfg.optimization
    n = 3  # theta dimensions for scalar theta

    def eval_fn(theta):
        from memory.graph_memory import GraphMemory, MemoryParams
        from agent.loop import run_episode_with_any_memory
        import numpy as np
        theta = np.clip(theta, 0, 1)
        rewards = []
        for ep in range(opt_cfg.n_episodes_per_candidate):
            mem = GraphMemory(MemoryParams(*theta[:3], "learnable"))
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=cfg.eval.k, episode_seed=ep
            )
            rewards.append(stats.get("reward", 0.0))
        return float(sum(rewards) / len(rewards))

    if method == "none":
        return tuple(cfg.memory.theta), []
    elif method == "cmaes":
        from optimization.cma_es import run_cmaes_optimization
        best, history = run_cmaes_optimization(
            eval_fn, n_params=n,
            n_generations=opt_cfg.n_generations,
            sigma=opt_cfg.sigma,
            verbose=cfg.eval.n_episodes > 0,
        )
        return tuple(best[:3]), history
    elif method == "bayesian":
        from optimization.bayesian_opt import run_bayesian_optimization
        best, history = run_bayesian_optimization(
            eval_fn, n_params=n,
            n_trials=opt_cfg.n_trials,
            n_random_init=opt_cfg.n_random_init,
            verbose=True,
        )
        return tuple(best[:3]), history
    elif method == "es":
        # Use existing main.py ES
        import numpy as np
        from memory.graph_memory import GraphMemory, MemoryParams
        from agent.loop import run_episode_with_any_memory
        mu = np.array([0.5, 0.1, 0.8])
        sigma = opt_cfg.sigma
        best_theta = tuple(mu)
        best_reward = -1e9
        history = []
        for gen in range(opt_cfg.n_generations):
            candidates = [np.clip(mu + sigma * np.random.randn(n), 0, 1)
                          for _ in range(opt_cfg.n_candidates)]
            rewards = [eval_fn(c) for c in candidates]
            best_idx = int(np.argmax(rewards))
            if rewards[best_idx] > best_reward:
                best_reward = rewards[best_idx]
                best_theta = tuple(candidates[best_idx].tolist())
            mu = candidates[best_idx]
            sigma = max(0.05, sigma * 0.9)
            history.append({"generation": gen + 1, "best_fitness": best_reward, "mean": mu.tolist()})
            print(f"  ES gen {gen+1}: reward={best_reward:.4f}")
        return best_theta, history
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def main() -> None:
    args = _parse_args()

    # Handle query commands
    if args.list_runs or args.compare:
        from results.db import ResultsDB
        db = ResultsDB()
        if args.list_runs:
            runs = db.list_runs(n=20)
            print(f"\nRecent runs ({len(runs)}):")
            for r in runs:
                print(f"  {r['id'][:8]} | {r['name']:<20} | {r['env_name']:<16} | {r['memory_system']}")
        if args.compare:
            systems = db.compare_systems(args.compare)
            print(f"\nSystem comparison for {args.compare}:")
            for s in systems:
                theta = json.loads(s.get("theta", "[]"))
                print(f"  {s['memory_system']:<22} reward={s['metric_value']:.4f} theta={theta}")
        db.close()
        return

    cfg = load_config(args)
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"Environment: {cfg.environment.name}")
    print(f"Memory: {cfg.memory.system}")
    print(f"Optimization: {cfg.optimization.method}")
    print(f"{'='*60}")

    # Instantiate components
    env = instantiate_env(cfg)

    if cfg.llm.enabled:
        from agent.llm_agent import LLMAgent
        from agent.context_formatter import FormatStyle
        style = FormatStyle(cfg.llm.format_style)
        policy = LLMAgent(model=cfg.llm.model, format_style=style)
    else:
        from agent import ExplorationPolicy
        policy = ExplorationPolicy(seed=cfg.seed)

    t0 = time.time()

    # DocumentQA + LLM path: skip grid-world optimization, run DocumentQA episodes with LLM.
    # Keep rewards/tokens/sizes defined in both branches so DB save never sees undefined vars.
    docqa_llm = cfg.environment.name == "DocumentQA" and cfg.llm.enabled
    rewards, tokens, sizes = [], [], []

    if docqa_llm:
        print("\nPhase: DocumentQA + LLM evaluation (no theta optimization)")
        from agent.loop import run_document_qa_episode_with_llm
        lambda_cost = getattr(cfg.llm, "lambda_cost", 1000.0)
        costs = []
        for ep in range(cfg.eval.n_episodes):
            mem = instantiate_memory(cfg)
            _, score, cost_usd, stats = run_document_qa_episode_with_llm(
                env, mem, policy, k=cfg.eval.k, episode_seed=cfg.seed + ep
            )
            rewards.append(score)
            costs.append(cost_usd)
            tokens.append(stats.get("retrieval_tokens", 0))
            sizes.append(stats.get("memory_size", 0))
        import statistics as st
        mean_score = st.mean(rewards)
        mean_cost = st.mean(costs)
        J = mean_score - lambda_cost * mean_cost
        metrics = {
            "mean_reward": mean_score,
            "std_reward": st.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_cost_usd": mean_cost,
            "mean_tokens": st.mean(tokens),
            "mean_memory_size": st.mean(sizes),
            "J_score_minus_lambda_cost": J,
            "lambda_cost": lambda_cost,
        }
        best_theta = tuple(cfg.memory.theta)
        opt_history = []
    else:
        costs = []
        # Run optimization
        print(f"\nPhase 1: Optimization ({cfg.optimization.method})")
        best_theta, opt_history = run_optimization(cfg, env, policy)
        print(f"Best theta: {[round(x, 3) for x in best_theta]}")

        # Evaluation
        print(f"\nPhase 2: Evaluation ({cfg.eval.n_episodes} episodes)")
        from agent.loop import run_episode_with_any_memory
        rewards, tokens, sizes, precisions = [], [], [], []
        for ep in range(cfg.eval.n_episodes):
            mem = instantiate_memory(cfg, theta=best_theta)
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=cfg.eval.k, episode_seed=ep
            )
            rewards.append(stats.get("reward", 0.0))
            tokens.append(stats.get("retrieval_tokens", 0))
            sizes.append(stats.get("memory_size", 0))
            prec = stats.get("retrieval_precision")
            if prec is not None:
                precisions.append(prec)

        import statistics as st
        metrics = {
            "mean_reward": st.mean(rewards),
            "std_reward": st.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_tokens": st.mean(tokens),
            "mean_memory_size": st.mean(sizes),
            "mean_precision": st.mean(precisions) if precisions else -1.0,
            "efficiency": st.mean(rewards) / (1 + st.mean(tokens)),
        }

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save to database
    from results.db import ResultsDB
    import hashlib
    run_id = f"{cfg.config_hash()}_{int(time.time())}"
    episode_records = [
        {"reward": r, "retrieval_tokens": t, "memory_size": s}
        for r, t, s in zip(rewards, tokens, sizes)
    ]
    db = ResultsDB(Path(cfg.output_dir) / "thesis.db")
    db.save_run(
        run_id=run_id,
        name=cfg.name,
        env_name=cfg.environment.name,
        memory_system=cfg.memory.system,
        theta=best_theta,
        config=cfg.to_dict(),
        metrics=metrics,
        episode_records=episode_records,
        notes=cfg.notes,
    )
    db.close()
    print(f"Results saved to database (run_id={run_id})")

    # Save config
    out_dir = Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")
    import json as _json
    (out_dir / "metrics.json").write_text(_json.dumps(metrics, indent=2))
    (out_dir / "opt_history.json").write_text(_json.dumps(opt_history, indent=2, default=str))
    print(f"Config and metrics saved to {out_dir}")


if __name__ == "__main__":
    main()
