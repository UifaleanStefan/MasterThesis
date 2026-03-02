"""
NeuralMemoryControllerV2Small training via CMA-ES on MultiHopKeyDoor,
followed by zero-shot evaluation on MegaQuestRoom.

Architecture: 50 -> 32 -> 10 MLP (1,962 params)
  Input : 50-dim (31 TF-IDF + 10 task-agnostic features + 9 reserved zeros)
  Output: 10-dim MemoryParamsV4 per observation (context-dependent theta)

CMA-ES config:
  clip_to_unit=False  (weights are unbounded, unlike scalar theta)
  sigma=0.05          (small perturbations in weight space)
  n_generations=30
  n_episodes_per_candidate=20
  lambda ~ 27 candidates per generation (auto: 4 + floor(3*ln(1962)))

This is the thesis's most ambitious experiment: a neural meta-controller
that decides all 10 memory parameters dynamically per observation, trained
purely from reward on MultiHopKeyDoor, then evaluated zero-shot on a harder
environment (MegaQuestRoom) without any weight update.

Output:
    results/neural_controller_v2_results.json
    docs/figures/fig_neural_v2_curves.png

Usage:
    python run_neural_controller_v2.py
    python run_neural_controller_v2.py --generations 15 --episodes 10  (quick, ~8 min)
    python run_neural_controller_v2.py --no-transfer  (skip MegaQuestRoom eval)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def make_eval_fn(env, policy, controller_factory, n_episodes: int, k: int = 8):
    """Returns eval_fn(weights) -> mean_reward for CMA-ES."""
    from agent.loop import run_episode_with_any_memory

    def eval_fn(w: np.ndarray) -> float:
        ctrl = controller_factory()
        ctrl.set_weights(w)
        rewards = []
        for ep in range(n_episodes):
            ctrl.clear()
            _, _, stats = run_episode_with_any_memory(
                env, policy, ctrl, k=k, episode_seed=ep
            )
            rewards.append(stats.get("reward", 0.0))
        return float(np.mean(rewards))

    return eval_fn


def evaluate_controller(controller, env, policy, n_episodes: int = 100, k: int = 8,
                        seed_offset: int = 1000) -> dict:
    """Full evaluation: reward, precision, memory size on held-out episodes."""
    from agent.loop import run_episode_with_any_memory
    import statistics

    rewards, precisions, sizes, tokens = [], [], [], []
    for ep in range(n_episodes):
        controller.clear()
        _, _, stats = run_episode_with_any_memory(
            env, policy, controller, k=k, episode_seed=seed_offset + ep
        )
        rewards.append(stats.get("reward", 0.0))
        sizes.append(stats.get("memory_size", 0))
        tokens.append(stats.get("retrieval_tokens", 0))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            precisions.append(prec)

    return {
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_precision": statistics.mean(precisions) if precisions else None,
        "mean_memory_size": statistics.mean(sizes),
        "mean_tokens": statistics.mean(tokens),
        "n_episodes": n_episodes,
    }


def plot_learning_curve(history: list[dict], output_path: str) -> None:
    """Plot CMA-ES fitness vs generation for the neural controller."""
    import matplotlib.pyplot as plt
    from pathlib import Path as P

    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    sigmas = [h["sigma"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(gens, best, "b-o", markersize=4, linewidth=1.5, label="Best fitness")
    ax1.set_ylabel("Mean Reward (best candidate)", fontsize=11)
    ax1.set_title("NeuralMemoryControllerV2Small — CMA-ES Training\n"
                  "(50->32->10 MLP, 1,962 params, MultiHopKeyDoor)", fontsize=12)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=9)

    ax2.plot(gens, sigmas, "r-", linewidth=1.5, label="Sigma (step size)")
    ax2.set_ylabel("CMA-ES Sigma", fontsize=11)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    P(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Neural V2 Fig] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NeuralMemoryControllerV2Small CMA-ES Training")
    parser.add_argument("--generations", type=int, default=30,
                        help="CMA-ES generations (default: 30)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per candidate during training (default: 20)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Episodes for final evaluation (default: 100)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Initial CMA-ES sigma (default: 0.05)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-transfer", action="store_true",
                        help="Skip zero-shot MegaQuestRoom evaluation")
    args = parser.parse_args()

    from environment.env import MultiHopKeyDoor
    from environment.mega_quest import MegaQuestRoom
    from agent.policy import ExplorationPolicy
    from memory.neural_controller_v2_small import NeuralMemoryControllerV2Small
    from optimization.cma_es import CMAES

    np.random.seed(args.seed)

    env_train = MultiHopKeyDoor(seed=0)
    env_transfer = MegaQuestRoom(seed=0)
    policy_train = ExplorationPolicy(seed=args.seed)
    policy_transfer = ExplorationPolicy(seed=args.seed)

    # Determine n_params
    dummy = NeuralMemoryControllerV2Small(seed=0)
    n_params = dummy.n_params
    lam = max(4, 4 + int(3 * math.log(n_params)))

    print("=" * 65)
    print("  NeuralMemoryControllerV2Small CMA-ES Training")
    print("=" * 65)
    print(f"  Architecture  : {50}->{32}->{10} MLP")
    print(f"  Parameters    : {n_params:,}")
    print(f"  Optimizer     : CMA-ES (lambda={lam})")
    print(f"  Generations   : {args.generations}")
    print(f"  Episodes/cand : {args.episodes}")
    print(f"  Sigma         : {args.sigma}")
    print(f"  Train env     : MultiHopKeyDoor")
    print(f"  Transfer env  : MegaQuestRoom (zero-shot)")
    est_min = args.generations * lam * args.episodes * 0.288 / 60
    print(f"  Est. runtime  : ~{est_min:.0f} min")
    print("=" * 65)

    # --- CMA-ES training ---
    print(f"\n[Phase 1] CMA-ES training on MultiHopKeyDoor")
    eval_fn = make_eval_fn(
        env_train, policy_train,
        lambda: NeuralMemoryControllerV2Small(seed=0),
        n_episodes=args.episodes,
        k=args.k,
    )

    optimizer = CMAES(
        n_params=n_params,
        sigma=args.sigma,
        seed=args.seed,
        clip_to_unit=False,  # weights are unbounded
    )
    history = []
    t0 = time.time()

    for gen in range(args.generations):
        candidates = optimizer.ask()
        fitnesses = [eval_fn(c) for c in candidates]
        optimizer.tell(candidates, fitnesses)
        summary = optimizer.summary()
        history.append(summary)
        print(f"  CMA-ES gen {gen+1:3d} | best={summary['best_fitness']:.4f} "
              f"| sigma={summary['sigma']:.5f}")

    elapsed_train = time.time() - t0
    best_weights = optimizer.best_solution
    print(f"\n  Training complete in {elapsed_train:.1f}s")
    print(f"  Best training fitness: {optimizer.best_fitness:.4f}")

    # --- Final evaluation on MultiHopKeyDoor (held-out) ---
    print(f"\n[Phase 2] Final evaluation on MultiHopKeyDoor ({args.eval_episodes} held-out episodes)")
    best_ctrl = NeuralMemoryControllerV2Small(seed=0)
    best_ctrl.set_weights(best_weights)
    multihop_eval = evaluate_controller(
        best_ctrl, env_train, policy_train,
        n_episodes=args.eval_episodes, k=args.k, seed_offset=1000,
    )
    print(f"  MultiHop: reward={multihop_eval['mean_reward']:.4f}  "
          f"precision={multihop_eval.get('mean_precision', 0):.4f}  "
          f"mem_size={multihop_eval['mean_memory_size']:.1f}")

    # --- Zero-shot transfer to MegaQuestRoom ---
    megaquest_eval = None
    if not args.no_transfer:
        print(f"\n[Phase 3] Zero-shot transfer to MegaQuestRoom ({args.eval_episodes} episodes)")
        print(f"  (No weight update — same weights as trained on MultiHop)")
        transfer_ctrl = NeuralMemoryControllerV2Small(seed=0)
        transfer_ctrl.set_weights(best_weights)
        megaquest_eval = evaluate_controller(
            transfer_ctrl, env_transfer, policy_transfer,
            n_episodes=args.eval_episodes, k=args.k, seed_offset=1000,
        )
        print(f"  MegaQuest: reward={megaquest_eval['mean_reward']:.4f}  "
              f"precision={megaquest_eval.get('mean_precision', 0):.4f}  "
              f"mem_size={megaquest_eval['mean_memory_size']:.1f}")

    # --- Results report ---
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    # Load V4 scalar baseline for comparison
    try:
        v4_data = json.loads(Path("results/graphmemory_v4_cmaes_results.json").read_text())
        v4_reward = v4_data["v4"]["eval"]["mean_reward"]
        v4_prec = v4_data["v4"]["eval"]["mean_precision"]
        print(f"\n  Comparison (MultiHopKeyDoor, 200 held-out episodes):")
        print(f"    GraphMemoryV4 (scalar theta): reward={v4_reward:.4f}  precision={v4_prec:.4f}")
    except Exception:
        v4_reward, v4_prec = None, None

    print(f"\n  NeuralControllerV2Small (100 held-out episodes):")
    print(f"    MultiHopKeyDoor: reward={multihop_eval['mean_reward']:.4f}  "
          f"precision={multihop_eval.get('mean_precision', 0):.4f}  "
          f"mem_size={multihop_eval['mean_memory_size']:.1f}")
    if v4_reward is not None:
        delta = multihop_eval["mean_reward"] - v4_reward
        print(f"    vs V4 scalar: {delta:+.4f}")

    if megaquest_eval is not None:
        print(f"\n  Zero-shot transfer to MegaQuestRoom:")
        print(f"    MegaQuestRoom: reward={megaquest_eval['mean_reward']:.4f}  "
              f"precision={megaquest_eval.get('mean_precision', 0):.4f}  "
              f"mem_size={megaquest_eval['mean_memory_size']:.1f}")
        ratio = megaquest_eval["mean_reward"] / max(multihop_eval["mean_reward"], 1e-9)
        print(f"    Transfer ratio (MegaQuest/MultiHop): {ratio:.3f}")
        if ratio > 0.7:
            print(f"    -> STRONG TRANSFER: neural controller generalizes well")
        elif ratio > 0.3:
            print(f"    -> PARTIAL TRANSFER: some generalization, task-specific theta still better")
        else:
            print(f"    -> POOR TRANSFER: task-specific theta is necessary (thesis claim confirmed)")

    # --- Figure ---
    plot_learning_curve(history, "docs/figures/fig_neural_v2_curves.png")

    # --- Save JSON ---
    out = {
        "experiment": "neural_controller_v2_small_cmaes",
        "config": {
            "architecture": f"{_INPUT_DIM if True else 50}->{32}->{10}",
            "n_params": n_params,
            "n_generations": args.generations,
            "n_episodes_per_candidate": args.episodes,
            "n_eval_episodes": args.eval_episodes,
            "sigma": args.sigma,
            "seed": args.seed,
            "train_env": "MultiHopKeyDoor",
            "transfer_env": "MegaQuestRoom",
        },
        "training": {
            "best_fitness": float(optimizer.best_fitness),
            "elapsed_s": elapsed_train,
            "history": history,
        },
        "eval_multihop": multihop_eval,
        "eval_megaquest": megaquest_eval,
        "v4_scalar_comparison": {
            "mean_reward": v4_reward,
            "mean_precision": v4_prec,
        } if v4_reward is not None else None,
    }
    out_path = Path("results/neural_controller_v2_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"  Figure saved to docs/figures/fig_neural_v2_curves.png")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


# Fix reference to _INPUT_DIM in the config dict
try:
    from memory.neural_controller_v2 import _INPUT_DIM
except ImportError:
    _INPUT_DIM = 50

if __name__ == "__main__":
    main()
