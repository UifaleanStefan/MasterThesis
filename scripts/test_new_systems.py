"""Quick functional test for Phase A memory systems and envs (dev utility)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import MultiHopKeyDoor
from agent import ExplorationPolicy
from agent.loop import run_episode_with_any_memory
from memory.hierarchical_memory import HierarchicalMemory
from memory.working_memory import WorkingMemory
from memory.causal_memory import CausalMemory
from memory.attention_memory import AttentionMemory

env = MultiHopKeyDoor(seed=36)
policy = ExplorationPolicy(seed=36)

systems = {
    "HierarchicalMemory": HierarchicalMemory(),
    "WorkingMemory": WorkingMemory(capacity=7),
    "CausalMemory": CausalMemory(),
    "AttentionMemory": AttentionMemory(),
}

for name, mem in systems.items():
    success, events, stats = run_episode_with_any_memory(env, policy, mem, episode_seed=36, k=8)
    r = stats["reward"]
    ms = stats["memory_size"]
    tok = stats["retrieval_tokens"]
    print(f"{name}: reward={r:.3f} mem_size={ms} tokens={tok}")

print("\nAll Phase A memory systems working!")

# Quick CMA-ES test
from optimization.cma_es import CMAES
opt = CMAES(n_params=3, sigma=0.3, seed=42)
candidates = opt.ask()
fitnesses = [0.5 + 0.1 * i for i, _ in enumerate(candidates)]
opt.tell(candidates, fitnesses)
print(f"CMA-ES: gen={opt.generation}, best={opt.best_fitness:.3f}, sigma={opt.sigma:.3f}")

# Quick Bayesian opt test
from optimization.bayesian_opt import BayesianOptimizer
bo = BayesianOptimizer(n_params=3, n_random_init=3, seed=42)
for i in range(5):
    theta = bo.suggest()
    fitness = float(-((theta[0]-0.8)**2 + (theta[1]-0.1)**2))
    bo.update(theta, fitness)
print(f"BayesianOpt: best_fitness={bo.best_fitness:.3f}, best_theta={[round(x,3) for x in bo.best_theta]}")

# Quick DocumentQA test
from environment.document_qa import DocumentQA
env_qa = DocumentQA("fantasy_lore", seed=0)
obs = env_qa.reset()
print(f"\nDocumentQA reset: {obs[:80]}...")
obs2, done, _ = env_qa.step("next")
print(f"Step 1: {obs2[:80]}...")

# MegaQuestRoom test
from environment.mega_quest import MegaQuestRoom
mega = MegaQuestRoom(seed=42)
obs = mega.reset()
print(f"\nMegaQuestRoom reset: {obs[:100]}...")
obs2, done, success = mega.step("move_north")
print(f"Step 1: done={done}, partial_score={mega.partial_score:.2f}")

print("\nAll systems functional!")
