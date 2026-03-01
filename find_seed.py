"""Find episode seeds where EpisodicSemanticMemory opens at least 1 door."""
from environment import MultiHopKeyDoor
from agent import ExplorationPolicy
from memory.episodic_semantic_memory import EpisodicSemanticMemory
from agent.loop import run_episode_with_any_memory

found = []
for seed in range(0, 300):
    env = MultiHopKeyDoor(seed=seed)
    policy = ExplorationPolicy(seed=seed)
    mem = EpisodicSemanticMemory(episodic_size=30)
    success, events, stats = run_episode_with_any_memory(env, policy, mem, episode_seed=seed, k=8)
    score = stats.get("partial_score", stats.get("reward", 0))
    if score > 0:
        found.append((seed, score))
        print(f"FOUND seed={seed}  score={score:.3f}  mem={stats['memory_size']}")

print(f"\nTotal seeds with score > 0: {len(found)} / 300")
if found:
    print("Best seeds:", sorted(found, key=lambda x: -x[1])[:5])
