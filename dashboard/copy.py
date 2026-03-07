"""
Narrative copy for the thesis dashboard — story-led sections and key takeaways.
"""

# --- The Question ---
QUESTION_HOOK = (
    "Most AI agents learn a **policy** and a **value function**, but not **how to build their own memory**. "
    "In practice, memory is fixed: context windows, RAG indexes, vector DBs. "
    "Different tasks need different memory structures — what to store, which concepts to track, how much to retain."
)
QUESTION_CLAIM = (
    "**The thesis claim:** Memory construction — what to store, which concepts to track, how to chain events — "
    "should itself be **learned** and should be **task-adaptive**."
)
QUESTION_ONE_LINE = (
    "We optimize **θ** (memory construction parameters), not the policy — meta-level learning over memory representation."
)

# --- The Approach ---
APPROACH_INTRO = (
    "We use a **parameterized graph memory**: events and entities form a graph; a vector **θ** controls "
    "what gets stored, how entities are tracked, and how retrieval is scored. "
    "We optimize θ with **CMA-ES** (black-box search) from task reward alone."
)
APPROACH_BULLETS = [
    "**Graph memory:** Event nodes + entity nodes; temporal and mention edges.",
    "**θ (3D → 10D):** θ_store, θ_entity, θ_temporal (V1); + retrieval weights and importance scoring (V4).",
    "**Optimization:** CMA-ES on θ; objective = mean reward. No gradients, no task-specific hints.",
]

# --- Evidence: Benchmark ---
EVIDENCE_BENCHMARK = (
    "We compare **12 memory systems** on **4 environments**. No single system wins everywhere. "
    "On MultiHopKeyDoor, **retrieval precision** strongly predicts reward — systems that retain the right hints succeed. "
    "Our learned **GraphMemoryV4** ranks first on MultiHop."
)
BENCHMARK_KEY_TAKEAWAY = "GraphMemoryV4 (learned 10D θ) leads on MultiHopKeyDoor; precision ≈ 1.0 gates reward."

# --- Evidence: Ablation ---
EVIDENCE_ABLATION = (
    "**Which parts of θ matter?** We remove each component (set to 0) and measure degradation. "
    "Removing **theta_novel** collapses performance; theta_erich and w_recency are next. "
    "This shows *what* the agent is learning to control."
)
ABLATION_KEY_TAKEAWAY = "Removing theta_novel causes ~100% degradation — novelty-based storage is the load-bearing pillar."

# --- Evidence: Sensitivity ---
EVIDENCE_SENSITIVITY = (
    "How sensitive is reward to small changes in θ? We sweep **theta_novel** and **w_recency** (other θ fixed at learned values). "
    "A **broad plateau** means the system is robust; a sharp peak would mean the optimizer must find precise values."
)
SENSITIVITY_KEY_TAKEAWAY = "The reward landscape is a broad plateau — learned θ is robust."

# --- Task-dependent memory (Transfer) ---
TRANSFER_NARRATIVE = (
    "When we take **θ learned on MultiHop** and evaluate on other environments **without retraining**: "
    "GoalRoom transfers well, HardKeyDoor partially, MegaQuestRoom fails completely. "
    "**Same θ, different task → different outcome.** Memory construction should be task-adaptive."
)
TRANSFER_KEY_TAKEAWAY = "Zero-shot transfer fails on out-of-distribution tasks — task-dependent memory."

# --- Neural meta-controller ---
NEURAL_NARRATIVE = (
    "A **neural network** that outputs θ per observation (NeuralControllerV2Small) can **match scalar V4** on MultiHop "
    "with enough training (200 generations, sigma 0.3). Transfer to MegaQuest still fails. "
    "**Expressivity vs trainability:** the neural controller is more expressive but costlier to train."
)
NEURAL_KEY_TAKEAWAY = "Neural meta-controller matches V4 on MultiHop with 200-gen budget; transfer fails (task-dependent)."

# --- Compare & explore ---
COMPARE_INTRO = (
    "Explore the full benchmark: filter by **environment** and **systems**, see rankings and precision–reward scatter. "
    "Data from benchmark_results.json; V4/V1 from graphmemory_v4_cmaes when available."
)

# --- Takeaways / What's next ---
TAKEAWAYS = [
    "**θ is task-dependent** — optimal memory construction differs across environments.",
    "**Learned θ beats fixed** — CMA-ES on 10D θ yields GraphMemoryV4 #1 on MultiHop.",
    "**Neural can match with budget** — 200-gen run reaches V4 level; transfer still fails.",
    "**Next step:** DocumentQA + LLM, optimize J = score − λ×cost; scale to real token cost.",
]
WHATS_NEXT = (
    "**Scaling to real tasks:** DocumentQA with GPT-4o, optimize QA score minus λ×API cost. "
    "Long lore, multi-session dialogue, and beyond-context tasks are the target applications. "
    "See docs/RUNNING_EXPERIMENTS.md for commands."
)

# --- Empty states (include copyable command) ---
def empty_state(section: str, command: str) -> str:
    return (
        f"This section will come to life when you run the experiment. "
        f"Run: `{command}`"
    )

COMMANDS = {
    "benchmark": "python run_benchmark.py",
    "ablation": "python run_ablation.py",
    "transfer": "python run_transfer.py",
    "sensitivity": "python run_sensitivity.py",
    "neural": "python run_neural_controller_v2.py",
    "document_qa": "python run_document_qa_memory.py",
}
