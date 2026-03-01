"""
MetaLearner — MAML-inspired cross-task theta initialization.

Standard optimization finds the best theta for ONE task. MetaLearner finds a theta_init
that is a good *starting point* for ANY task in a distribution — after just 3-5 ES
adaptation steps on a new task, the meta-initialized theta reaches near-optimal performance.

This is the Model-Agnostic Meta-Learning (MAML) idea applied to memory construction:
  - Inner loop: adapt theta for a specific task (K ES steps).
  - Outer loop: update theta_init to minimize expected loss after inner loop adaptation.

Algorithm:
    theta_meta = mean initialization (e.g., [0.5, 0.1, 0.8])
    for each outer iteration:
        sample a batch of tasks T = {task_1, ..., task_B}
        for each task t in T:
            theta_t = run ES for K_inner steps starting from theta_meta
            evaluate J(theta_t) on task t
        update theta_meta toward the average best theta across tasks
        (gradient of outer loss = average direction from theta_meta to theta_t)

In practice (since our ES is black-box, not gradient-based), we implement this as:
    outer loop: sample B tasks → run inner ES → average the best thetas → update theta_meta

This is the "Reptile" algorithm (Nichol et al. 2018), which is simpler than MAML
but achieves similar results for black-box optimization.

Usage:
    meta = MetaLearner(task_factory=lambda seed: make_env(seed), eval_fn=...)
    theta_meta = meta.train(n_outer=10, n_tasks_per_outer=5, n_inner_steps=5)
    # Now fine-tune on a new task:
    theta_final = meta.adapt(new_task_eval_fn, n_steps=5)

Thesis motivation: if theta_meta generalizes across tasks, it means there is a
universal "good memory structure" — a powerful finding. If it doesn't generalize,
that confirms the thesis claim that memory structure MUST be task-specific.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .cma_es import CMAES


class MetaLearner:
    """
    Reptile-style meta-learner for theta initialization across a task distribution.

    Parameters
    ----------
    task_factory : callable(seed) -> (eval_fn: callable(theta) -> float)
        Factory that creates task-specific evaluation functions.
    n_params : int
        Dimensionality of theta (3 for scalar theta, ~4000 for NeuralController).
    meta_lr : float
        Outer loop learning rate (step size toward average adapted theta).
    inner_sigma : float
        ES sigma for inner-loop adaptation.
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        task_factory: Callable[[int], Callable[[np.ndarray], float]],
        n_params: int = 3,
        meta_lr: float = 0.3,
        inner_sigma: float = 0.2,
        seed: int = 0,
    ) -> None:
        self._task_factory = task_factory
        self._n = n_params
        self._meta_lr = meta_lr
        self._inner_sigma = inner_sigma
        self._rng = np.random.RandomState(seed)

        # Meta-initialization
        self.theta_meta = np.full(n_params, 0.5, dtype=np.float64)
        if n_params == 3:
            self.theta_meta = np.array([0.5, 0.1, 0.8], dtype=np.float64)

        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        n_outer: int = 10,
        n_tasks_per_outer: int = 5,
        n_inner_steps: int = 5,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Run Reptile meta-training.

        Returns
        -------
        theta_meta : np.ndarray
            The meta-initialized theta after training.
        """
        for outer in range(n_outer):
            task_seeds = [self._rng.randint(0, 10000) for _ in range(n_tasks_per_outer)]
            adapted_thetas: list[np.ndarray] = []
            task_rewards: list[float] = []

            for t_seed in task_seeds:
                eval_fn = self._task_factory(t_seed)
                theta_adapted, best_reward = self._inner_adapt(eval_fn, n_inner_steps)
                adapted_thetas.append(theta_adapted)
                task_rewards.append(best_reward)

            # Reptile update: move theta_meta toward the average adapted theta
            avg_adapted = np.mean(adapted_thetas, axis=0)
            self.theta_meta = self.theta_meta + self._meta_lr * (avg_adapted - self.theta_meta)
            self.theta_meta = np.clip(self.theta_meta, 0.0, 1.0)

            record = {
                "outer": outer + 1,
                "mean_task_reward": float(np.mean(task_rewards)),
                "theta_meta": self.theta_meta.tolist(),
            }
            self._history.append(record)

            if verbose:
                print(
                    f"  Meta outer {outer+1:3d}/{n_outer} | "
                    f"mean_reward={record['mean_task_reward']:.4f} | "
                    f"theta_meta={[round(x, 3) for x in self.theta_meta]}"
                )

        return self.theta_meta.copy()

    def adapt(
        self,
        eval_fn: Callable[[np.ndarray], float],
        n_steps: int = 5,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Fast-adapt theta_meta to a specific task using n_steps of inner ES.
        Returns the adapted theta.
        """
        theta_adapted, _ = self._inner_adapt(eval_fn, n_steps, verbose=verbose)
        return theta_adapted

    @property
    def history(self) -> list[dict]:
        return self._history.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inner_adapt(
        self,
        eval_fn: Callable[[np.ndarray], float],
        n_steps: int,
        verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Run CMA-ES inner adaptation starting from theta_meta."""
        optimizer = CMAES(
            n_params=self._n,
            mean=self.theta_meta.copy(),
            sigma=self._inner_sigma,
            seed=int(self._rng.randint(0, 10000)),
            clip_to_unit=(self._n <= 10),
        )
        for _ in range(n_steps):
            candidates = optimizer.ask()
            fitnesses = [eval_fn(c) for c in candidates]
            optimizer.tell(candidates, fitnesses)

        return optimizer.best_solution, optimizer.best_fitness


def make_task_factory(env_class, policy_class, memory_class, n_episodes: int = 20):
    """
    Convenience: create a task_factory for MetaLearner using existing env/policy/memory classes.

    Parameters
    ----------
    env_class : class
        Environment class (e.g. MultiHopKeyDoor).
    policy_class : class
        Policy class (e.g. ExplorationPolicy).
    memory_class : class
        Memory class accepting theta as constructor arg or MemoryParams.
    n_episodes : int
        Episodes to evaluate each theta configuration.

    Returns
    -------
    task_factory : callable(seed) -> eval_fn
    """
    def task_factory(seed: int):
        from memory.graph_memory import GraphMemory, MemoryParams
        from agent.loop import run_episode_with_any_memory

        env = env_class(seed=seed)
        policy = policy_class(seed=seed)

        def eval_fn(theta: np.ndarray) -> float:
            theta = np.clip(theta, 0.0, 1.0)
            rewards = []
            for ep in range(n_episodes):
                mem = GraphMemory(MemoryParams(float(theta[0]), float(theta[1]), float(theta[2]), "learnable"))
                _, _, stats = run_episode_with_any_memory(env, policy, mem, episode_seed=seed * 100 + ep)
                rewards.append(stats.get("reward", 0.0))
            return float(np.mean(rewards))

        return eval_fn

    return task_factory
