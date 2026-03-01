"""
SQLite results database for cross-run comparison and reproducibility.

Every experiment run writes its complete results to a local SQLite database.
This enables:
  - Query best theta per environment: db.best_theta(env="MultiHop")
  - Compare systems across runs: db.compare_systems(env="MultiHop")
  - List all experiments: db.list_runs()
  - Retrieve raw rewards for statistical tests: db.get_rewards(run_id=...)
  - Reproduce any past run: db.get_config(run_id=...)

Schema:
  runs(
    id TEXT PRIMARY KEY,           -- config hash + timestamp
    name TEXT,
    env_name TEXT,
    memory_system TEXT,
    theta TEXT,                    -- JSON list
    config TEXT,                   -- full config JSON
    timestamp REAL,
    notes TEXT
  )
  results(
    run_id TEXT,
    metric TEXT,                   -- e.g. "mean_reward", "retrieval_precision"
    value REAL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
  )
  episodes(
    run_id TEXT,
    episode INT,
    reward REAL,
    tokens REAL,
    memory_size REAL,
    retrieval_precision REAL
  )

Usage:
    db = ResultsDB("results/thesis.db")
    db.save_run(config, results, episode_records)
    best = db.best_theta(env_name="MultiHopKeyDoor")
    df = db.compare_systems(env_name="MultiHopKeyDoor")
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class ResultsDB:
    """SQLite-backed results store for thesis experiments."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        name TEXT,
        env_name TEXT,
        memory_system TEXT,
        theta TEXT,
        config TEXT,
        timestamp REAL,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS results (
        run_id TEXT,
        metric TEXT,
        value REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    CREATE TABLE IF NOT EXISTS episodes (
        run_id TEXT,
        episode INTEGER,
        reward REAL,
        tokens REAL,
        memory_size REAL,
        retrieval_precision REAL
    );
    CREATE INDEX IF NOT EXISTS idx_runs_env ON runs(env_name);
    CREATE INDEX IF NOT EXISTS idx_runs_sys ON runs(memory_system);
    CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
    """

    def __init__(self, db_path: str | Path = "results/thesis.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def save_run(
        self,
        run_id: str,
        name: str,
        env_name: str,
        memory_system: str,
        theta: tuple | list | None,
        config: Any,
        metrics: dict[str, float],
        episode_records: list[dict] | None = None,
        notes: str = "",
    ) -> str:
        """
        Save a complete experiment run to the database.

        Parameters
        ----------
        run_id : str
            Unique identifier (config_hash + timestamp or experiment name).
        name : str
            Human-readable experiment name.
        env_name : str
            Environment name.
        memory_system : str
            Memory system name.
        theta : tuple or list, optional
            Learned theta (for GraphMemory-based experiments).
        config : dict or ExperimentConfig
            Full experiment config.
        metrics : dict
            Aggregate metrics (mean_reward, retrieval_precision, etc.).
        episode_records : list of dict, optional
            Per-episode data (reward, tokens, memory_size, retrieval_precision).
        notes : str
            Free-text notes.

        Returns
        -------
        run_id : str
        """
        config_json = json.dumps(
            config if isinstance(config, dict) else getattr(config, "to_dict", lambda: {})(),
            default=str
        )
        theta_json = json.dumps(list(theta) if theta else [])

        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, name, env_name, memory_system, theta_json, config_json, time.time(), notes)
        )

        # Save metrics
        self._conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?)",
            [(run_id, k, float(v)) for k, v in metrics.items() if isinstance(v, (int, float))]
        )

        # Save episode records
        if episode_records:
            self._conn.executemany(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        i,
                        r.get("reward", 0.0),
                        r.get("retrieval_tokens", 0.0),
                        r.get("memory_size", 0.0),
                        r.get("retrieval_precision") or -1.0,
                    )
                    for i, r in enumerate(episode_records)
                ]
            )

        self._conn.commit()
        return run_id

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def best_theta(self, env_name: str, memory_system: str = "GraphMemory") -> dict | None:
        """Return the run with the highest mean_reward for a given env and memory system."""
        cur = self._conn.execute(
            """
            SELECT r.*, res.value as mean_reward
            FROM runs r
            JOIN results res ON r.id = res.run_id
            WHERE r.env_name = ? AND r.memory_system = ? AND res.metric = 'mean_reward'
            ORDER BY res.value DESC LIMIT 1
            """,
            (env_name, memory_system)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "run_id": row["id"],
            "name": row["name"],
            "theta": json.loads(row["theta"]),
            "mean_reward": row["mean_reward"],
        }

    def compare_systems(self, env_name: str, metric: str = "mean_reward") -> list[dict]:
        """Return all systems ranked by metric for a given environment."""
        cur = self._conn.execute(
            """
            SELECT r.memory_system, r.theta, res.value as metric_value,
                   r.timestamp, r.id
            FROM runs r
            JOIN results res ON r.id = res.run_id
            WHERE r.env_name = ? AND res.metric = ?
            ORDER BY res.value DESC
            """,
            (env_name, metric)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_rewards(self, run_id: str) -> list[float]:
        """Return per-episode rewards for a specific run."""
        cur = self._conn.execute(
            "SELECT reward FROM episodes WHERE run_id = ? ORDER BY episode",
            (run_id,)
        )
        return [row["reward"] for row in cur.fetchall()]

    def get_config(self, run_id: str) -> dict | None:
        """Return the full config for a specific run."""
        cur = self._conn.execute("SELECT config FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row["config"])

    def list_runs(self, env_name: str | None = None, n: int = 20) -> list[dict]:
        """List recent experiment runs."""
        if env_name:
            cur = self._conn.execute(
                "SELECT id, name, env_name, memory_system, timestamp FROM runs "
                "WHERE env_name = ? ORDER BY timestamp DESC LIMIT ?",
                (env_name, n)
            )
        else:
            cur = self._conn.execute(
                "SELECT id, name, env_name, memory_system, timestamp FROM runs "
                "ORDER BY timestamp DESC LIMIT ?",
                (n,)
            )
        return [dict(row) for row in cur.fetchall()]

    def get_all_metrics(self, run_id: str) -> dict:
        """Return all metrics for a run as a dict."""
        cur = self._conn.execute(
            "SELECT metric, value FROM results WHERE run_id = ?", (run_id,)
        )
        return {row["metric"]: row["value"] for row in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *args) -> None:
        self.close()
