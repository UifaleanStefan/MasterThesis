"""
HippoRAGMemory — faithful reimplementation of HippoRAG (Gutierrez et al.,
NeurIPS 2024, arXiv:2405.14831) behind our standard memory interface.

REAL HippoRAG, in three stages:

  1. Offline indexing. For each passage (chunk), an LLM performs OpenIE to
     extract (subject, predicate, object) triples. Subjects/objects become
     schemaless *phrase/entity* nodes; passages become *passage* nodes.
     Edges: (a) triple relation edges between co-occurring phrases, (b)
     phrase<->passage membership, and (c) *synonymy* edges between phrase
     nodes whose retrieval-encoder embeddings have cosine similarity above a
     threshold tau (paper: tau = 0.8). This is the "hippocampal index".

  2. Query time. An LLM extracts the query's named entities; each is matched
     to its nearest KG phrase node by embedding cosine (query nodes R_q).
     These become the *seeds* for Personalized PageRank. Each seed's
     probability is multiplied by that node's *specificity* s_i = 1/|P_i|
     (|P_i| = number of passages the node was extracted from) - a
     neurobiologically-motivated stand-in for IDF that down-weights
     ubiquitous nodes.

  3. Ranking. PPR is run over the KG from the seed distribution, producing an
     updated node probability vector n'. Passage scores are p = P^T n', where
     P is the |N|x|P| node-passage occurrence matrix - i.e. each passage is
     scored by the summed PPR mass of the phrase nodes it contains. Top-k
     passages are returned.

DOCUMENTED DIVERGENCE FROM THE REAL SYSTEM (disclosable in the thesis):

  * OpenIE-via-LLM is REPLACED by non-LLM extraction. We use the project's
    ``extract_entities()`` (auto-dispatched to the text extractor for NL
    passages) to produce phrase/entity nodes, and we replace LLM-extracted
    triple *relation* edges with intra-passage **co-occurrence** edges
    (every pair of entities appearing in the same passage is linked). This
    keeps ingest LLM-free and tractable for 5k-20k passages on CPU. The
    paper's own error analysis attributes 28% of its errors to OpenIE
    misses and 48% to NER omissions, so the extractor is the single biggest
    lever we are changing - we are explicit that our recall ceiling is set
    by ``extract_entities`` rather than an LLM OpenIE module.

  * Query-entity extraction (LLM in the real system) is likewise replaced by
    the same ``extract_entities()`` applied to the question string. No LLM
    call happens at query time (nor at ingest).

  * Everything else is faithful: schemaless phrase nodes + passage nodes,
    embedding-synonymy edges at cosine > tau, node specificity 1/|P_i|,
    seed = matched query nodes weighted by specificity, ``networkx``
    Personalized PageRank, and passage ranking by summed PPR of contained
    entities (tie-broken here by query embedding cosine, which is not in the
    original but only orders exact ties).

Passages are stored as ``Event`` nodes (event.observation is the passage
text). The answerer (gpt-4o-mini) is identical across all configs; this
adapter only changes *retrieval*.

Interface mirrors memory/semantic_memory.py exactly:
    add_event(event, episode_seed=None) -> None      # ingest one passage
    get_relevant_events(observation, current_step, k=8) -> list[Event]
    clear() -> None
    get_stats() -> dict  # {'n_events','n_entities','n_nodes','n_edges'}
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities_auto
from .event import Event


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class HippoRAGMemory:
    """
    Faithful HippoRAG retrieval memory.

    Parameters
    ----------
    synonymy_tau : float
        Cosine threshold above which two phrase nodes get a synonymy edge
        (paper default 0.8). Set > 1.0 to disable synonymy edges entirely.
    synonymy_topk : int
        For tractability, when adding a *new* phrase node we only consider
        linking it to its ``synonymy_topk`` most-similar existing phrase
        nodes (by brute-force cosine over the phrase embedding matrix)
        instead of all-pairs. Faithful to the ANN-index step of the paper
        (which also only inspects near neighbours), just brute-forced.
    damping : float
        PPR damping factor alpha (networkx default 0.85; HippoRAG uses 0.5).
    ppr_max_iter : int
        Power-iteration cap for networkx.pagerank.
    use_node_specificity : bool
        Multiply each seed probability by s_i = 1/|P_i| before PPR (the
        paper's IDF surrogate). True == faithful.
    query_match_min_cos : float
        A query entity that has no exact node match is matched to its nearest
        phrase node only if cosine >= this value; otherwise it is dropped
        (prevents seeding on unrelated nodes for OOV query terms).
    """

    def __init__(
        self,
        synonymy_tau: float = 0.8,
        synonymy_topk: int = 8,
        damping: float = 0.5,
        ppr_max_iter: int = 100,
        use_node_specificity: bool = True,
        query_match_min_cos: float = 0.5,
    ) -> None:
        self.synonymy_tau = float(synonymy_tau)
        self.synonymy_topk = int(synonymy_topk)
        self.damping = float(damping)
        self.ppr_max_iter = int(ppr_max_iter)
        self.use_node_specificity = bool(use_node_specificity)
        self.query_match_min_cos = float(query_match_min_cos)

        self._graph = nx.Graph()  # undirected KG (phrase & passage nodes)

        # Event / passage bookkeeping.
        self._events: dict[str, Event] = {}          # passage_node_id -> Event
        self._passage_emb: dict[str, np.ndarray] = {}  # passage_node_id -> emb

        # Phrase/entity bookkeeping.
        # entity_id -> set of passage_node_ids it was extracted from (|P_i|).
        self._entity_passages: dict[str, set[str]] = {}
        # entity_id -> L2-normalized MiniLM embedding (for synonymy + query NN).
        self._entity_emb: dict[str, np.ndarray] = {}
        # Parallel arrays for fast brute-force nearest-neighbour search.
        self._entity_ids: list[str] = []
        self._entity_mat: np.ndarray | None = None    # (n_entities, 384), L2-normed
        self._entity_mat_dirty: bool = True

        self._n_passages: int = 0

    # ------------------------------------------------------------------
    # Passage-node id
    # ------------------------------------------------------------------
    @staticmethod
    def _passage_node(step: int) -> str:
        return f"passage_{step}"

    @staticmethod
    def _entity_node(entity: str) -> str:
        # extract_entities_auto already returns canonical lowercased ids;
        # namespace them so they can't collide with passage ids.
        return f"ent::{entity}"

    # ------------------------------------------------------------------
    # Ingest (offline indexing) - NO LLM CALLS
    # ------------------------------------------------------------------
    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        """Index one passage into the hippocampal KG.

        Cost: 1 MiniLM embed for the passage + 1 MiniLM embed per *new*
        entity phrase (LRU-cached, so repeats are free) + a brute-force
        top-k cosine scan against existing phrases for synonymy edges.
        No LLM/OpenIE call.
        """
        pid = self._passage_node(event.step)

        # Passage (Event) node.
        p_emb = embed_observation(event.observation)
        if not self._graph.has_node(pid):
            self._n_passages += 1
        self._graph.add_node(pid, node_type="passage")
        self._events[pid] = event
        self._passage_emb[pid] = p_emb

        # Phrase/entity nodes (OpenIE replaced by extract_entities).
        entities = extract_entities_auto(event.observation)
        # Dedupe within-passage while preserving order (for co-occ pairing).
        seen: list[str] = []
        seen_set: set[str] = set()
        for e in entities:
            if e not in seen_set:
                seen_set.add(e)
                seen.append(e)

        entity_node_ids: list[str] = []
        for e in seen:
            eid = self._entity_node(e)
            is_new = eid not in self._entity_passages

            # phrase node + membership edge (phrase <-> passage).
            self._graph.add_node(eid, node_type="entity")
            self._graph.add_edge(pid, eid, edge_type="membership")

            self._entity_passages.setdefault(eid, set()).add(pid)

            if is_new:
                emb = _l2_normalize(embed_observation(e).astype(np.float32))
                self._entity_emb[eid] = emb
                # Synonymy edges: link to near neighbours above tau.
                self._add_synonymy_edges(eid, emb)
                self._entity_ids.append(eid)
                self._entity_mat_dirty = True

            entity_node_ids.append(eid)

        # Intra-passage co-occurrence edges (OpenIE relation-edge surrogate).
        # Every unordered pair of entities in this passage is linked.
        n = len(entity_node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = entity_node_ids[i], entity_node_ids[j]
                if self._graph.has_edge(a, b):
                    # bump weight so repeatedly co-occurring phrases bind tighter
                    self._graph[a][b]["weight"] = self._graph[a][b].get("weight", 1.0) + 1.0
                else:
                    self._graph.add_edge(a, b, edge_type="cooccur", weight=1.0)

    def _add_synonymy_edges(self, new_eid: str, new_emb: np.ndarray) -> None:
        """Link a freshly-added phrase to existing phrases with cosine > tau.

        Brute-force top-k over the existing phrase-embedding matrix. Uses L2-
        normalized rows so a single matvec gives all cosines.
        """
        if self.synonymy_tau > 1.0 or not self._entity_ids:
            return
        mat = self._get_entity_matrix()  # (m, 384) normalized, aligned to _entity_ids
        if mat is None or mat.shape[0] == 0:
            return
        sims = mat @ new_emb  # (m,)
        # Top-k candidates, then threshold.
        m = sims.shape[0]
        k = min(self.synonymy_topk, m)
        if k <= 0:
            return
        cand = np.argpartition(-sims, k - 1)[:k]
        for idx in cand:
            if sims[idx] >= self.synonymy_tau:
                other = self._entity_ids[idx]
                if other != new_eid and not self._graph.has_edge(new_eid, other):
                    self._graph.add_edge(
                        new_eid, other, edge_type="synonymy",
                        weight=float(sims[idx]),
                    )

    def _get_entity_matrix(self) -> np.ndarray | None:
        """(len(_entity_ids), 384) L2-normalized matrix, rebuilt when dirty."""
        if not self._entity_ids:
            return None
        if self._entity_mat_dirty or self._entity_mat is None or \
                self._entity_mat.shape[0] != len(self._entity_ids):
            self._entity_mat = np.vstack([self._entity_emb[e] for e in self._entity_ids])
            self._entity_mat_dirty = False
        return self._entity_mat

    # ------------------------------------------------------------------
    # Query time - NO LLM CALLS
    # ------------------------------------------------------------------
    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """Seed PPR on query-entity nodes, rank passages by summed PPR mass."""
        if not self._events:
            return []

        query_emb = embed_observation(observation)

        # 1) Extract query named entities (OpenIE/NER replaced by extractor).
        q_entities = extract_entities_auto(observation)

        # 2) Map each query entity to a KG phrase node -> seeds R_q.
        #    Exact id match preferred; else nearest phrase by cosine (>= min).
        seed_scores: dict[str, float] = {}
        ent_mat = self._get_entity_matrix()
        for e in q_entities:
            eid = self._entity_node(e)
            if eid in self._entity_passages:
                matched = eid
            elif ent_mat is not None:
                q_e = _l2_normalize(embed_observation(e).astype(np.float32))
                sims = ent_mat @ q_e
                best = int(np.argmax(sims))
                if sims[best] >= self.query_match_min_cos:
                    matched = self._entity_ids[best]
                else:
                    continue
            else:
                continue

            # 3) Weight seed by node specificity s_i = 1/|P_i| (IDF surrogate).
            if self.use_node_specificity:
                spec = 1.0 / max(len(self._entity_passages.get(matched, ())), 1)
            else:
                spec = 1.0
            seed_scores[matched] = seed_scores.get(matched, 0.0) + spec

        # Fallback: no query entities matched any node -> pure embedding cosine
        # over passages (keeps the adapter robust; the paper degrades to dense
        # retrieval in the same no-seed situation).
        if not seed_scores:
            return self._embedding_only_topk(query_emb, k)

        # Normalize personalization to a probability distribution.
        total = sum(seed_scores.values())
        personalization = {node: s / total for node, s in seed_scores.items()}

        # 4) Personalized PageRank over the KG.
        try:
            ppr = nx.pagerank(
                self._graph,
                alpha=self.damping,
                personalization=personalization,
                max_iter=self.ppr_max_iter,
                weight="weight",
                dangling=personalization,
            )
        except nx.PowerIterationFailedConvergence:
            ppr = nx.pagerank(
                self._graph,
                alpha=self.damping,
                personalization=personalization,
                max_iter=self.ppr_max_iter * 5,
                tol=1e-4,
                weight="weight",
                dangling=personalization,
            )

        # 5) Rank passages: p = P^T n'  (summed PPR mass of contained phrases).
        passage_score: dict[str, float] = {}
        for eid, passages in self._entity_passages.items():
            mass = ppr.get(eid, 0.0)
            if mass <= 0.0:
                continue
            for pid in passages:
                passage_score[pid] = passage_score.get(pid, 0.0) + mass

        # Include the small direct PPR mass that leaks onto passage nodes too
        # (they are graph nodes and receive rank via membership edges). This
        # only helps break ties toward passages structurally near the seeds.
        for pid in self._events:
            if ppr.get(pid, 0.0) > 0.0:
                passage_score[pid] = passage_score.get(pid, 0.0) + ppr[pid]

        if not passage_score:
            return self._embedding_only_topk(query_emb, k)

        # Tie-break by query-passage embedding cosine (not in the original;
        # only orders exact PPR ties deterministically).
        ranked = sorted(
            passage_score.keys(),
            key=lambda pid: (
                passage_score[pid],
                _cosine_sim(query_emb, self._passage_emb[pid]),
            ),
            reverse=True,
        )
        return [self._events[pid] for pid in ranked[:k]]

    def _embedding_only_topk(self, query_emb: np.ndarray, k: int) -> list[Event]:
        scored = [
            (pid, _cosine_sim(query_emb, emb))
            for pid, emb in self._passage_emb.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return [self._events[pid] for pid, _ in scored[:k]]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._graph.clear()
        self._events.clear()
        self._passage_emb.clear()
        self._entity_passages.clear()
        self._entity_emb.clear()
        self._entity_ids.clear()
        self._entity_mat = None
        self._entity_mat_dirty = True
        self._n_passages = 0

    def get_stats(self) -> dict:
        n_entities = len(self._entity_passages)
        return {
            "n_events": len(self._events),
            "n_entities": n_entities,
            "n_nodes": self._graph.number_of_nodes(),
            "n_edges": self._graph.number_of_edges(),
        }

    def get_graph(self) -> nx.Graph:
        return self._graph
