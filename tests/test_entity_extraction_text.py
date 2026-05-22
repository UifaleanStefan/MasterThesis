"""Tests for the Stage 3 text-mode entity extractor (Block A.4 / critique #16).

Covers:
  * extract_entities (grid-world) — unchanged behavior
  * extract_entities_text — money, years, quarters, sections, proper nouns
  * extract_entities_auto — correct dispatch based on observation shape
  * V4 with text_mode_entities=True — entities created on first mention
  * V4 with text_mode_entities=False — Bayesian gate still applies
"""

from __future__ import annotations

import pytest

from memory.entity_extraction import (
    _looks_like_grid_world,
    extract_entities,
    extract_entities_auto,
    extract_entities_text,
)
from memory.event import Event
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4


# ----------------------------------------------------------------------
# Grid-world extractor — backward compatibility
# ----------------------------------------------------------------------


class TestGridExtractorUnchanged:
    def test_red_key(self):
        assert extract_entities("You see a red key on the floor.") == ["red_key"]

    def test_blue_door(self):
        assert extract_entities("There is a blue door to the north.") == ["blue_door"]

    def test_goal_marker(self):
        assert extract_entities("Goal cell ahead.") == ["goal"]

    def test_multiple_colors(self):
        e = extract_entities("Red key and blue door visible.")
        assert "red_key" in e and "blue_door" in e

    def test_no_entities(self):
        assert extract_entities("You move forward.") == []

    def test_case_insensitive(self):
        assert extract_entities("YOU SEE A RED KEY") == ["red_key"]


# ----------------------------------------------------------------------
# Text extractor — Stage 3 paths
# ----------------------------------------------------------------------


class TestTextExtractor:
    def test_money_dollar(self):
        e = extract_entities_text("Revenue was $1.2 billion this quarter.")
        assert any("money" in ent for ent in e)

    def test_year(self):
        e = extract_entities_text("Microsoft reported strong growth in 2023.")
        assert any(ent == "year_2023" for ent in e)
        assert "microsoft" in e

    def test_quarter(self):
        e = extract_entities_text("In Q3 2024 the company recorded a loss.")
        assert any("q3" in ent.lower() for ent in e)

    def test_fy_marker(self):
        e = extract_entities_text("Throughout FY2024 the segment grew.")
        assert any("fy2024" in ent.lower() for ent in e)

    def test_section_reference(self):
        e = extract_entities_text("Refer to Section 5.2 of the contract.")
        assert any("section" in ent.lower() and "5_2" in ent for ent in e)

    def test_article_reference(self):
        e = extract_entities_text("As described in Article 12, both parties agree.")
        assert any("article" in ent.lower() and "12" in ent for ent in e)

    def test_proper_noun_multi_word(self):
        e = extract_entities_text("United States Securities and Exchange Commission filing.")
        # Should detect a multi-word proper noun
        assert any(len(ent.split("_")) >= 2 for ent in e)

    def test_proper_noun_single_word(self):
        e = extract_entities_text("Microsoft expanded its cloud business.")
        assert "microsoft" in e

    def test_stopword_excluded(self):
        # "The" at sentence start should not become an entity by itself
        e = extract_entities_text("The contract was signed.")
        assert "the" not in e

    def test_dedup_separate_occurrences(self):
        # Separate occurrences of the same entity should appear once.
        # ("Microsoft Microsoft Microsoft" in a row is interpreted as a
        # single multi-word proper noun by the regex, which is correct.)
        e = extract_entities_text(
            "Microsoft expanded its cloud business. The Microsoft Azure platform grew rapidly."
        )
        assert e.count("microsoft") == 1

    def test_min_length(self):
        # Sub-3-char "AB" should not be an entity
        e = extract_entities_text("Stock AB rose.")
        assert "ab" not in e

    def test_cap_max_entities(self):
        # Build a pathologically entity-rich paragraph
        words = " ".join(f"Apple{i}" for i in range(50))
        e = extract_entities_text(words, max_entities=10)
        assert len(e) <= 10

    def test_empty_observation(self):
        assert extract_entities_text("") == []

    def test_finance_paragraph(self):
        para = (
            "Microsoft's Q3 2023 revenue rose to $56.5 billion, an increase of 13% "
            "year-over-year. The Office Commercial segment, described in Section 4.1 "
            "of the 10-K filing, grew 17% during fiscal year 2023."
        )
        e = extract_entities_text(para)
        # Should pick up Microsoft, dollar amount, quarter, section, year
        ent_str = " ".join(e)
        assert "microsoft" in ent_str
        assert "money" in ent_str
        assert "section" in ent_str or "4_1" in ent_str
        assert "q3" in ent_str.lower() or "year_2023" in ent_str


# ----------------------------------------------------------------------
# Auto-dispatch
# ----------------------------------------------------------------------


class TestAutoDispatch:
    def test_short_grid_obs_routes_to_grid(self):
        assert _looks_like_grid_world("You see a red key.")
        assert extract_entities_auto("You see a red key.") == ["red_key"]

    def test_short_grid_obs_with_goal(self):
        assert _looks_like_grid_world("Goal is to the north.")

    def test_long_text_obs_routes_to_text(self):
        para = "Microsoft Corporation announced its quarterly earnings for fiscal year 2023, reporting revenue of $56.5 billion. " * 3
        assert not _looks_like_grid_world(para)
        e = extract_entities_auto(para)
        assert "microsoft" in str(e).lower() or "microsoft_corporation" in str(e).lower()

    def test_short_text_without_grid_keywords_routes_to_text(self):
        # Short paragraph but no grid-world keywords
        e = extract_entities_auto("Microsoft posts record Q4 earnings.")
        ent_str = " ".join(e).lower()
        assert "microsoft" in ent_str

    def test_long_obs_with_grid_keyword_routes_to_text(self):
        # If observation is long enough, even with "red key" mention, text mode dispatches
        para = "The red key concept in HiPPO's hippocampal memory model. " * 5
        assert not _looks_like_grid_world(para)


# ----------------------------------------------------------------------
# V4 integration with text_mode_entities
# ----------------------------------------------------------------------


class TestV4TextModeEntities:
    def _params(self, text_mode: bool) -> MemoryParamsV4:
        # Use a permissive theta_store so storage gate doesn't block events.
        return MemoryParamsV4(
            theta_store=0.0,
            theta_novel=0.5,
            theta_erich=0.0,
            theta_surprise=0.0,
            theta_entity=0.7,  # high threshold so the Bayesian gate suppresses most
            theta_temporal=1.0,
            theta_decay=0.0,
            w_graph=0.0,
            w_embed=1.0,
            w_recency=0.0,
            text_mode_entities=text_mode,
        )

    def test_text_mode_creates_entities_on_first_mention(self):
        m = GraphMemoryV4(self._params(text_mode=True))
        ev = Event(
            step=0,
            observation="Microsoft reported $56 billion revenue in Q3 2023.",
            action="read",
        )
        m.add_event(ev, episode_seed=42)
        entity_nodes = [
            n for n, d in m._graph.nodes(data=True) if d.get("type") == "entity"
        ]
        # With text_mode_entities=True, the gate is bypassed.
        assert len(entity_nodes) > 0, "text_mode should create entities on first mention"

    def test_legacy_mode_suppresses_entities_on_first_mention(self):
        m = GraphMemoryV4(self._params(text_mode=False))
        ev = Event(
            step=0,
            observation="Microsoft reported $56 billion revenue in Q3 2023.",
            action="read",
        )
        m.add_event(ev, episode_seed=42)
        entity_nodes = [
            n for n, d in m._graph.nodes(data=True) if d.get("type") == "entity"
        ]
        # With theta_entity=0.7 and Bayesian gate, single-mention entities are
        # below threshold and should be filtered. Legacy behavior preserved.
        assert len(entity_nodes) == 0, "legacy gate should suppress single-mention entities"

    def test_text_mode_accumulates_mentions(self):
        m = GraphMemoryV4(self._params(text_mode=True))
        for i in range(5):
            ev = Event(
                step=i,
                observation=f"Microsoft Q3 2023 earnings call number {i}.",
                action="read",
            )
            m.add_event(ev, episode_seed=42)
        assert m._entity_mention_count.get("microsoft", 0) == 5

    def test_grid_observations_still_use_grid_extractor(self):
        # Grid-world observations should produce grid-world entities even
        # when text_mode_entities=True (auto-dispatch handles this).
        m = GraphMemoryV4(self._params(text_mode=True))
        ev = Event(step=0, observation="You see a red key.", action="up")
        m.add_event(ev, episode_seed=42)
        assert "red_key" in [
            n for n, d in m._graph.nodes(data=True) if d.get("type") == "entity"
        ]


# ----------------------------------------------------------------------
# Corpus tracer smoke test (no API, no LLM)
# ----------------------------------------------------------------------


class TestCorpusTracerSmoke:
    """Run the actual ingestion script on FinanceBench with --limit-docs 5.

    Validates the full pipeline: load adapter, build V4ₜ, ingest paragraphs,
    snapshot, dump final graph, write meta.json. No API calls.
    """

    def test_ingestion_smoke(self, tmp_path):
        import subprocess
        import json
        import sys

        out_dir = tmp_path / "smoke_trace"
        cmd = [
            sys.executable,
            "scripts/run_corpus_ingestion.py",
            "--benchmark", "financebench",
            "--config", "v4t-tuned",
            "--limit-docs", "5",
            "--out-dir", str(out_dir),
            "--progress-every-docs", "5",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"ingestion failed: {result.stderr[-500:]}"

        # Verify outputs.
        assert (out_dir / "snapshots.json").exists()
        assert (out_dir / "meta.json").exists()
        assert (out_dir / "final_graph.json").exists()

        meta = json.loads((out_dir / "meta.json").read_text())
        assert meta["benchmark"] == "financebench"
        assert meta["n_docs"] == 5
        assert meta["limit_docs"] == 5
        assert meta["final_n_event_nodes"] > 0
        # With text-mode entities + financial text, should produce entities.
        assert meta["final_n_entity_nodes"] > 0

        snapshots = json.loads((out_dir / "snapshots.json").read_text())
        assert len(snapshots) > 0
        # First snapshot should have at least 1 entity if text mode is wired right.
        last_snap = snapshots[-1]
        assert last_snap["n_entity_nodes"] > 0
