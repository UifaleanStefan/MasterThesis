"""Rule-based entity extraction from observations.

Two modes:

* Grid-world (default for short obs containing color+object phrases):
  detects `<color> <key|door>` and `goal`. Used by Stage 1 + Stage 2.
  Backward-compatible — `extract_entities(obs)` keeps this behavior.

* Natural-language text (Stage 3 corpus ingestion, real benchmarks):
  detects capitalized multi-word proper nouns, dollar amounts, years,
  quarter markers, and section/clause references. Used when the
  observation looks like a paragraph from a contract, scientific
  paper, financial filing, etc. — i.e. the V4 graph builds entities
  over actual domain language rather than grid-world keywords.

The auto-dispatch (`extract_entities_auto`) selects based on the
observation length and keyword content. Stage 1/2 grid environments
produce short observations with color+object phrases -> grid mode.
Stage 3 benchmarks produce long natural-language paragraphs -> text
mode.
"""

from __future__ import annotations

import re

# ----------------------------------------------------------------------
# Grid-world entity extractor (Stage 1 + Stage 2 -- UNCHANGED)
# ----------------------------------------------------------------------

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "white"]
OBJECTS = ["key", "door"]


def extract_entities(observation: str) -> list[str]:
    """
    Extract entity names from observation using simple string matching.
    "red key" -> "red_key", "blue door" -> "blue_door", etc.
    Deterministic, no ML.

    BACKWARD COMPATIBLE: this is the original Stage 1/2 grid-world
    extractor. Tests and grid environments depend on it. Do not change
    its semantics.
    """
    entities: list[str] = []
    obs_lower = observation.lower()

    for color in COLORS:
        for obj in OBJECTS:
            phrase = f"{color} {obj}"
            if phrase in obs_lower:
                entities.append(f"{color}_{obj}")

    if "goal" in obs_lower:
        entities.append("goal")

    return entities


# ----------------------------------------------------------------------
# Natural-language text entity extractor (Stage 3)
# ----------------------------------------------------------------------

# Capitalized multi-word proper noun (Title Case, 1-5 words). Examples:
#   "Microsoft", "United States Securities", "Q1 2024".
_PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'\-]{2,}(?:\s+[A-Z][a-zA-Z'\-]{2,}){0,4})\b"
)

# Dollar / currency amounts. "$1.2 billion", "$1,234.56", "USD 100M".
_MONEY_RE = re.compile(
    r"(?:\$|USD\s)\s*\d[\d,.]*\s*(?:billion|million|thousand|M|B|K)?",
    re.IGNORECASE,
)

# Years and quarters.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\bQ[1-4]\s*(?:19|20)?\d{2}\b", re.IGNORECASE)
_FY_RE = re.compile(r"\bFY\s*(?:19|20)?\d{2}\b", re.IGNORECASE)

# Section / clause / article references.
_SECTION_RE = re.compile(
    r"\b(?:Section|Article|Clause|Subsection|Paragraph)\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)

# Standard English stopwords that often appear capitalized at sentence
# starts but aren't entities. Curated to be conservative.
_PROPER_NOUN_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Their", "They",
    "If", "When", "Where", "While", "Whereas", "Whether", "What", "Which",
    "Who", "Why", "How", "It", "Its", "Is", "Was", "Were", "Are", "Be",
    "But", "And", "Or", "Not", "We", "Us", "Our", "You", "Your", "He",
    "She", "Him", "Her", "His", "Hers", "I", "Me", "My", "Mine",
    "Yes", "No", "However", "Therefore", "Thus", "Moreover", "Furthermore",
    "Although", "Because", "Since", "Until", "Before", "After",
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "All", "Any", "Some", "Each", "Every", "Many", "Most", "More", "Less",
    "Such", "Other", "Another", "Same", "Different", "New", "Old",
    "Maybe", "Perhaps", "Indeed", "Surely",
    "Note", "Notes", "See", "Refer", "Also",
    "Inc", "Corp", "Ltd", "Co",
    # Common sentence-starters seen on QASPER pilot run that produced
    # high-mention false positives. Generally these are NOT entity-like
    # in scientific papers either.
    "For", "Finally", "Then", "Now", "Here", "Thus",
    "Furthermore", "Moreover", "Specifically", "Importantly",
    "Notably", "Conversely", "Similarly", "Additionally", "Consequently",
    "Subsequently", "Generally", "Typically", "Usually", "Often",
    "Sometimes", "Always", "Never", "Once", "Twice",
    "Within", "Without", "Above", "Below", "Beyond", "Inside", "Outside",
    "Recently", "Currently", "Previously", "Lastly",
}


def _normalize_entity(text: str) -> str:
    """Canonicalize an entity string for use as a graph node ID."""
    s = re.sub(r"[\s\-/.,;:'\"\[\](){}]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def extract_entities_text(observation: str, max_entities: int = 20) -> list[str]:
    """Extract entities from natural-language text.

    Captures:
      * Capitalized proper noun phrases (people, organizations, places)
      * Dollar amounts
      * Years, quarters, fiscal-year markers
      * Section / clause / article references

    Returns a list of canonical entity IDs (lowercased, underscored),
    deduplicated while preserving first-seen order. Capped at
    `max_entities` per observation to avoid pathological inflation
    on entity-heavy paragraphs.
    """
    seen: set[str] = set()
    entities: list[str] = []

    def _add(raw: str) -> bool:
        canonical = _normalize_entity(raw)
        if not canonical or canonical in seen:
            return False
        if len(canonical) < 3:
            return False
        seen.add(canonical)
        entities.append(canonical)
        return True

    # Numeric patterns first (length-anchored, reliable).
    for m in _MONEY_RE.finditer(observation):
        _add("money_" + m.group(0))
        if len(entities) >= max_entities:
            return entities
    for m in _SECTION_RE.finditer(observation):
        _add(m.group(0))
        if len(entities) >= max_entities:
            return entities
    for m in _QUARTER_RE.finditer(observation):
        _add(m.group(0))
        if len(entities) >= max_entities:
            return entities
    for m in _FY_RE.finditer(observation):
        _add(m.group(0))
        if len(entities) >= max_entities:
            return entities
    for m in _YEAR_RE.finditer(observation):
        _add("year_" + m.group(0))
        if len(entities) >= max_entities:
            return entities
    for m in _PROPER_NOUN_RE.finditer(observation):
        raw = m.group(0).strip()
        words = raw.split()
        if len(words) == 1 and words[0] in _PROPER_NOUN_STOPWORDS:
            continue
        if all(w in _PROPER_NOUN_STOPWORDS for w in words):
            continue
        _add(raw)
        if len(entities) >= max_entities:
            return entities

    return entities


# ----------------------------------------------------------------------
# Auto-dispatch
# ----------------------------------------------------------------------

# Heuristic: a grid-world observation is short and contains at least
# one color+object phrase or "goal".
_GRID_THRESHOLD_CHARS = 200


def _looks_like_grid_world(observation: str) -> bool:
    if len(observation) > _GRID_THRESHOLD_CHARS:
        return False
    obs_lower = observation.lower()
    if "goal" in obs_lower:
        return True
    for color in COLORS:
        for obj in OBJECTS:
            if f"{color} {obj}" in obs_lower:
                return True
    return False


def extract_entities_auto(observation: str) -> list[str]:
    """Dispatch to grid-world or text extractor based on observation shape.

    Used by Stage 3 corpus ingestion where observations may be either
    grid-world or natural-language. Backward compatible for Stage 1/2:
    short observations with color+object phrases still produce the
    original grid-world entity list.
    """
    if _looks_like_grid_world(observation):
        return extract_entities(observation)
    return extract_entities_text(observation)
