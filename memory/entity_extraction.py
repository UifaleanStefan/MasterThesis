"""Rule-based entity extraction from observations."""

COLORS = ["red", "blue", "green", "yellow", "purple"]
OBJECTS = ["key", "door"]


def extract_entities(observation: str) -> list[str]:
    """
    Extract entity names from observation using simple string matching.
    "red key" -> "red_key", "blue door" -> "blue_door", etc.
    Deterministic, no ML.
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
