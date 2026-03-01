"""Event embedding module - lightweight, deterministic."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixed vocabulary covering our observation space (all environments).
# Includes MultiHopKeyDoor colors (orange, cyan, magenta, white) and
# hint-related words (sign, opens, north, east, south, requires).
VOCAB = [
    "you", "are", "in", "a", "room", "see", "red", "blue", "green",
    "key", "door", "nothing", "interest", "carrying", "of", "goal",
    "yellow", "purple", "opened", "have", "doors",
    "orange", "cyan", "magenta", "white",
    "sign", "opens", "north", "east", "south", "requires",
]
_vectorizer: TfidfVectorizer | None = None


def _get_vectorizer() -> TfidfVectorizer:
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            vocabulary={w: i for i, w in enumerate(VOCAB)},
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        _vectorizer.fit([" ".join(VOCAB)])
    return _vectorizer


def embed_observation(observation: str) -> np.ndarray:
    """
    Embed observation into fixed-size vector.
    Uses TF-IDF with fixed vocabulary. Deterministic, lightweight, fast.
    """
    vec = _get_vectorizer()
    X = vec.transform([observation])
    return X.toarray().flatten().astype(np.float32)
