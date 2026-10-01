"""Text normalisation and similarity shared by the ILO labeller and the ARI approaches."""
import re
from difflib import SequenceMatcher


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()
