"""A3 answers -> pair scores in A1's format and turn-type votes, through A2's parser.

A3's answers carry no confidence, so every listed relation weighs 1: a pair's score is the share of
answers that list it (D065). A1's decision layer then applies unchanged.
"""
from a2_llm import parse as a2


def with_confidence(answer: dict) -> dict:
    return {"illocutions": answer.get("illocutions", []),
            "relations": [dict(r, confidence="high") for r in answer.get("relations", [])]}


def relations(answer: dict, s):
    return a2.relations(with_confidence(answer), s)


def pair_scores(s, answers: list[dict]):
    return a2.pair_scores(s, [with_confidence(a) for a in answers])


def type_votes(s, answers: list[dict]):
    return a2.type_votes(s, answers)
