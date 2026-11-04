"""Relevance ranking for generated research sub-questions."""

import json
import logging
from typing import Any, Dict, List

import json_repair

from ..config import Config
from ..llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


def parse_ranked_sub_questions(
    response: str,
    questions: List[str],
) -> List[Dict[str, Any]]:
    """Parse model rankings, retaining only generated questions.

    Missing or malformed scores receive a zero score, and the original
    ordering is retained as a stable tie-breaker.
    """
    try:
        parsed = json_repair.loads(response)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        parsed = parsed.get("questions", parsed.get("rankedQuestions", []))
    if not isinstance(parsed, list):
        parsed = []

    question_by_normalized = {question.casefold(): question for question in questions}
    scores: Dict[str, tuple[float, str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        if not isinstance(question, str):
            continue
        canonical = question_by_normalized.get(question.strip().casefold())
        if canonical is None:
            continue
        try:
            score = max(0.0, min(100.0, float(item.get("score", 0))))
        except (TypeError, ValueError):
            score = 0.0
        reason = item.get("reason", "")
        scores[canonical] = (score, reason if isinstance(reason, str) else "")

    ranked = [
        {
            "question": question,
            "score": scores.get(question, (0.0, ""))[0],
            "reason": scores.get(question, (0.0, ""))[1],
        }
        for question in questions
    ]
    return sorted(ranked, key=lambda item: -item["score"])


async def rank_sub_questions(
    original_query: str,
    questions: List[str],
    cfg: Config,
    cost_callback: callable = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Rank generated questions by how much they help answer the original."""
    if len(questions) < 2:
        return [
            {"question": question, "score": 100.0, "reason": "Only generated question."}
            for question in questions
        ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research-planning expert. Rank questions by relevance and "
                "importance to answering the original user question. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n\n"
                f"Generated sub-questions: {json.dumps(questions, ensure_ascii=False)}\n\n"
                "Score each supplied question from 0 to 100. Consider direct relevance, "
                "importance to a complete answer, and whether it adds distinct useful "
                "coverage. Do not add, remove, or rewrite questions.\n"
                'Return only {"questions":[{"question":"exact supplied question",'
                '"score":0,"reason":"brief rationale"}]}'
            ),
        },
    ]
    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=messages,
            llm_provider=cfg.strategic_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs,
        )
    except Exception:
        logger.exception("Could not rank sub-questions; preserving generated order.")
        return [
            {"question": question, "score": 0.0, "reason": ""}
            for question in questions
        ]

    ranked = parse_ranked_sub_questions(response, questions)
    if not any(item["score"] > 0 for item in ranked):
        return [
            {"question": question, "score": 0.0, "reason": ""}
            for question in questions
        ]
    return ranked