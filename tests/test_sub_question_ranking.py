"""Tests for relevance ranking of generated research sub-questions."""

from types import SimpleNamespace

import pytest

import gpt_researcher.actions.sub_question_ranking as ranking
from gpt_researcher.actions.sub_question_ranking import (
    parse_ranked_sub_questions,
    rank_sub_questions,
)


def test_parser_sorts_scores_and_ignores_unknown_questions():
    ranked = parse_ranked_sub_questions(
        '{"questions":[{"question":"side topic","score":100},'
        '{"question":"core topic","score":88},'
        '{"question":"invented","score":99}]}',
        ["core topic", "side topic"],
    )

    assert [item["question"] for item in ranked] == ["side topic", "core topic"]
    assert [item["score"] for item in ranked] == [100, 88]


def test_parser_clamps_scores_and_preserves_input_order_for_ties():
    ranked = parse_ranked_sub_questions(
        '[{"question":"first","score":150},{"question":"second","score":-5}]',
        ["first", "second"],
    )

    assert [item["question"] for item in ranked] == ["first", "second"]
    assert [item["score"] for item in ranked] == [100, 0]


@pytest.mark.asyncio
async def test_rank_sub_questions_uses_strategic_llm(monkeypatch):
    cfg = SimpleNamespace(
        strategic_llm_model="test-model",
        strategic_llm_provider="test-provider",
        llm_kwargs={},
    )
    captured = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return '{"questions":[{"question":"q2","score":90},'
        '{"question":"q1","score":20}]}'

    monkeypatch.setattr(ranking, "create_chat_completion", fake_completion)
    ranked = await rank_sub_questions("original", ["q1", "q2"], cfg)

    assert [item["question"] for item in ranked] == ["q2", "q1"]
    assert captured["model"] == "test-model"


@pytest.mark.asyncio
async def test_rank_sub_questions_preserves_order_when_llm_fails(monkeypatch):
    cfg = SimpleNamespace(
        strategic_llm_model="test-model",
        strategic_llm_provider="test-provider",
        llm_kwargs={},
    )

    async def failed_completion(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(ranking, "create_chat_completion", failed_completion)
    ranked = await rank_sub_questions("original", ["q1", "q2"], cfg)

    assert [item["question"] for item in ranked] == ["q1", "q2"]
    assert all(item["score"] == 0 for item in ranked)