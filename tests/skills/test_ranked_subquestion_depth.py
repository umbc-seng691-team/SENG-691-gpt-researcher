"""Tests for bounded deeper research on high-ranked questions."""

from types import SimpleNamespace

import pytest

import gpt_researcher.skills.researcher as researcher_module
from gpt_researcher.skills.researcher import ResearchConductor


@pytest.mark.asyncio
async def test_only_selected_question_gets_bounded_expansion(monkeypatch):
    cfg = SimpleNamespace(subquestion_deep_research_extra_queries=2)
    researcher = SimpleNamespace(
        cfg=cfg,
        parent_query="parent topic",
        report_type="research_report",
        prompt_family=object(),
        kwargs={},
        add_costs=lambda *_: None,
    )
    conductor = ResearchConductor.__new__(ResearchConductor)
    conductor.researcher = researcher
    conductor.logger = researcher_module.logging.getLogger("test")
    conductor._deep_research_queries = {"high priority"}
    captured = {}

    async def fake_generate_sub_queries(**kwargs):
        captured.update(kwargs)
        return ["focused one", "high priority", "focused two"]

    monkeypatch.setattr(researcher_module, "generate_sub_queries", fake_generate_sub_queries)

    assert await conductor._get_deeper_sub_queries("low priority", []) == []
    deeper = await conductor._get_deeper_sub_queries("high priority", "evidence")

    assert deeper == ["focused one", "focused two"]
    assert captured["num_queries"] == 2
    assert captured["context"] == "evidence"