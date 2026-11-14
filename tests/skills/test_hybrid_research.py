"""Tests for hybrid local and web research."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpt_researcher.skills.researcher import ResearchConductor


class PromptFamilyStub:
    @staticmethod
    def join_local_web_documents(local_context, web_context):
        return f"LOCAL:\n{local_context}\n\nWEB:\n{web_context}"


def make_conductor(*, document_urls=None):
    researcher = SimpleNamespace(
        document_urls=document_urls,
        cfg=SimpleNamespace(doc_path="/tmp/research-documents"),
        vector_store=None,
        prompt_family=PromptFamilyStub,
    )
    return ResearchConductor(researcher)


@pytest.mark.asyncio
async def test_web_search_starts_while_local_documents_are_loading():
    conductor = make_conductor()
    web_started = asyncio.Event()
    documents = [{"raw_content": "local evidence"}]

    async def load_documents():
        await asyncio.wait_for(web_started.wait(), timeout=0.5)
        return documents

    async def get_context(query, scraped_data, query_domains):
        if not scraped_data:
            web_started.set()
            return "web evidence"
        assert scraped_data is documents
        return "local evidence"

    with patch("gpt_researcher.skills.researcher.DocumentLoader") as loader_class:
        loader_class.return_value.load = AsyncMock(side_effect=load_documents)
        conductor._get_context_by_web_search = AsyncMock(side_effect=get_context)

        result = await asyncio.wait_for(
            conductor._get_context_by_hybrid_search(
                "hybrid question",
                ["example.edu"],
            ),
            timeout=1,
        )

    assert result == "LOCAL:\nlocal evidence\n\nWEB:\nweb evidence"
    assert conductor._get_context_by_web_search.await_count == 2


@pytest.mark.asyncio
async def test_web_context_is_kept_when_local_loading_fails():
    conductor = make_conductor()

    with patch("gpt_researcher.skills.researcher.DocumentLoader") as loader_class:
        loader_class.return_value.load = AsyncMock(
            side_effect=OSError("local file could not be read")
        )
        conductor._get_context_by_web_search = AsyncMock(return_value="web evidence")

        result = await conductor._get_context_by_hybrid_search("hybrid question")

    assert result == "LOCAL:\n\n\nWEB:\nweb evidence"


@pytest.mark.asyncio
async def test_local_context_is_kept_when_web_search_fails():
    conductor = make_conductor()
    documents = [{"raw_content": "local evidence"}]

    async def get_context(query, scraped_data, query_domains):
        if scraped_data:
            return ["local evidence", "supporting local detail"]
        raise RuntimeError("web provider unavailable")

    with patch("gpt_researcher.skills.researcher.DocumentLoader") as loader_class:
        loader_class.return_value.load = AsyncMock(return_value=documents)
        conductor._get_context_by_web_search = AsyncMock(side_effect=get_context)

        result = await conductor._get_context_by_hybrid_search("hybrid question")

    assert result == "LOCAL:\nlocal evidence\nsupporting local detail\n\nWEB:\n"


@pytest.mark.asyncio
async def test_empty_result_when_both_sources_fail():
    conductor = make_conductor()

    with patch("gpt_researcher.skills.researcher.DocumentLoader") as loader_class:
        loader_class.return_value.load = AsyncMock(
            side_effect=OSError("local unavailable")
        )
        conductor._get_context_by_web_search = AsyncMock(
            side_effect=RuntimeError("web unavailable")
        )

        result = await conductor._get_context_by_hybrid_search("hybrid question")

    assert result == []
