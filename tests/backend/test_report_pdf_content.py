import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORT_STORE_PATH", str(tmp_path / "reports.json"))
    if "server.app" in sys.modules:
        del sys.modules["server.app"]
    return importlib.import_module("server.app")


def make_multi_agent_request(app_module):
    return app_module.ResearchRequest(
        task="Research topic",
        report_type="multi_agents",
        report_source="web",
        tone="Objective",
        repo_name="",
        branch_name="",
    )


@pytest.mark.asyncio
async def test_multi_agent_endpoint_exports_full_report(app_module):
    markdown = "# Complete report\n\nReport body."
    with (
        patch.object(app_module, "run_agent", new=AsyncMock(return_value=markdown)),
        patch.object(app_module, "write_md_to_word", new=AsyncMock(return_value="report.docx")) as word,
        patch.object(app_module, "write_md_to_pdf", new=AsyncMock(return_value="report.pdf")) as pdf,
    ):
        response = await app_module.write_report(
            make_multi_agent_request(app_module), "research-id"
        )

    assert response["report"] == markdown
    assert response["pdf_path"] == "report.pdf"
    word.assert_awaited_once_with(markdown, "research-id")
    pdf.assert_awaited_once_with(markdown, "research-id")


@pytest.mark.asyncio
async def test_empty_report_skips_document_export(app_module):
    with (
        patch.object(app_module, "run_agent", new=AsyncMock(return_value="")),
        patch.object(app_module, "write_md_to_word", new=AsyncMock()) as word,
        patch.object(app_module, "write_md_to_pdf", new=AsyncMock()) as pdf,
    ):
        response = await app_module.write_report(
            make_multi_agent_request(app_module), "research-id"
        )

    assert response["pdf_path"] == ""
    assert response["docx_path"] == ""
    word.assert_not_awaited()
    pdf.assert_not_awaited()