"""CLI 원커맨드 준비와 MCP 서버 도구 호출까지 검증하는 E2E 테스트."""

import asyncio
from pathlib import Path
import subprocess

from mcp.server.mcpserver.exceptions import ToolError
import pytest

from hhrag.cli import run_cli
from hhrag.config import DOCS_FILE_NAME, INDEX_DIR_NAME, MANIFEST_FILE_NAME, IndexConfig
from hhrag.mcp_server import create_mcp_server
from hhrag.repository import PaperRepository
from tests.conftest import FakeEmbeddingModel


def _extract_text_content(tool_result: object) -> str:
	"""MCPServer.call_tool 반환값에서 첫 번째 텍스트 블록을 추출한다."""
	content = getattr(tool_result, "content", [])
	assert content
	return str(getattr(content[0], "text", ""))


async def _verify_mcp_tools(
	sample_pdf_dir: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""생성된 인덱스 위에서 MCP 서버의 목록·검색·청크 도구를 검증한다."""
	repo = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	server = create_mcp_server(repo=repo, embedding_model=fake_embedding)

	list_res = await server.call_tool("list_papers", {})
	list_text = _extract_text_content(list_res)
	assert "총 3편" in list_text
	assert "korean_rag_thesis" in list_text

	search_res = await server.call_tool(
		"search_papers",
		{"query": "도시 열섬 현상 녹지 공간", "top_k": 2},
	)
	search_text = _extract_text_content(search_res)
	assert "urban_climate_study" in search_text

	chunk_res = await server.call_tool(
		"get_paper_chunks",
		{"doc_name": "attention_mechanisms", "offset": 0, "limit": 2},
	)
	chunk_text = _extract_text_content(chunk_res)
	assert "Transformer self-attention" in chunk_text


def test_e2e_one_command_setup_and_mcp_tools(
	sample_pdf_dir: Path,
	tmp_path: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""성공 경로: 원커맨드 실행 -> 인덱스 및 MCP 등록 -> MCP 도구 검색·열람 검증."""
	executed_cmds: list[list[str]] = []

	def fake_runner(
		args: list[str],
		*,
		check: bool,
	) -> subprocess.CompletedProcess[str]:
		executed_cmds.append(args)
		return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

	exit_code = run_cli(
		["--pdf-dir", str(sample_pdf_dir)],
		embedding_model=fake_embedding,
		command_runner=fake_runner,
		project_dir=tmp_path,
	)
	assert exit_code == 0
	assert (sample_pdf_dir / INDEX_DIR_NAME / DOCS_FILE_NAME).exists()
	assert (sample_pdf_dir / INDEX_DIR_NAME / MANIFEST_FILE_NAME).exists()
	assert any(cmd[:4] == ["codex", "mcp", "add", "hhrag"] for cmd in executed_cmds)
	asyncio.run(_verify_mcp_tools(sample_pdf_dir, fake_embedding))


def test_e2e_missing_dir_fails_with_korean_message(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""실패 경로 1: 존재하지 않는 PDF 디렉토리를 전달하면 오류 코드 1을 반환한다."""
	missing = tmp_path / "does_not_exist"
	exit_code = run_cli(
		["setup", "--pdf-dir", str(missing), "--skip-mcp"],
		embedding_model=fake_embedding,
	)
	captured = capsys.readouterr()
	assert exit_code == 1
	assert "PDF 디렉토리를 찾을 수 없습니다" in captured.err


def test_e2e_empty_pdf_dir_fails_with_korean_message(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""실패 경로 2: PDF가 없는 빈 디렉토리를 전달하면 명확한 한국어 오류를 반환한다."""
	empty_dir = tmp_path / "empty_papers"
	empty_dir.mkdir()
	exit_code = run_cli(
		["index", "--pdf-dir", str(empty_dir)],
		embedding_model=fake_embedding,
	)
	captured = capsys.readouterr()
	assert exit_code == 1
	assert "인덱싱할 PDF 파일을 찾을 수 없습니다" in captured.err


async def test_e2e_mcp_empty_query_raises_tool_error(
	sample_pdf_dir: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""실패 경로 3: MCP search_papers에 빈 질의를 주면 한국어 ToolError가 발생한다."""
	repo = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	await repo.index_directory(embedding_model=fake_embedding)
	server = create_mcp_server(repo=repo, embedding_model=fake_embedding)

	with pytest.raises(ToolError, match="검색어\\(query\\)는 비어 있을 수 없습니다"):
		await server.call_tool("search_papers", {"query": "   "})
