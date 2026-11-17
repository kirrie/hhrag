"""석사논문 RAG 검색 도구를 제공하는 MCP 서버 모듈."""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from paperqa import EmbeddingModel

from hhrag.config import DEFAULT_MCP_NAME, DEFAULT_READ_LIMIT, DEFAULT_TOP_K
from hhrag.repository import PaperRepository, PaperSummary, SearchResult


def _format_search_results(results: list[SearchResult], query: str) -> str:
	"""벡터 검색 결과를 에이전트가 읽기 좋은 마크다운 문자열로 포매팅한다."""
	if not results:
		return f"질의 '{query}'와 관련성이 높은 논문 내용을 찾지 못했습니다."
	lines = [f"### 검색 결과 ({len(results)}건) — 질의: {query}"]
	for idx, item in enumerate(results, start=1):
		lines.append(
			f"\n#### {idx}. {item.doc_name} ({item.chunk_name})\n"
			f"- 출처: {item.citation}\n"
			f"- 내용:\n{item.text.strip()}"
		)
	return "\n".join(lines)


def _format_paper_list(papers: list[PaperSummary]) -> str:
	"""인덱싱된 논문 목록을 마크다운 목록으로 포매팅한다."""
	if not papers:
		return "현재 인덱싱된 논문이 없습니다. 먼저 인덱싱 명령을 실행해 주세요."
	lines = [f"### 인덱싱된 논문 목록 (총 {len(papers)}편)"]
	for idx, paper in enumerate(papers, start=1):
		lines.append(
			f"{idx}. **{paper.doc_name}** | 청크 {paper.chunk_count}개 | {paper.citation}"
		)
	return "\n".join(lines)


def _format_chunks(chunks: list[SearchResult], doc_name: str, offset: int) -> str:
	"""특정 논문의 순차 청크 조회 결과를 마크다운 문자열로 포매팅한다."""
	if not chunks:
		return f"논문 '{doc_name}'의 offset={offset} 위치에 해당하는 청크가 없습니다."
	lines = [f"### 논문 원문 청크 — {chunks[0].doc_name} (시작 offset={offset})"]
	for idx, item in enumerate(chunks, start=offset):
		lines.append(f"\n#### 청크 #{idx} ({item.chunk_name})\n{item.text.strip()}")
	return "\n".join(lines)


def _register_search_tool(
	server: MCPServer,
	repo: PaperRepository,
	embedding_model: EmbeddingModel | None,
) -> None:
	"""벡터 검색 MCP 도구를 등록한다."""

	@server.tool(
		description=(
			"한국어 또는 영어 질의로 인덱싱된 석사논문 PDF에서 가장 관련성 높은 "
			"구절을 벡터 검색(bge-m3)으로 찾아 반환합니다."
		)
	)
	async def search_papers(query: str, top_k: int = DEFAULT_TOP_K) -> str:
		"""질의와 관련된 논문 구절 상위 top_k개를 검색한다."""
		try:
			results = await repo.search(
				query=query,
				top_k=top_k,
				embedding_model=embedding_model,
			)
		except (ValueError, RuntimeError, FileNotFoundError, TypeError) as exc:
			raise ToolError(str(exc)) from exc
		return _format_search_results(results, query)


def _register_read_tools(server: MCPServer, repo: PaperRepository) -> None:
	"""논문 목록 조회 및 원문 청크 열람 MCP 도구를 등록한다."""

	@server.tool(description="현재 벡터 인덱스에 저장된 논문 PDF 목록과 청크 수를 조회합니다.")
	def list_papers() -> str:
		"""인덱싱된 논문 목록을 반환한다."""
		try:
			papers = repo.list_papers()
		except (RuntimeError, FileNotFoundError, TypeError) as exc:
			raise ToolError(str(exc)) from exc
		return _format_paper_list(papers)

	@server.tool(
		description=(
			"특정 논문의 원문 청크를 순서대로 열람합니다. "
			"초록, 서론, 결론 등 전체 문맥을 이어서 읽을 때 사용하세요."
		)
	)
	def get_paper_chunks(
		doc_name: str,
		offset: int = 0,
		limit: int = DEFAULT_READ_LIMIT,
	) -> str:
		"""특정 논문의 청크 구간을 반환한다."""
		try:
			chunks = repo.get_paper_chunks(doc_name=doc_name, offset=offset, limit=limit)
		except (ValueError, KeyError, RuntimeError, TypeError) as exc:
			raise ToolError(str(exc)) from exc
		return _format_chunks(chunks, doc_name, offset)


def create_mcp_server(
	repo: PaperRepository,
	embedding_model: EmbeddingModel | None = None,
	server_name: str = DEFAULT_MCP_NAME,
) -> MCPServer:
	"""논문 RAG 도구가 등록된 MCPServer 인스턴스를 생성한다."""
	clean_name = server_name.strip() or DEFAULT_MCP_NAME
	server = MCPServer(
		name=clean_name,
		instructions=(
			"석사논문 준비를 위해 로컬에 저장된 논문 PDF들을 검색하고 원문 청크를 "
			"열람할 수 있는 RAG 서버입니다. search_papers로 관련 논문을 찾고 "
			"필요하면 get_paper_chunks로 전후 문맥을 확인하세요."
		),
	)
	_register_search_tool(server, repo, embedding_model)
	_register_read_tools(server, repo)
	return server
