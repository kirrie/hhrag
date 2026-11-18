"""hhrag CLI 진입점: 원커맨드 setup(인덱싱+Codex MCP 등록), index, install-mcp, serve."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import uuid

from paperqa import EmbeddingModel

from hhrag.codex_installer import CommandRunner, install_codex_mcp
from hhrag.config import (
	DEFAULT_CHUNK_CHARS,
	DEFAULT_CHUNK_OVERLAP,
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_MCP_NAME,
	IndexConfig,
)
from hhrag.mcp_server import create_mcp_server
from hhrag.pdf_scanner import validate_pdf_dir
from hhrag.progress import print_console_progress
from hhrag.repository import PaperRepository

KNOWN_COMMANDS = frozenset({"setup", "index", "install-mcp", "serve"})
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _JsonLogFormatter(logging.Formatter):
	"""UTC 타임스탬프와 실행 ID를 포함한 구조적 JSON 로그 포매터."""

	def __init__(self, run_id: str) -> None:
		"""실행 상관관계 ID를 초기화한다."""
		super().__init__()
		self._run_id = run_id

	def format(self, record: logging.LogRecord) -> str:
		"""로그 레코드를 단일 줄 JSON 문자열로 직렬화한다."""
		payload = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"level": record.levelname,
			"run_id": self._run_id,
			"logger": record.name,
			"message": record.getMessage(),
		}
		return json.dumps(payload, ensure_ascii=False)


def configure_logging(run_id: str) -> logging.Logger:
	"""MCP stdio 스트림을 오염시키지 않도록 stderr 전용 구조적 로거를 설정한다."""
	logger = logging.getLogger("hhrag")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(_JsonLogFormatter(run_id))
	logger.addHandler(handler)
	return logger


def _add_common_options(parser: argparse.ArgumentParser) -> None:
	"""모든 서브커맨드가 공유하는 CLI 옵션을 등록한다."""
	parser.add_argument(
		"--pdf-dir",
		type=Path,
		required=True,
		help="논문 PDF 파일이 저장된 디렉토리 경로",
	)
	parser.add_argument(
		"--embedding-model",
		default=DEFAULT_EMBEDDING_MODEL,
		help=f"사용할 PaperQA2 임베딩 모델 (기본: {DEFAULT_EMBEDDING_MODEL})",
	)
	parser.add_argument(
		"--mcp-name",
		default=DEFAULT_MCP_NAME,
		help=f"Codex에 등록할 MCP 서버 이름 (기본: {DEFAULT_MCP_NAME})",
	)
	parser.add_argument(
		"--chunk-chars",
		type=int,
		default=DEFAULT_CHUNK_CHARS,
		help=f"청크 당 최대 문자 수 (기본: {DEFAULT_CHUNK_CHARS})",
	)
	parser.add_argument(
		"--chunk-overlap",
		type=int,
		default=DEFAULT_CHUNK_OVERLAP,
		help=f"청크 간 겹침 문자 수 (기본: {DEFAULT_CHUNK_OVERLAP})",
	)


def build_parser() -> argparse.ArgumentParser:
	"""hhrag CLI 인수 파서를 생성한다."""
	parser = argparse.ArgumentParser(
		prog="hhrag",
		description=(
			"PaperQA2와 bge-m3 기반 석사논문 PDF 벡터 인덱싱 및 Codex MCP 도구 "
			"(서브커맨드 생략 시 setup 실행)"
		),
	)
	subparsers = parser.add_subparsers(dest="command", required=True)
	setup_parser = subparsers.add_parser(
		"setup",
		help="PDF 벡터 인덱싱과 Codex MCP 등록을 한 번에 수행",
	)
	_add_common_options(setup_parser)
	setup_parser.add_argument(
		"--skip-mcp",
		action="store_true",
		help="Codex MCP 등록을 건너뛰고 벡터 인덱싱만 수행",
	)
	for cmd, desc in [
		("index", "지정 디렉토리의 PDF 파일만 증분 벡터 인덱싱"),
		("install-mcp", "Codex에 hhrag MCP 서버만 등록"),
		("serve", "Codex용 stdio MCP 서버 실행"),
	]:
		sub = subparsers.add_parser(cmd, help=desc)
		_add_common_options(sub)
	return parser


def _normalize_argv(argv: list[str] | None) -> list[str]:
	"""서브커맨드가 생략된 경우 기본 명령인 setup을 자동으로 보완한다."""
	raw_args = list(sys.argv[1:] if argv is None else argv)
	if not raw_args:
		return raw_args
	first = raw_args[0]
	if first in KNOWN_COMMANDS or first in {"-h", "--help"}:
		return raw_args
	return ["setup", *raw_args]


def _build_config(args: argparse.Namespace) -> IndexConfig:
	"""파싱된 CLI 인자로 검증된 IndexConfig를 생성한다."""
	validated_dir = validate_pdf_dir(args.pdf_dir)
	return IndexConfig(
		pdf_dir=validated_dir,
		embedding_model=args.embedding_model,
		mcp_name=args.mcp_name,
		chunk_chars=args.chunk_chars,
		chunk_overlap=args.chunk_overlap,
	)


def _run_indexing(
	repo: PaperRepository,
	embedding_model: EmbeddingModel | None,
	logger: logging.Logger,
) -> dict[str, int]:
	"""PDF 벡터 인덱싱을 실행하고 비어 있는 인덱스 여부를 검증한다."""
	logger.info("PDF 벡터 인덱싱을 시작합니다.")
	stats = asyncio.run(
		repo.index_directory(
			embedding_model=embedding_model,
			on_progress=print_console_progress,
		)
	)
	if stats["total"] == 0:
		raise ValueError(
			"인덱싱할 PDF 파일을 찾을 수 없습니다. 디렉토리 내 .pdf 파일을 확인해 주세요."
		)
	logger.info(
		"인덱싱 완료: 신규=%d, 스킵=%d, 제거=%d, 전체=%d",
		stats["added"],
		stats["skipped"],
		stats["removed"],
		stats["total"],
	)
	return stats


def _execute_command(
	args: argparse.Namespace,
	config: IndexConfig,
	embedding_model: EmbeddingModel | None,
	command_runner: CommandRunner | None,
) -> None:
	"""검증된 설정을 바탕으로 요청된 하위 명령을 실행한다."""
	logger = logging.getLogger("hhrag")
	repo = PaperRepository(config)
	target_project = getattr(args, "project_dir", PROJECT_ROOT)
	if args.command in {"setup", "index"}:
		stats = _run_indexing(repo, embedding_model, logger)
		print(
			f"인덱싱 완료 (신규 {stats['added']}편, 기존 {stats['skipped']}편, "
			f"제거 {stats['removed']}편, 총 {stats['total']}편)"
		)
	if args.command == "install-mcp" or (args.command == "setup" and not args.skip_mcp):
		install_codex_mcp(
			mcp_name=config.mcp_name,
			project_dir=target_project,
			pdf_dir=config.pdf_dir,
			embedding_model=config.embedding_model,
			runner=command_runner,
		)
		logger.info("Codex MCP 서버 등록 완료: %s", config.mcp_name)
		print(f"Codex MCP '{config.mcp_name}' 등록 완료 (대상 경로: {config.pdf_dir})")
	if args.command == "serve":
		logger.info("MCP stdio 서버를 시작합니다: %s", config.pdf_dir)
		server = create_mcp_server(
			repo=repo,
			embedding_model=embedding_model,
			server_name=config.mcp_name,
		)
		server.run(transport="stdio")


def run_cli(
	argv: list[str] | None = None,
	*,
	embedding_model: EmbeddingModel | None = None,
	command_runner: CommandRunner | None = None,
	project_dir: Path | None = None,
) -> int:
	"""CLI 인수를 파싱해 명령을 실행하고 종료 코드를 반환한다."""
	run_id = uuid.uuid4().hex[:12]
	logger = configure_logging(run_id)
	parser = build_parser()
	normalized = _normalize_argv(argv)
	args = parser.parse_args(normalized)
	if project_dir is not None:
		args.project_dir = project_dir
	try:
		config = _build_config(args)
		_execute_command(args, config, embedding_model, command_runner)
		return 0
	except (
		ValueError,
		FileNotFoundError,
		NotADirectoryError,
		TypeError,
		RuntimeError,
		KeyError,
	) as exc:
		logger.error("실행 실패: %s", exc)
		print(f"오류: {exc}", file=sys.stderr)
		return 1


def main() -> None:
	"""콘솔 스크립트 엔트리포인트."""
	raise SystemExit(run_cli())


if __name__ == "__main__":
	main()
