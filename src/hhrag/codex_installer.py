"""Codex CLI에 hhrag MCP 서버를 자동 등록하는 경계 레이어 모듈."""

from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess

DEFAULT_CODEX_BIN = "codex"
DEFAULT_UV_BIN = "uv"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_codex_mcp_args(
	mcp_name: str,
	project_dir: Path,
	pdf_dir: Path,
	embedding_model: str,
) -> list[str]:
	"""`codex mcp add` 실행에 필요한 인자 목록을 생성한다."""
	clean_name = mcp_name.strip()
	if not clean_name:
		raise ValueError("MCP 서버 이름(mcp_name)은 비어 있을 수 없습니다.")
	if not embedding_model.strip():
		raise ValueError("임베딩 모델 이름은 비어 있을 수 없습니다.")
	resolved_project = project_dir.expanduser().resolve()
	resolved_pdf = pdf_dir.expanduser().resolve()
	return [
		DEFAULT_CODEX_BIN,
		"mcp",
		"add",
		clean_name,
		"--",
		DEFAULT_UV_BIN,
		"--directory",
		str(resolved_project),
		"run",
		"hhrag",
		"serve",
		"--pdf-dir",
		str(resolved_pdf),
		"--embedding-model",
		embedding_model.strip(),
	]


def _default_runner(
	args: list[str],
	*,
	check: bool,
) -> subprocess.CompletedProcess[str]:
	"""표준 subprocess.run 래퍼."""
	return subprocess.run(
		args,
		check=check,
		capture_output=True,
		text=True,
	)


def install_codex_mcp(
	mcp_name: str,
	project_dir: Path,
	pdf_dir: Path,
	embedding_model: str,
	runner: CommandRunner | None = None,
) -> list[str]:
	"""기존 동명 MCP 설정을 정리한 뒤 Codex에 hhrag MCP 서버를 등록한다."""
	add_args = build_codex_mcp_args(mcp_name, project_dir, pdf_dir, embedding_model)
	exec_runner = runner or _default_runner
	if runner is None and shutil.which(DEFAULT_CODEX_BIN) is None:
		raise FileNotFoundError(
			"codex CLI를 찾을 수 없습니다. Codex 설치 및 PATH 설정을 확인해 주세요."
		)
	remove_args = [DEFAULT_CODEX_BIN, "mcp", "remove", mcp_name.strip()]
	try:
		# 기존 동일 이름 등록이 있더라도 멱등하게 갱신하기 위해 먼저 제거를 시도한다.
		exec_runner(remove_args, check=False)
	except OSError as exc:
		raise RuntimeError(f"기존 Codex MCP 제거 실행 중 오류가 발생했습니다: {exc}") from exc

	try:
		exec_runner(add_args, check=True)
	except subprocess.CalledProcessError as exc:
		detail = (exc.stderr or exc.stdout or str(exc)).strip()
		raise RuntimeError(f"Codex MCP 등록에 실패했습니다: {detail}") from exc
	except OSError as exc:
		raise RuntimeError(f"Codex CLI 실행 중 시스템 오류가 발생했습니다: {exc}") from exc
	return add_args
