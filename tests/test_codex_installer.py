"""codex_installer 모듈의 명령 인자 구성 및 서브프로세스 경계 단위 테스트."""

from pathlib import Path
import subprocess

import pytest

from hhrag.codex_installer import build_codex_mcp_args, install_codex_mcp


def test_build_codex_mcp_args_structure(tmp_path: Path) -> None:
	"""codex mcp add 명령 인자가 정확한 순서와 절대 경로로 생성된다."""
	project_dir = tmp_path / "proj"
	pdf_dir = tmp_path / "pdfs"
	project_dir.mkdir()
	pdf_dir.mkdir()

	args = build_codex_mcp_args("hhrag", project_dir, pdf_dir, "st-BAAI/bge-m3")
	assert args[:5] == ["codex", "mcp", "add", "hhrag", "--"]
	assert args[5:11] == [
		"uv",
		"--directory",
		str(project_dir.resolve()),
		"run",
		"hhrag",
		"serve",
	]
	assert args[11:] == [
		"--pdf-dir",
		str(pdf_dir.resolve()),
		"--embedding-model",
		"st-BAAI/bge-m3",
	]


def test_install_codex_mcp_success_invokes_remove_then_add(tmp_path: Path) -> None:
	"""설치 함수가 remove 후 add 순으로 명령을 실행한다."""
	recorded: list[tuple[list[str], bool]] = []

	def fake_runner(
		args: list[str],
		*,
		check: bool,
	) -> subprocess.CompletedProcess[str]:
		recorded.append((args, check))
		return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

	installed = install_codex_mcp(
		"hhrag",
		tmp_path,
		tmp_path,
		"st-BAAI/bge-m3",
		runner=fake_runner,
	)
	assert len(recorded) == 2
	assert recorded[0] == (["codex", "mcp", "remove", "hhrag"], False)
	assert recorded[1][0] == installed
	assert recorded[1][1] is True


def test_install_codex_mcp_failure_raises_korean_error(tmp_path: Path) -> None:
	"""codex mcp add 실행 실패 시 한국어 에러 메시지를 포함한 RuntimeError를 던진다."""

	def failing_runner(
		args: list[str],
		*,
		check: bool,
	) -> subprocess.CompletedProcess[str]:
		if check:
			raise subprocess.CalledProcessError(
				returncode=1,
				cmd=args,
				stderr="config write error",
			)
		return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")

	with pytest.raises(RuntimeError, match="Codex MCP 등록에 실패했습니다"):
		install_codex_mcp(
			"hhrag",
			tmp_path,
			tmp_path,
			"st-BAAI/bge-m3",
			runner=failing_runner,
		)


def test_build_codex_mcp_args_rejects_empty_name(tmp_path: Path) -> None:
	"""빈 MCP 이름을 전달하면 ValueError가 발생한다."""
	with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
		build_codex_mcp_args("  ", tmp_path, tmp_path, "st-BAAI/bge-m3")
