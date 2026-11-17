"""pdf_scanner 모듈의 경로 검증, PDF 탐색, 해시 및 매니페스트 단위 테스트."""

from pathlib import Path

import pytest

from hhrag.pdf_scanner import (
	compute_file_sha256,
	load_manifest,
	save_manifest_atomic,
	scan_pdf_files,
	validate_pdf_dir,
)


def test_validate_pdf_dir_success(tmp_path: Path) -> None:
	"""존재하는 디렉토리를 전달하면 정규화된 절대 경로를 반환한다."""
	target = tmp_path / "pdfs"
	target.mkdir()
	resolved = validate_pdf_dir(target)
	assert resolved == target.resolve()


def test_validate_pdf_dir_missing_raises(tmp_path: Path) -> None:
	"""존재하지 않는 경로는 FileNotFoundError를 발생시킨다."""
	missing = tmp_path / "nonexistent"
	with pytest.raises(FileNotFoundError):
		validate_pdf_dir(missing)


def test_validate_pdf_dir_file_raises(tmp_path: Path) -> None:
	"""디렉토리가 아닌 파일 경로를 전달하면 NotADirectoryError를 발생시킨다."""
	file_path = tmp_path / "sample.pdf"
	file_path.write_bytes(b"%PDF-1.4")
	with pytest.raises(NotADirectoryError):
		validate_pdf_dir(file_path)


def test_scan_pdf_files_ignores_hidden_and_non_pdf(tmp_path: Path) -> None:
	"""숨김 폴더(.hhrag 등)와 비-PDF 파일은 제외하고 PDF만 정렬해 반환한다."""
	(tmp_path / "b_paper.PDF").write_bytes(b"pdf")
	(tmp_path / "sub").mkdir()
	(tmp_path / "sub" / "a_paper.pdf").write_bytes(b"pdf")
	(tmp_path / "notes.txt").write_text("메모", encoding="utf-8")
	(tmp_path / ".hhrag").mkdir()
	(tmp_path / ".hhrag" / "cached.pdf").write_bytes(b"pdf")

	found = scan_pdf_files(tmp_path)
	rel_names = [path.relative_to(tmp_path).as_posix() for path in found]
	assert rel_names == ["b_paper.PDF", "sub/a_paper.pdf"]


def test_compute_file_sha256_deterministic(tmp_path: Path) -> None:
	"""동일한 바이트 내용에 대해 결정적인 SHA-256 해시를 반환한다."""
	file_a = tmp_path / "a.pdf"
	file_b = tmp_path / "b.pdf"
	file_a.write_bytes(b"thesis-content")
	file_b.write_bytes(b"thesis-content")
	hash_a = compute_file_sha256(file_a)
	assert len(hash_a) == 64
	assert hash_a == compute_file_sha256(file_b)


def test_manifest_atomic_save_and_load(tmp_path: Path) -> None:
	"""매니페스트를 원자적으로 저장한 뒤 다시 읽으면 동일한 항목이 복원된다."""
	manifest_path = tmp_path / ".hhrag" / "manifest.json"
	assert load_manifest(manifest_path) == {}
	entries = {"paper1.pdf": "abc123", "sub/paper2.pdf": "def456"}
	save_manifest_atomic(manifest_path, entries)
	assert manifest_path.exists()
	assert not manifest_path.with_name("manifest.json.tmp").exists()
	assert load_manifest(manifest_path) == entries
