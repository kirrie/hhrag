"""PDF 디렉토리 검증, 파일 탐색, SHA-256 해시 및 매니페스트 경계 I/O 모듈."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from hhrag.config import INDEX_DIR_NAME

HASH_CHUNK_BYTES = 65536
PDF_SUFFIX = ".pdf"
TEMP_SUFFIX = ".tmp"


def validate_pdf_dir(pdf_dir: Path) -> Path:
	"""PDF 디렉토리 존재 여부와 타입을 검증하고 정규화된 절대 경로를 반환한다."""
	if not isinstance(pdf_dir, Path):
		raise TypeError("pdf_dir는 pathlib.Path 인스턴스여야 합니다.")
	resolved = pdf_dir.expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"PDF 디렉토리를 찾을 수 없습니다: {resolved}")
	if not resolved.is_dir():
		raise NotADirectoryError(f"지정한 경로가 디렉토리가 아닙니다: {resolved}")
	return resolved


def _is_hidden_relative(path: Path, root: Path) -> bool:
	"""루트 기준 상대 경로에 숨김 폴더나 인덱스 폴더가 포함되었는지 판별한다."""
	rel_parts = path.relative_to(root).parts
	return any(part == INDEX_DIR_NAME or part.startswith(".") for part in rel_parts)


def scan_pdf_files(pdf_dir: Path) -> list[Path]:
	"""지정한 디렉토리 하위의 모든 PDF 파일을 정렬된 목록으로 반환한다."""
	validated_dir = validate_pdf_dir(pdf_dir)
	candidates = [
		path
		for path in validated_dir.rglob("*")
		if path.is_file()
		and path.suffix.lower() == PDF_SUFFIX
		and not _is_hidden_relative(path, validated_dir)
	]
	return sorted(candidates)


def compute_file_sha256(file_path: Path) -> str:
	"""파일의 SHA-256 해시를 스트리밍 방식으로 계산해 16진수 문자열로 반환한다."""
	if not file_path.is_file():
		raise FileNotFoundError(f"해시를 계산할 PDF 파일을 찾을 수 없습니다: {file_path}")
	digest = hashlib.sha256()
	with file_path.open("rb") as handle:
		while chunk := handle.read(HASH_CHUNK_BYTES):
			digest.update(chunk)
	return digest.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
	"""기존 매니페스트 JSON을 읽어 상대경로->해시 사전을 반환한다."""
	if not manifest_path.exists():
		return {}
	try:
		raw_text = manifest_path.read_text(encoding="utf-8")
		payload = json.loads(raw_text)
	except (OSError, json.JSONDecodeError) as exc:
		raise RuntimeError(
			f"매니페스트 파일을 읽을 수 없습니다 ({manifest_path}): {exc}"
		) from exc
	files_map = payload.get("files") if isinstance(payload, dict) else None
	if not isinstance(files_map, dict):
		raise ValueError(f"매니페스트 형식이 올바르지 않습니다: {manifest_path}")
	return {str(key): str(val) for key, val in files_map.items()}


def save_manifest_atomic(manifest_path: Path, entries: dict[str, str]) -> None:
	"""매니페스트 데이터를 임시 파일에 기록한 뒤 원자적으로 교체한다."""
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"updated_at": datetime.now(timezone.utc).isoformat(),
		"files": dict(sorted(entries.items())),
	}
	encoded = json.dumps(payload, ensure_ascii=False, indent="\t")
	tmp_path = manifest_path.with_name(f"{manifest_path.name}{TEMP_SUFFIX}")
	try:
		with tmp_path.open("w", encoding="utf-8") as handle:
			handle.write(encoded)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(tmp_path, manifest_path)
	except OSError as exc:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		raise RuntimeError(
			f"매니페스트 저장 중 파일 오류가 발생했습니다 ({manifest_path}): {exc}"
		) from exc
