"""PaperQA2 Docs 영속화 및 검색 결과 모델/변환 경계 모듈."""

from dataclasses import dataclass
import os
from pathlib import Path
import pickle

from paperqa import Docs, Text

TEMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class SearchResult:
	"""벡터 검색 또는 청크 조회로 반환되는 논문 청크 단위 결과."""

	doc_name: str
	chunk_name: str
	citation: str
	text: str
	dockey: str


@dataclass(frozen=True)
class PaperSummary:
	"""인덱싱된 개별 논문의 요약 메타데이터."""

	doc_name: str
	dockey: str
	citation: str
	chunk_count: int


def load_docs(docs_path: Path) -> Docs:
	"""디스크에 저장된 Docs 인덱스를 로드하고, 없으면 빈 Docs를 반환한다."""
	if not docs_path.exists():
		return Docs()
	try:
		raw_bytes = docs_path.read_bytes()
		loaded = pickle.loads(raw_bytes)
	except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
		raise RuntimeError(
			f"저장된 인덱스 파일이 손상되어 읽을 수 없습니다 ({docs_path}): {exc}"
		) from exc
	if not isinstance(loaded, Docs):
		raise TypeError(f"인덱스 파일 형식이 올바르지 않습니다: {docs_path}")
	return loaded


def save_docs_atomic(docs_path: Path, docs: Docs) -> None:
	"""Docs 객체를 임시 파일에 직렬화한 뒤 원자적으로 교체한다."""
	docs_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = docs_path.with_name(f"{docs_path.name}{TEMP_SUFFIX}")
	try:
		payload = pickle.dumps(docs)
		with tmp_path.open("wb") as handle:
			handle.write(payload)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(tmp_path, docs_path)
	except (OSError, pickle.PicklingError) as exc:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		raise RuntimeError(
			f"인덱스 파일 저장 중 오류가 발생했습니다 ({docs_path}): {exc}"
		) from exc


def to_search_result(text_obj: Text) -> SearchResult:
	"""PaperQA2 Text 객체를 내부 SearchResult 데이터클래스로 변환한다."""
	return SearchResult(
		doc_name=text_obj.doc.docname,
		chunk_name=text_obj.name,
		citation=text_obj.doc.citation,
		text=text_obj.text,
		dockey=str(text_obj.doc.dockey),
	)


def find_matching_dockey(docs: Docs, query_name: str) -> str:
	"""문서명 정확 일치 또는 고유 부분 일치로 dockey를 찾는다."""
	if not docs.docs:
		raise KeyError("인덱싱된 논문이 없습니다. 먼저 인덱싱을 실행해 주세요.")
	for dockey, doc in docs.docs.items():
		if doc.docname == query_name:
			return str(dockey)
	lowered = query_name.lower()
	partial = [
		(str(dockey), doc.docname)
		for dockey, doc in docs.docs.items()
		if lowered in doc.docname.lower() or lowered in doc.citation.lower()
	]
	if not partial:
		raise KeyError(f"'{query_name}'와 일치하는 논문을 찾을 수 없습니다.")
	if len(partial) > 1:
		names = ", ".join(name for _, name in partial[:5])
		raise ValueError(
			f"'{query_name}'와 일치하는 논문이 여러 개입니다: {names}"
		)
	return partial[0][0]
