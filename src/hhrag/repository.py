"""PaperQA2 기반 논문 PDF 벡터 인덱싱·영속화·검색 퍼사드 모듈."""

from dataclasses import dataclass
import os
from pathlib import Path
import pickle

from paperqa import Docs, EmbeddingModel, Settings, Text, embedding_model_factory

from hhrag.config import (
	DEFAULT_READ_LIMIT,
	DEFAULT_TOP_K,
	MAX_READ_LIMIT,
	MAX_TOP_K,
	IndexConfig,
	create_paperqa_settings,
)
from hhrag.pdf_scanner import (
	compute_file_sha256,
	load_manifest,
	save_manifest_atomic,
	scan_pdf_files,
	validate_pdf_dir,
)

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


class PaperRepository:
	"""PaperQA2 Docs, 로컬 임베딩 및 디스크 영속화를 캡슐화하는 퍼사드."""

	def __init__(self, config: IndexConfig) -> None:
		"""설정을 검증하고 저장소 인스턴스를 초기화한다."""
		self._config = config
		self._cached_embedding: EmbeddingModel | None = None

	def _resolve_embedding(self, embedding_model: EmbeddingModel | None) -> EmbeddingModel:
		"""외부 주입 모델이 없으면 설정된 로컬 임베딩 모델을 지연 로드해 캐시한다."""
		if embedding_model is not None:
			return embedding_model
		if self._cached_embedding is None:
			self._cached_embedding = embedding_model_factory(self._config.embedding_model)
		return self._cached_embedding

	def _load_docs(self) -> Docs:
		"""디스크에 저장된 Docs 인덱스를 로드하고, 없으면 빈 Docs를 반환한다."""
		docs_path = self._config.docs_path
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

	def _save_docs_atomic(self, docs: Docs) -> None:
		"""Docs 객체를 임시 파일에 직렬화한 뒤 원자적으로 교체한다."""
		docs_path = self._config.docs_path
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

	def _collect_current_hashes(self, validated_dir: Path) -> dict[str, str]:
		"""디렉토리 내 PDF 파일을 스캔해 상대경로->SHA256 매핑을 생성한다."""
		pdf_files = scan_pdf_files(validated_dir)
		return {
			pdf_path.relative_to(validated_dir).as_posix(): compute_file_sha256(pdf_path)
			for pdf_path in pdf_files
		}

	def _prune_stale_docs(
		self,
		docs: Docs,
		old_manifest: dict[str, str],
		current_hashes: dict[str, str],
	) -> int:
		"""삭제되었거나 내용이 바뀐 기존 문서를 Docs 인덱스에서 제거한다."""
		active_hashes = set(current_hashes.values())
		stale_keys = {
			old_hash
			for rel_path, old_hash in old_manifest.items()
			if current_hashes.get(rel_path) != old_hash
			and old_hash not in active_hashes
			and old_hash in docs.docs
		}
		for dockey in stale_keys:
			docs.delete(dockey=dockey)
		if stale_keys:
			docs.texts_index.clear()
			docs.deleted_dockeys.clear()
		return len(stale_keys)

	async def _add_single_pdf(
		self,
		docs: Docs,
		rel_path: str,
		file_hash: str,
		model: EmbeddingModel,
	) -> bool:
		"""단일 PDF를 PaperQA2 Docs에 추가하고 임베딩을 계산한다."""
		if file_hash in docs.docs:
			return False
		validated_dir = validate_pdf_dir(self._config.pdf_dir)
		pdf_path = validated_dir / rel_path
		docname = Path(rel_path).with_suffix("").as_posix()
		citation = f"{pdf_path.name} (경로: {rel_path})"
		settings = create_paperqa_settings(self._config)
		try:
			added_name = await docs.aadd(
				pdf_path,
				citation=citation,
				docname=docname,
				dockey=file_hash,
				settings=settings,
				embedding_model=model,
			)
		except ValueError as exc:
			raise ValueError(f"PDF 텍스트를 파싱할 수 없습니다 ({rel_path}): {exc}") from exc
		return added_name is not None

	async def index_directory(
		self,
		embedding_model: EmbeddingModel | None = None,
	) -> dict[str, int]:
		"""PDF 디렉토리를 스캔해 신규·변경 파일만 증분 임베딩 및 인덱싱한다."""
		validated_dir = validate_pdf_dir(self._config.pdf_dir)
		current_hashes = self._collect_current_hashes(validated_dir)
		old_manifest = load_manifest(self._config.manifest_path)
		docs = self._load_docs()
		removed = self._prune_stale_docs(docs, old_manifest, current_hashes)
		model = self._resolve_embedding(embedding_model)
		added = 0
		skipped = 0
		for rel_path, file_hash in current_hashes.items():
			if await self._add_single_pdf(docs, rel_path, file_hash, model):
				added += 1
			else:
				skipped += 1
		if docs.texts:
			await docs._build_texts_index(model, with_enrichment=False)
		manifest_changed = old_manifest != current_hashes
		if added > 0 or removed > 0 or manifest_changed or not self._config.docs_path.exists():
			self._save_docs_atomic(docs)
			save_manifest_atomic(self._config.manifest_path, current_hashes)
		return {
			"added": added,
			"skipped": skipped,
			"removed": removed,
			"total": len(docs.docs),
		}

	async def search(
		self,
		query: str,
		top_k: int = DEFAULT_TOP_K,
		embedding_model: EmbeddingModel | None = None,
	) -> list[SearchResult]:
		"""질의어와 가장 관련성이 높은 논문 청크를 벡터 유사도로 검색한다."""
		clean_query = query.strip()
		if not clean_query:
			raise ValueError("검색어(query)는 비어 있을 수 없습니다.")
		if top_k < 1 or top_k > MAX_TOP_K:
			raise ValueError(f"top_k는 1 이상 {MAX_TOP_K} 이하의 정수여야 합니다.")
		docs = self._load_docs()
		if not docs.texts:
			return []
		model = self._resolve_embedding(embedding_model)
		settings = create_paperqa_settings(self._config)
		matches = await docs.retrieve_texts(
			clean_query,
			k=top_k,
			settings=settings,
			embedding_model=model,
		)
		return [_to_search_result(item) for item in matches]

	def list_papers(self) -> list[PaperSummary]:
		"""현재 인덱싱된 모든 논문의 이름, 키, 출처, 청크 수를 반환한다."""
		docs = self._load_docs()
		if not docs.docs:
			return []
		chunk_counts: dict[str, int] = {}
		for text_obj in docs.texts:
			key = str(text_obj.doc.dockey)
			chunk_counts[key] = chunk_counts.get(key, 0) + 1
		summaries = [
			PaperSummary(
				doc_name=doc.docname,
				dockey=str(doc.dockey),
				citation=doc.citation,
				chunk_count=chunk_counts.get(str(dockey), 0),
			)
			for dockey, doc in docs.docs.items()
		]
		return sorted(summaries, key=lambda item: item.doc_name)

	def get_paper_chunks(
		self,
		doc_name: str,
		offset: int = 0,
		limit: int = DEFAULT_READ_LIMIT,
	) -> list[SearchResult]:
		"""지정한 논문의 청크 목록을 offset과 limit 범위만큼 순서대로 반환한다."""
		clean_name = doc_name.strip()
		if not clean_name:
			raise ValueError("논문 이름(doc_name)은 비어 있을 수 없습니다.")
		if offset < 0:
			raise ValueError("offset은 0 이상이어야 합니다.")
		if limit < 1 or limit > MAX_READ_LIMIT:
			raise ValueError(f"limit은 1 이상 {MAX_READ_LIMIT} 이하이어야 합니다.")
		docs = self._load_docs()
		target_key = _find_matching_dockey(docs, clean_name)
		doc_chunks = [
			item for item in docs.texts if str(item.doc.dockey) == target_key
		]
		return [_to_search_result(item) for item in doc_chunks[offset : offset + limit]]


def _to_search_result(text_obj: Text) -> SearchResult:
	"""PaperQA2 Text 객체를 내부 SearchResult 데이터클래스로 변환한다."""
	return SearchResult(
		doc_name=text_obj.doc.docname,
		chunk_name=text_obj.name,
		citation=text_obj.doc.citation,
		text=text_obj.text,
		dockey=str(text_obj.doc.dockey),
	)


def _find_matching_dockey(docs: Docs, query_name: str) -> str:
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
