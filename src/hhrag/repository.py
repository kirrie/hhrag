"""PaperQA2 기반 논문 PDF 벡터 인덱싱·영속화·검색 퍼사드 모듈."""

from pathlib import Path
import time

from paperqa import Docs, EmbeddingModel, embedding_model_factory

from hhrag.config import (
	DEFAULT_READ_LIMIT,
	DEFAULT_TOP_K,
	MAX_READ_LIMIT,
	MAX_TOP_K,
	IndexConfig,
	create_paperqa_settings,
)
from hhrag.docs_store import (
	PaperSummary,
	SearchResult,
	find_matching_dockey,
	load_docs,
	save_docs_atomic,
	to_search_result,
)
from hhrag.pdf_scanner import (
	compute_file_sha256,
	load_manifest,
	save_manifest_atomic,
	scan_pdf_files,
	validate_pdf_dir,
)
from hhrag.progress import (
	STATUS_ADDED,
	STATUS_FAILED,
	STATUS_LOADING_MODEL,
	STATUS_PROCESSING,
	STATUS_SAVING,
	STATUS_SCANNING,
	STATUS_SKIPPED,
	IndexProgress,
	ProgressCallback,
	emit_progress,
)

__all__ = ["PaperRepository", "PaperSummary", "SearchResult"]


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

	async def _process_pdf_entry(
		self,
		docs: Docs,
		entry: tuple[int, int, str, str],
		model: EmbeddingModel,
		on_progress: ProgressCallback | None,
	) -> bool:
		"""단일 PDF 항목의 진행 상황을 보고하며 증분 인덱싱한다."""
		idx, total, rel_path, file_hash = entry
		if file_hash in docs.docs:
			emit_progress(on_progress, IndexProgress(idx, total, rel_path, STATUS_SKIPPED))
			return False
		emit_progress(on_progress, IndexProgress(idx, total, rel_path, STATUS_PROCESSING))
		started = time.monotonic()
		try:
			added = await self._add_single_pdf(docs, rel_path, file_hash, model)
		except ValueError as exc:
			elapsed = time.monotonic() - started
			emit_progress(
				on_progress,
				IndexProgress(idx, total, rel_path, STATUS_FAILED, elapsed, str(exc)),
			)
			raise
		elapsed = time.monotonic() - started
		status = STATUS_ADDED if added else STATUS_SKIPPED
		emit_progress(on_progress, IndexProgress(idx, total, rel_path, status, elapsed))
		return added

	async def index_directory(
		self,
		embedding_model: EmbeddingModel | None = None,
		on_progress: ProgressCallback | None = None,
	) -> dict[str, int]:
		"""PDF 디렉토리를 스캔해 신규·변경 파일만 증분 임베딩 및 인덱싱한다."""
		validated_dir = validate_pdf_dir(self._config.pdf_dir)
		current_hashes = self._collect_current_hashes(validated_dir)
		total_files = len(current_hashes)
		emit_progress(on_progress, IndexProgress(0, total_files, "", STATUS_SCANNING))
		old_manifest = load_manifest(self._config.manifest_path)
		docs = load_docs(self._config.docs_path)
		removed = self._prune_stale_docs(docs, old_manifest, current_hashes)
		emit_progress(
			on_progress,
			IndexProgress(
				0,
				total_files,
				"",
				STATUS_LOADING_MODEL,
				detail=self._config.embedding_model,
			),
		)
		model = self._resolve_embedding(embedding_model)
		added, skipped = 0, 0
		for idx, (rel_path, file_hash) in enumerate(current_hashes.items(), start=1):
			entry = (idx, total_files, rel_path, file_hash)
			if await self._process_pdf_entry(docs, entry, model, on_progress):
				added += 1
			else:
				skipped += 1
		if docs.texts:
			await docs._build_texts_index(model, with_enrichment=False)
		manifest_changed = old_manifest != current_hashes
		if added > 0 or removed > 0 or manifest_changed or not self._config.docs_path.exists():
			save_ev = IndexProgress(
				0,
				total_files,
				"",
				STATUS_SAVING,
				detail=f"총 {len(docs.docs)}편",
			)
			emit_progress(on_progress, save_ev)
			save_docs_atomic(self._config.docs_path, docs)
			save_manifest_atomic(self._config.manifest_path, current_hashes)
		return {"added": added, "skipped": skipped, "removed": removed, "total": len(docs.docs)}

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
		docs = load_docs(self._config.docs_path)
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
		return [to_search_result(item) for item in matches]

	def list_papers(self) -> list[PaperSummary]:
		"""현재 인덱싱된 모든 논문의 이름, 키, 출처, 청크 수를 반환한다."""
		docs = load_docs(self._config.docs_path)
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
		docs = load_docs(self._config.docs_path)
		target_key = find_matching_dockey(docs, clean_name)
		doc_chunks = [
			item for item in docs.texts if str(item.doc.dockey) == target_key
		]
		return [to_search_result(item) for item in doc_chunks[offset : offset + limit]]
