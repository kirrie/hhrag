"""PaperRepository 인덱싱·증분 갱신·벡터 검색·청크 조회 단위 테스트."""

from pathlib import Path

import pytest

from hhrag.config import IndexConfig
from hhrag.repository import PaperRepository
from tests.conftest import FakeEmbeddingModel, write_unicode_pdf


async def test_index_and_incremental_updates(
	sample_pdf_dir: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""초기 인덱싱 후 재실행 스킵, 신규 추가, 파일 삭제가 증분 반영된다."""
	repo = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	first = await repo.index_directory(embedding_model=fake_embedding)
	assert first == {"added": 3, "skipped": 0, "removed": 0, "total": 3}

	second = await repo.index_directory(embedding_model=fake_embedding)
	assert second == {"added": 0, "skipped": 3, "removed": 0, "total": 3}

	write_unicode_pdf(
		sample_pdf_dir / "new_methodology.pdf",
		"질적 연구 방법론과 심층 면담 코딩 절차를 다룬 석사학위 예비 논문입니다.",
	)
	third = await repo.index_directory(embedding_model=fake_embedding)
	assert third == {"added": 1, "skipped": 3, "removed": 0, "total": 4}

	(sample_pdf_dir / "urban_climate_study.pdf").unlink()
	fourth = await repo.index_directory(embedding_model=fake_embedding)
	assert fourth == {"added": 0, "skipped": 3, "removed": 1, "total": 3}


async def test_search_korean_and_english_queries(
	sample_pdf_dir: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""저장된 인덱스에서 한국어·영어 질의 모두 관련 논문 청크를 상위로 검색한다."""
	repo = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	await repo.index_directory(embedding_model=fake_embedding)

	reloaded = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	ko_hits = await reloaded.search(
		"한국어 bge-m3 다국어 임베딩 검색 증강 생성",
		top_k=2,
		embedding_model=fake_embedding,
	)
	assert ko_hits
	assert ko_hits[0].doc_name == "korean_rag_thesis"

	en_hits = await reloaded.search(
		"Transformer self-attention academic summarization",
		top_k=2,
		embedding_model=fake_embedding,
	)
	assert en_hits
	assert any(hit.doc_name == "english/attention_mechanisms" for hit in en_hits)


async def test_list_and_get_paper_chunks(
	sample_pdf_dir: Path,
	fake_embedding: FakeEmbeddingModel,
) -> None:
	"""인덱싱된 논문 목록 조회와 특정 논문 청크 열람 및 예외 처리를 검증한다."""
	repo = PaperRepository(IndexConfig(pdf_dir=sample_pdf_dir))
	await repo.index_directory(embedding_model=fake_embedding)

	papers = repo.list_papers()
	assert len(papers) == 3
	assert {paper.doc_name for paper in papers} == {
		"english/attention_mechanisms",
		"korean_rag_thesis",
		"urban_climate_study",
	}

	chunks = repo.get_paper_chunks("korean_rag_thesis", offset=0, limit=5)
	assert chunks
	assert "bge-m3" in chunks[0].text

	with pytest.raises(KeyError, match="찾을 수 없습니다"):
		repo.get_paper_chunks("missing_paper")

	with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
		await repo.search("   ", embedding_model=fake_embedding)
