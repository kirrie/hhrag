"""테스트 공통 픽스처: 결정적 Fake 임베딩과 순수 Python 유니코드 PDF 생성기."""

import hashlib
import math
from pathlib import Path

from paperqa import EmbeddingModel
import pytest

VECTOR_DIM = 64


class FakeEmbeddingModel(EmbeddingModel):
	"""외부 모델 다운로드 없이 문자 n-gram 해시로 결정적 벡터를 생성하는 Fake 임베딩."""

	name: str = "fake-bge-m3"
	ndim: int | None = VECTOR_DIM

	async def embed_documents(self, texts: list[str]) -> list[list[float]]:
		"""각 텍스트를 정규화된 고정 차원 벡터로 변환한다."""
		return [_hash_embed(text) for text in texts]


def _hash_embed(text: str) -> list[float]:
	"""공백 토큰과 문자 2-gram을 해시 버킷에 누적한 L2 정규화 벡터를 반환한다."""
	vec = [0.0] * VECTOR_DIM
	cleaned = text.strip().lower()
	if not cleaned:
		vec[0] = 1.0
		return vec
	tokens = cleaned.split()
	ngrams = [cleaned[i : i + 2] for i in range(max(1, len(cleaned) - 1))]
	for feature in [*tokens, *ngrams]:
		digest = hashlib.sha256(feature.encode("utf-8")).digest()
		bucket = int.from_bytes(digest[:2], "big") % VECTOR_DIM
		sign = 1.0 if digest[2] % 2 == 0 else -1.0
		vec[bucket] += sign
	norm = math.sqrt(sum(val * val for val in vec)) or 1.0
	return [val / norm for val in vec]


def _build_tounicode_cmap(char_map: dict[str, str]) -> bytes:
	"""유니코드 문자를 CID 코드로 매핑하는 PDF ToUnicode CMap 스트림을 생성한다."""
	bfchars = [
		f"<{cid}> <{ch.encode('utf-16-be').hex().upper()}>"
		for ch, cid in char_map.items()
	]
	cmap_text = (
		"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
		"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
		"/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
		"1 begincodespacerange\n<0001> <FFFF>\nendcodespacerange\n"
		f"{len(bfchars)} beginbfchar\n" + "\n".join(bfchars) + "\nendbfchar\n"
		"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
	)
	return cmap_text.encode("latin1")


def _build_pdf_content_stream(lines: list[str], char_map: dict[str, str]) -> bytes:
	"""각 줄을 CID hex 문자열로 변환한 PDF 페이지 콘텐츠 스트림을 생성한다."""
	parts = ["BT", "/F1 12 Tf", "72 720 Td"]
	for line in lines:
		encoded = "".join(char_map[ch] for ch in line)
		parts.append(f"<{encoded}> Tj")
		parts.append("0 -16 Td")
	parts.append("ET")
	return "\n".join(parts).encode("latin1")


def write_unicode_pdf(path: Path, text: str) -> Path:
	"""pypdf 및 PaperQA2가 한글·영어 텍스트를 추출할 수 있는 PDF 파일을 작성한다."""
	lines = [line for line in text.splitlines() if line] or ["empty"]
	unique_chars = list(dict.fromkeys("".join(lines)))
	char_map = {ch: f"{idx:04X}" for idx, ch in enumerate(unique_chars, start=1)}
	cmap = _build_tounicode_cmap(char_map)
	content = _build_pdf_content_stream(lines, char_map)
	objs = [
		b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
		b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
		(
			b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
			b"/Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >> endobj\n"
		),
		(
			b"4 0 obj << /Type /Font /Subtype /Type0 /BaseFont /FakeUnicode "
			b"/Encoding /Identity-H /DescendantFonts [7 0 R] "
			b"/ToUnicode 5 0 R >> endobj\n"
		),
		f"5 0 obj << /Length {len(cmap)} >> stream\n".encode("latin1")
		+ cmap
		+ b"\nendstream\nendobj\n",
		f"6 0 obj << /Length {len(content)} >> stream\n".encode("latin1")
		+ content
		+ b"\nendstream\nendobj\n",
		(
			b"7 0 obj << /Type /Font /Subtype /CIDFontType2 /BaseFont /FakeUnicode "
			b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
			b"/DW 1000 >> endobj\n"
		),
	]
	buf = bytearray(b"%PDF-1.4\n")
	offsets: list[int] = []
	for obj in objs:
		offsets.append(len(buf))
		buf.extend(obj)
	xref_pos = len(buf)
	buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin1"))
	for off in offsets:
		buf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
	trailer = (
		f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
		f"startxref\n{xref_pos}\n%%EOF\n"
	).encode("latin1")
	buf.extend(trailer)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(bytes(buf))
	return path


@pytest.fixture
def fake_embedding() -> FakeEmbeddingModel:
	"""결정적 Fake 임베딩 모델 픽스처를 반환한다."""
	return FakeEmbeddingModel()


@pytest.fixture
def sample_pdf_dir(tmp_path: Path) -> Path:
	"""한국어 및 영어 논문 PDF 3편이 포함된 임시 디렉토리를 생성한다."""
	papers_dir = tmp_path / "papers"
	papers_dir.mkdir()
	write_unicode_pdf(
		papers_dir / "korean_rag_thesis.pdf",
		"본 석사논문은 한국어 학술 문헌 검색을 위한 bge-m3 다국어 임베딩과 "
		"검색 증강 생성(RAG) 아키텍처의 성능을 정량적으로 비교 분석한다.\n"
		"실험 결과 한국어 질의에서 높은 재현율과 정밀도를 달성하였다.",
	)
	write_unicode_pdf(
		papers_dir / "urban_climate_study.pdf",
		"도시 열섬 현상 완화를 위한 녹지 공간 배치 최적화 연구입니다. "
		"미세먼지와 지표면 온도 간의 상관관계를 위성 영상으로 분석하였다.",
	)
	sub_dir = papers_dir / "english"
	write_unicode_pdf(
		sub_dir / "attention_mechanisms.pdf",
		"Transformer self-attention mechanisms enable long-range dependency modeling "
		"across academic document summarization and cross-lingual retrieval benchmarks.",
	)
	return papers_dir
