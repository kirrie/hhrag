"""hhrag 전역 상수, 설정 데이터클래스 및 PaperQA2 설정 팩토리."""

from dataclasses import dataclass
from pathlib import Path

from paperqa import Settings

# 기본 임베딩 모델: 다국어 및 한국어 학술 검색에 특화된 bge-m3 (PaperQA2 SentenceTransformer 접두사)
DEFAULT_EMBEDDING_MODEL = "st-BAAI/bge-m3"
DEFAULT_MCP_NAME = "hhrag"
INDEX_DIR_NAME = ".hhrag"
DOCS_FILE_NAME = "docs.pkl"
MANIFEST_FILE_NAME = "manifest.json"
DEFAULT_TOP_K = 5
MAX_TOP_K = 30
DEFAULT_CHUNK_CHARS = 1500
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 50


@dataclass(frozen=True)
class IndexConfig:
	"""벡터 인덱싱 및 MCP 서버 실행에 필요한 설정 모음."""

	pdf_dir: Path
	embedding_model: str = DEFAULT_EMBEDDING_MODEL
	mcp_name: str = DEFAULT_MCP_NAME
	chunk_chars: int = DEFAULT_CHUNK_CHARS
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

	def __post_init__(self) -> None:
		"""경로 및 청크 파라미터의 기본 불변 조건을 검증한다."""
		if not self.embedding_model.strip():
			raise ValueError("임베딩 모델 이름은 비어 있을 수 없습니다.")
		if self.chunk_chars <= 0:
			raise ValueError("청크 길이(chunk_chars)는 1 이상이어야 합니다.")
		if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_chars:
			raise ValueError("청크 겹침(chunk_overlap)은 0 이상 chunk_chars 미만이어야 합니다.")

	@property
	def index_dir(self) -> Path:
		"""인덱스 산출물이 저장되는 디렉토리 경로를 반환한다."""
		return self.pdf_dir / INDEX_DIR_NAME

	@property
	def docs_path(self) -> Path:
		"""직렬화된 PaperQA2 Docs 파일 경로를 반환한다."""
		return self.index_dir / DOCS_FILE_NAME

	@property
	def manifest_path(self) -> Path:
		"""파일 해시 매니페스트 JSON 경로를 반환한다."""
		return self.index_dir / MANIFEST_FILE_NAME


def create_paperqa_settings(config: IndexConfig) -> Settings:
	"""로컬 임베딩 전용 PaperQA2 Settings 인스턴스를 생성한다."""
	# 외부 LLM 및 메타데이터 API 호출을 막기 위해 use_doc_details와 multimodal을 비활성화한다.
	parsing_config = {
		"use_doc_details": False,
		"multimodal": False,
		"reader_config": {
			"chunk_chars": config.chunk_chars,
			"overlap": config.chunk_overlap,
		},
	}
	return Settings(
		embedding=config.embedding_model,
		parsing=parsing_config,
	)
