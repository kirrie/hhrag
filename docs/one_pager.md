# 문제 1페이지 정의서 (One-Pager) — hhrag

## 1. 맥락 (Context)
- 아내의 석사논문 준비를 위해 수집한 다수의 학술 논문 PDF(한국어·영어 혼용)가 로컬 디렉토리에 저장되어 있음.
- 사용 중인 Codex 에이전트가 이 논문들을 근거(RAG)로 검색·인용할 수 있는 로컬 MCP 서버 환경이 필요함.

## 2. 문제 (Problem)
- PaperQA2 기본 설정은 OpenAI 등 외부 유료 임베딩 및 요약 LLM에 의존하여, 한국어 학술 검색 특화 및 외부 API 키 없는 로컬 단독 실행이 번거로움.
- 새로운 논문이 추가될 때마다 수동으로 인덱싱하고 Codex MCP 설정을 맞추는 과정이 반복적이고 불편함.

## 3. 목표 (Goals)
1. **PaperQA2 + `bge-m3` 임베딩 전환**: `paper-qa`를 코어로 사용하되 임베딩을 다국어·한국어 성능이 우수한 로컬 `BAAI/bge-m3` (`st-BAAI/bge-m3`)로 구성한다.
2. **디렉토리 기반 증분 벡터 인덱싱**: 디렉토리 경로를 인자로 전달하면 하위 `.pdf` 파일을 스캔하고 SHA-256 해시 비교를 통해 신규·변경 파일만 증분 인덱싱한다.
3. **원커맨드 준비 (`setup` / 기본 실행)**: 단일 명령으로 PDF 벡터 인덱싱과 `codex mcp add` 등록을 한 번에 완료한다.
4. **MCP 검색 도구 제공**: 에이전트가 `search_papers`, `list_papers`, `get_paper_chunks` 도구를 통해 논문 원문 청크를 검색·열람할 수 있게 한다.

## 4. 비목표 (Non-Goals)
1. 별도의 웹 UI 또는 GUI 프론트엔드 구축.
2. 외부 분산 벡터 데이터베이스 서버(Qdrant, Milvus 등) 운영.
3. 텍스트 레이어가 없는 스캔 이미지 PDF의 고급 OCR 처리.

## 5. 제약조건 (Constraints)
1. macOS, Python 3.11+, `uv`, `codex-cli` 환경에서 외부 유료 LLM API 키 없이 100% 로컬 동작해야 한다.
2. 코드 품질 한계: 탭(`\t`, size 4) 들여쓰기, 줄당 ≤ 100자, 파일 ≤ 300 LOC, 함수 ≤ 50 LOC, 파라미터 ≤ 5개, 순환 복잡도 ≤ 10.
3. 모든 대화·문서·주석·에러 메시지는 한국어로 작성한다.

## 6. 가정 및 기술 결정 기록 (Assumptions & ADR)

### 가정 (Assumptions)
1. 수집된 논문 PDF 파일에는 추출 가능한 텍스트 레이어가 포함되어 있다.
2. 최초 1회 실행 시 HuggingFace에서 `BAAI/bge-m3` 가중치(약 2.2GB)를 내려받을 수 있는 인터넷 연결이 존재한다(이후 로컬 캐시 사용).
3. 최종 문헌 종합과 추론은 MCP 클라이언트인 Codex 에이전트가 수행하므로, PaperQA2 내부에서는 LLM 요약 호출 없이 벡터 청크 검색에 집중한다.

### ADR-01: 로컬 임베딩 연동 방식
- **옵션 A (채택)**: PaperQA2 내장 `SentenceTransformerEmbeddingModel` (`st-BAAI/bge-m3`). 별도 데몬 없이 단일 프로세스로 동작해 가장 단순함.
- **옵션 B (기각)**: Ollama 서버 + LiteLLM 연동. 로컬에 Ollama가 미설치되어 있고 데몬 관리 부담이 큼.

### ADR-02: PaperQA2 외부 LLM 의존성 분리
- **옵션 A (채택)**: `use_doc_details=False` 및 명시적 `citation`/`docname`/`dockey` 전달로 `aadd` 시 LLM 호출을 차단하고, 검색은 `Docs.aget_texts()`로 벡터 청크를 직접 반환.
- **옵션 B (기각)**: `Docs.aquery()` 전체 파이프라인 사용. 외부 유료 API 키가 필수이며 호출 지연·비용이 발생함.

### ADR-03: 로컬 인덱스 영속화 방식
- **옵션 A (채택)**: PaperQA2 `Docs` pickle 직렬화 + SHA-256 매니페스트 JSON을 임시 파일 작성 후 `os.replace`로 원자적 교체.
- **옵션 B (기각)**: 별도 외부 벡터 DB 연동. PaperQA2 상태와 이중 동기화가 필요해 복잡도가 증가함.
