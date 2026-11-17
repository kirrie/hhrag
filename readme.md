# hhrag

석사논문 준비를 위해 수집한 PDF 논문들을 로컬에서 벡터 인덱싱하고, Codex 등 에이전트가
MCP(Model Context Protocol)를 통해 한국어·영어로 검색·열람할 수 있게 해주는 도구입니다.

- **코어 엔진**: PaperQA2 (`paper-qa`)
- **임베딩 모델**: 로컬 다국어 `BAAI/bge-m3` (`st-BAAI/bge-m3`) — 외부 유료 API 키 불필요
- **저장소**: `<pdf-dir>/.hhrag/` 내부에 PaperQA2 `docs.pkl` 및 SHA-256 `manifest.json` 저장

## 설치

### 방법 1: `uv tool install` 전역 설치 (권장)

```bash
uv tool install git+https://github.com/kirrie/hhrag.git
```

설치 후 어느 디렉토리에서나 `hhrag` 명령을 바로 사용할 수 있습니다.
(`requires-python`이 `<3.15`로 고정되어 있어 Rust 빌드 없이 prebuilt wheel로 설치됩니다.)

### 방법 2: 저장소 클론 후 로컬 실행

```bash
git clone https://github.com/kirrie/hhrag.git
cd hhrag
uv sync
```

> 최초 1회 실행 시 `BAAI/bge-m3` 모델 가중치(약 2.2GB)를 로컬 HuggingFace 캐시로 내려받습니다.

## 원커맨드 사용 준비 (`setup`)

PDF 폴더 경로를 지정해 실행하면 **증분 벡터 인덱싱**과 **`codex mcp add` 등록**을 한 번에 마칩니다.

```bash
# 전역 설치(방법 1)인 경우
hhrag --pdf-dir "/path/to/thesis_papers"

# 로컬 저장소(방법 2)인 경우
uv run hhrag --pdf-dir "/path/to/thesis_papers"
```

위 명령은 아래와 동일합니다.

```bash
uv run hhrag setup --pdf-dir "/path/to/thesis_papers"
```

등록 결과는 `codex mcp list`로 확인할 수 있습니다.

## 주요 하위 명령

- `setup`: PDF 벡터 인덱싱 후 Codex MCP 서버 등록 (`--skip-mcp`로 등록 생략 가능)
- `index`: 새로 추가·수정된 PDF만 증분 인덱싱 (기존 파일은 SHA-256 비교로 건너뜀)
- `install-mcp`: Codex에 `hhrag` MCP 서버만 등록/갱신
- `serve`: Codex가 백그라운드 stdio로 실행하는 MCP 서버 엔트리포인트

```bash
# 새 논문 PDF 추가 후 재인덱싱만 실행
uv run hhrag index --pdf-dir "/path/to/thesis_papers"

# MCP 서버 이름 또는 청크 크기 변경
uv run hhrag setup --pdf-dir "/path/to/thesis_papers" --mcp-name thesis-rag --chunk-chars 1500
```

## 제공되는 MCP 도구

1. `search_papers(query, top_k=5)`: 한국어/영어 질의로 가장 관련성 높은 논문 구절 벡터 검색
2. `list_papers()`: 현재 인덱싱된 논문 목록 및 각 논문의 청크 수 조회
3. `get_paper_chunks(doc_name, offset=0, limit=10)`: 특정 논문의 원문 청크를 순서대로 열람

## 테스트 실행

```bash
uv run pytest -v
```
