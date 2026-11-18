"""벡터 인덱싱 진행 상태 이벤트 모델 및 콘솔 진행 표시 유틸리티."""

from collections.abc import Callable
from dataclasses import dataclass
import sys

STATUS_SCANNING = "scanning"
STATUS_LOADING_MODEL = "loading_model"
STATUS_PROCESSING = "processing"
STATUS_ADDED = "added"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_SAVING = "saving"


@dataclass(frozen=True)
class IndexProgress:
	"""개별 PDF 파일 또는 단계별 인덱싱 진행 상황을 표현하는 이벤트."""

	index: int
	total: int
	rel_path: str
	status: str
	elapsed_sec: float = 0.0
	detail: str = ""


ProgressCallback = Callable[[IndexProgress], None]


def emit_progress(
	callback: ProgressCallback | None,
	event: IndexProgress,
) -> None:
	"""콜백이 지정된 경우 진행 이벤트를 안전하게 전달한다."""
	if callback is not None:
		callback(event)


def format_progress_line(event: IndexProgress) -> str:
	"""진행 이벤트를 사용자가 읽기 쉬운 한 줄 상태 문자열로 포매팅한다."""
	prefix = f"[{event.index}/{event.total}]" if event.total > 0 else "[준비]"
	if event.status == STATUS_SCANNING:
		return f"PDF 탐색 완료: 총 {event.total}편 발견"
	if event.status == STATUS_LOADING_MODEL:
		return f"임베딩 모델 준비 중: {event.detail}"
	if event.status == STATUS_PROCESSING:
		return f"{prefix} 처리 중: {event.rel_path} ..."
	if event.status == STATUS_ADDED:
		return f"{prefix} 완료: {event.rel_path} ({event.elapsed_sec:.2f}s)"
	if event.status == STATUS_SKIPPED:
		return f"{prefix} 건너뜀 (기존): {event.rel_path}"
	if event.status == STATUS_FAILED:
		return f"{prefix} 실패: {event.rel_path} — {event.detail}"
	if event.status == STATUS_SAVING:
		return f"인덱스 디스크 저장 중... ({event.detail})"
	return f"{prefix} {event.status}: {event.rel_path}"


def print_console_progress(event: IndexProgress) -> None:
	"""표준 출력으로 즉시 flush하여 멈춤 없이 실시간 진행 상황을 표시한다."""
	print(format_progress_line(event), file=sys.stdout, flush=True)
