"""Structured, timed logging for pipeline stages (extract/chunk/embed/upsert/
search) so latency can be attributed to a specific stage instead of guessed
at. Each stage emits one JSON-line log record on completion, success or not."""

import json
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("codeseek.pipeline")


@contextmanager
def timed_stage(stage: str, **fields):
    start = time.perf_counter()
    error = None
    try:
        yield
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        record = {"stage": stage, "duration_ms": duration_ms, **fields}
        if error is not None:
            record["error"] = error
            logger.error(json.dumps(record))
        else:
            logger.info(json.dumps(record))


def configure_json_logging(level: int = logging.INFO) -> None:
    """Each stage already emits a JSON string as the log message, so the
    handler format is just the bare message -- no extra text wrapped around it.

    Quiets httpx/urllib3/huggingface_hub, which otherwise log one INFO line
    per HTTP request (model-loading alone fires dozens against the HF Hub)
    and bury the actual per-stage progress this exists to surface -- found
    while debugging what looked like a hung /ingest request but was really
    just this noise burying the pipeline's own log lines."""
    logging.basicConfig(level=level, format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
