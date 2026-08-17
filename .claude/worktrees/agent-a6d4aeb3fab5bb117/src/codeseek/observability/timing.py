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
    handler format is just the bare message -- no extra text wrapped around it."""
    logging.basicConfig(level=level, format="%(message)s")
