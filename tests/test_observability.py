import json
import logging

import pytest

from codeseek.observability.timing import timed_stage


def test_timed_stage_logs_structured_json_on_success(caplog):
    with caplog.at_level(logging.INFO, logger="codeseek.pipeline"):
        with timed_stage("embed", repo="demo", batch_size=3):
            pass

    assert len(caplog.records) == 1
    record = json.loads(caplog.records[0].message)
    assert record["stage"] == "embed"
    assert record["repo"] == "demo"
    assert record["batch_size"] == 3
    assert record["duration_ms"] >= 0
    assert "error" not in record


def test_timed_stage_logs_error_and_reraises(caplog):
    with caplog.at_level(logging.INFO, logger="codeseek.pipeline"):
        with pytest.raises(ValueError):
            with timed_stage("embed", repo="demo"):
                raise ValueError("boom")

    record = json.loads(caplog.records[0].message)
    assert record["stage"] == "embed"
    assert "boom" in record["error"]
