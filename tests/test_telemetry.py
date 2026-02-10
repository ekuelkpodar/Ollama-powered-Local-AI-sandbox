import json
from pathlib import Path

from agent.telemetry import Telemetry
from agent.config import TelemetryConfig


def test_telemetry_records_events(tmp_path: Path):
    config = TelemetryConfig(
        enabled=True,
        log_dir=str(tmp_path),
        otel_enabled=False,
        otel_endpoint=None,
        otel_service_name="local-ollama-agents",
    )
    telemetry = Telemetry(config, session_id="sess123")

    telemetry.record_llm_call(
        model="llama3.2",
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=123.4,
    )
    telemetry.record_tool_call(
        tool_name="memory",
        args={"action": "search"},
        duration_ms=55.5,
        cached=False,
        result_summary="Memories found",
    )
    telemetry.record_iteration(iteration=1, decision="tool:memory", duration_ms=200.0)
    telemetry.record_memory_operation(
        operation="search",
        query="test",
        result_count=2,
        top_score=0.9,
        namespace="default",
        duration_ms=10.0,
    )
    telemetry.finalize("response")

    summary = telemetry.summary()
    assert summary.total_iterations == 1
    assert summary.memory_operations == 1
    assert summary.final_tool == "response"

    log_path = tmp_path / "sess123.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 5
    events = [json.loads(line)["event"] for line in lines]
    assert "llm_call" in events
    assert "tool_call" in events
    assert "loop_iteration" in events
    assert "memory_operation" in events
    assert "session_summary" in events


def test_telemetry_disabled_no_log(tmp_path: Path):
    config = TelemetryConfig(enabled=False, log_dir=str(tmp_path))
    telemetry = Telemetry(config, session_id="sess456")
    telemetry.record_llm_call(
        model="llama3.2",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
    )
    log_path = tmp_path / "sess456.jsonl"
    assert not log_path.exists()
