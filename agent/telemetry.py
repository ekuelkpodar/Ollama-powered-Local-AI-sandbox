"""Telemetry and metrics logging for agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Any

from agent.config import TelemetryConfig


@dataclass
class LLMCallMetric:
    """Metrics for a single LLM call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: str | None = None


@dataclass
class ToolCallMetric:
    """Metrics for a single tool call."""
    tool_name: str
    args: dict
    duration_ms: float
    cached: bool
    result_summary: str
    error: str | None = None


@dataclass
class LoopIterationMetric:
    """Metrics for a single monologue loop iteration."""
    iteration: int
    decision: str
    duration_ms: float


@dataclass
class MemoryOpMetric:
    """Metrics for a memory operation."""
    operation: str
    query: str | None
    result_count: int
    top_score: float | None
    namespace: str | None
    duration_ms: float


@dataclass
class LoopMetrics:
    """Session-level metrics summary."""
    session_id: str
    total_iterations: int
    tool_calls: list[ToolCallMetric]
    llm_calls: list[LLMCallMetric]
    total_duration_ms: float
    memory_operations: int
    final_tool: str


class Telemetry:
    """Capture structured telemetry for a session."""

    def __init__(self, config: TelemetryConfig, session_id: str):
        self.config = config
        self.session_id = session_id
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._llm_calls: list[LLMCallMetric] = []
        self._tool_calls: list[ToolCallMetric] = []
        self._loop_iterations: list[LoopIterationMetric] = []
        self._memory_ops: list[MemoryOpMetric] = []
        self._total_iterations = 0
        self._final_tool = ""
        self._log_path: str | None = None
        self._tracer = None

        if self.config.enabled:
            os.makedirs(self.config.log_dir, exist_ok=True)
            self._log_path = os.path.join(self.config.log_dir, f"{session_id}.jsonl")
            if self.config.otel_enabled:
                self._setup_otel()

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record an LLM call metric."""
        if not self.config.enabled:
            return
        metric = LLMCallMetric(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            error=error,
        )
        self._llm_calls.append(metric)
        self._log_event("llm_call", asdict(metric))
        self._emit_span("llm_call", asdict(metric))

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        duration_ms: float,
        cached: bool,
        result_summary: str,
        error: str | None = None,
    ) -> None:
        """Record a tool call metric."""
        if not self.config.enabled:
            return
        metric = ToolCallMetric(
            tool_name=tool_name,
            args=args,
            duration_ms=duration_ms,
            cached=cached,
            result_summary=result_summary,
            error=error,
        )
        self._tool_calls.append(metric)
        self._log_event("tool_call", asdict(metric))
        self._emit_span("tool_call", asdict(metric))

    def record_iteration(self, iteration: int, decision: str, duration_ms: float) -> None:
        """Record a monologue loop iteration metric."""
        if not self.config.enabled:
            return
        metric = LoopIterationMetric(
            iteration=iteration,
            decision=decision,
            duration_ms=duration_ms,
        )
        self._total_iterations += 1
        self._loop_iterations.append(metric)
        self._log_event("loop_iteration", asdict(metric))

    def record_memory_operation(
        self,
        operation: str,
        query: str | None,
        result_count: int,
        top_score: float | None,
        namespace: str | None,
        duration_ms: float,
    ) -> None:
        """Record a memory operation metric."""
        if not self.config.enabled:
            return
        metric = MemoryOpMetric(
            operation=operation,
            query=query,
            result_count=result_count,
            top_score=top_score,
            namespace=namespace,
            duration_ms=duration_ms,
        )
        self._memory_ops.append(metric)
        self._log_event("memory_operation", asdict(metric))
        self._emit_span("memory_operation", asdict(metric))

    def finalize(self, final_tool: str) -> None:
        """Finalize session metrics with the last tool used."""
        if not self.config.enabled:
            return
        self._final_tool = final_tool
        self._log_event("session_summary", self.summary_dict())

    def summary(self) -> LoopMetrics:
        """Return a session-level metrics summary."""
        total_duration_ms = (time.monotonic() - self._start_time) * 1000
        return LoopMetrics(
            session_id=self.session_id,
            total_iterations=self._total_iterations,
            tool_calls=list(self._tool_calls),
            llm_calls=list(self._llm_calls),
            total_duration_ms=total_duration_ms,
            memory_operations=len(self._memory_ops),
            final_tool=self._final_tool,
        )

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        summary = self.summary()
        return {
            "session_id": summary.session_id,
            "total_iterations": summary.total_iterations,
            "tool_calls": [asdict(m) for m in summary.tool_calls],
            "llm_calls": [asdict(m) for m in summary.llm_calls],
            "total_duration_ms": summary.total_duration_ms,
            "memory_operations": summary.memory_operations,
            "final_tool": summary.final_tool,
        }

    def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.config.enabled or not self._log_path:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event_type,
            **payload,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _emit_span(self, name: str, attributes: dict[str, Any]) -> None:
        if not self._tracer:
            return
        try:
            with self._tracer.start_as_current_span(name) as span:
                for key, value in attributes.items():
                    if value is None:
                        continue
                    span.set_attribute(key, value)
        except Exception:
            return

    def _setup_otel(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            except Exception:
                OTLPSpanExporter = None

            resource = Resource.create({"service.name": self.config.otel_service_name})
            provider = TracerProvider(resource=resource)
            if OTLPSpanExporter and self.config.otel_endpoint:
                exporter = OTLPSpanExporter(endpoint=self.config.otel_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
        except Exception:
            self._tracer = None
            self._log_event(
                "telemetry_warning",
                {"message": "OpenTelemetry not available or failed to initialize."},
            )
