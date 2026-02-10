"""Metrics API routes."""

from flask import Blueprint, current_app, jsonify

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Return telemetry metrics for active sessions."""
    config = current_app.config["agent_config"]
    if not config.telemetry.enabled:
        return jsonify({"enabled": False, "sessions": []})

    sessions = current_app.config["sessions"]
    result = []
    for sid, session in sessions.items():
        telemetry = session.context.telemetry
        if telemetry:
            result.append({
                "session_id": sid,
                "metrics": telemetry.summary_dict(),
            })

    return jsonify({
        "enabled": True,
        "log_dir": config.telemetry.log_dir,
        "sessions": result,
    })
