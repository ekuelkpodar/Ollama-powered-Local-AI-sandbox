"""Authentication helpers for the web UI and API."""

from __future__ import annotations

from flask import Response, request

from agent.config import AuthConfig


def init_auth(app, auth_config: AuthConfig) -> None:
    """Register auth handler for the Flask app."""
    if not auth_config.enabled:
        return

    @app.before_request
    def _check_auth():
        if _is_authorized(request, auth_config):
            return None
        return _unauthorized_response()


def _is_authorized(req, auth_config: AuthConfig) -> bool:
    if auth_config.api_key and _check_api_key(req, auth_config.api_key):
        return True
    if auth_config.password_hash and _check_basic_auth(req, auth_config):
        return True
    return False


def _check_api_key(req, api_key: str) -> bool:
    header_key = req.headers.get("X-API-Key")
    if header_key and header_key == api_key:
        return True
    auth_header = req.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        return token == api_key
    return False


def _check_basic_auth(req, auth_config: AuthConfig) -> bool:
    auth = req.authorization
    if not auth or not auth.username or not auth.password:
        return False
    if auth.username != auth_config.username:
        return False
    return _verify_password(auth.password, auth_config.password_hash)


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        import bcrypt

        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def _unauthorized_response() -> Response:
    return Response(
        "Unauthorized",
        401,
        {"WWW-Authenticate": 'Basic realm="Local Ollama Agents"'},
    )
