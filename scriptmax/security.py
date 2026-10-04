"""Autenticação, limitação de tentativas, CSRF e cabeçalhos de segurança.

Modelo de ameaça: o app pode ser exposto via ngrok/LAN para gravar pelo celular.
- Sem APP_TOKEN: só aceita requisições locais diretas (sem proxy, Host local).
  Isso bloqueia ngrok e DNS rebinding.
- Com APP_TOKEN: login troca o token por um cookie httpOnly + SameSite=Strict.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import deque
from urllib.parse import urlsplit

from starlette.requests import Request

from scriptmax.config import LOCAL_HOSTS

SESSION_COOKIE = "scriptmax_session"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 3600
LOGIN_ATTEMPTS_PER_WINDOW = 10
LOGIN_WINDOW_SECONDS = 15 * 60
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "forwarded", "x-real-ip")
APP_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
BASE_SECURITY_HEADERS = {
    # Navegadores ignoram HSTS recebido por HTTP (RFC 6797 §8.1): inofensivo no uso local.
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "microphone=(self), display-capture=(self), camera=()",
}


def _session_value(app_token: str) -> str:
    return hmac.new(app_token.encode(), b"scriptmax-session-v1", hashlib.sha256).hexdigest()


def _host_without_port(host_header: str) -> str:
    if host_header.startswith("["):
        return host_header[1:].split("]", 1)[0]
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


class SessionAuth:
    def __init__(self, app_token: str | None) -> None:
        self._app_token = app_token
        self._expected_cookie = _session_value(app_token) if app_token else None

    @property
    def token_required(self) -> bool:
        return self._app_token is not None

    def token_matches(self, candidate: str) -> bool:
        if self._app_token is None:
            return False
        return hmac.compare_digest(candidate.encode(), self._app_token.encode())

    def session_cookie_value(self) -> str:
        if self._expected_cookie is None:
            raise RuntimeError("Sessão só existe com APP_TOKEN definido.")
        return self._expected_cookie

    def is_authorized(self, request: Request) -> bool:
        if self._expected_cookie is None:
            return is_direct_local_request(request)
        cookie = request.cookies.get(SESSION_COOKIE, "")
        return hmac.compare_digest(cookie.encode(), self._expected_cookie.encode())


def is_direct_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    if client_host not in LOCAL_HOSTS:
        return False
    if any(header in request.headers for header in PROXY_HEADERS):
        return False
    return _host_without_port(request.headers.get("host", "")) in LOCAL_HOSTS


def origin_is_trusted(request: Request) -> bool:
    """CSRF: em métodos que alteram estado, Origin (se presente) precisa ser o próprio host."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == request.headers.get("host", "")


class LoginRateLimiter:
    def __init__(self, max_attempts: int = LOGIN_ATTEMPTS_PER_WINDOW, window_seconds: int = LOGIN_WINDOW_SECONDS) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            attempts = self._attempts.setdefault(client_key, deque())
            while attempts and now - attempts[0] > self._window_seconds:
                attempts.popleft()
            if len(attempts) >= self._max_attempts:
                return False
            attempts.append(now)
            return True
