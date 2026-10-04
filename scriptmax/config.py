"""Configuração carregada do ambiente (.env na raiz do projeto)."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MIN_APP_TOKEN_LENGTH = 16


class ConfigError(RuntimeError):
    """Configuração obrigatória ausente ou insegura."""


@dataclass(frozen=True)
class EmailSettings:
    server: str
    port: int
    user: str
    password: str
    recipient: str


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    deepseek_api_key: str
    transcription_model: str
    summary_model: str
    host: str
    port: int
    app_token: str | None
    data_dir: Path
    library_dir: Path
    max_upload_bytes: int
    ffmpeg_path: str
    email: EmailSettings | None

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def transcript_cache_dir(self) -> Path:
        return self.data_dir / "transcripts"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, "").strip()
    return value or default


def _required(name: str) -> str:
    value = _env(name)
    if value is None:
        raise ConfigError(f"Variável {name} ausente no .env.")
    return value


def _load_email_settings() -> EmailSettings | None:
    user = _env("EMAIL_USER")
    password = _env("EMAIL_PASSWORD")
    recipient = _env("EMAIL_RECIPIENT") or _env("DEFAULT_RECIPIENT")
    if not (user and password and recipient):
        return None
    return EmailSettings(
        server=_env("SMTP_SERVER", "smtp.gmail.com") or "smtp.gmail.com",
        port=int(_env("SMTP_PORT", "587") or "587"),
        user=user,
        password=password,
        recipient=recipient,
    )


def _resolve_ffmpeg() -> str:
    ffmpeg_path = _env("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise ConfigError("ffmpeg não encontrado. Instale (winget install Gyan.FFmpeg) ou defina FFMPEG_PATH.")
    return ffmpeg_path


def _validate_app_token(app_token: str | None) -> None:
    if app_token is not None and len(app_token) < MIN_APP_TOKEN_LENGTH:
        raise ConfigError(f"APP_TOKEN precisa ter ao menos {MIN_APP_TOKEN_LENGTH} caracteres.")


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    host = _env("HOST", "127.0.0.1") or "127.0.0.1"
    app_token = _env("APP_TOKEN")
    _validate_app_token(app_token)
    if host not in LOCAL_HOSTS and app_token is None:
        raise ConfigError("HOST expõe o app na rede: defina APP_TOKEN no .env.")
    data_dir = Path(_env("DATA_DIR") or PROJECT_ROOT / "data")

    return Settings(
        groq_api_key=_required("GROQ_API_KEY"),
        deepseek_api_key=_required("DEEPSEEK_API_KEY"),
        transcription_model=_env("TRANSCRIPTION_MODEL", "whisper-large-v3") or "whisper-large-v3",
        summary_model=_env("SUMMARY_MODEL", "deepseek-flash") or "deepseek-flash",
        host=host,
        port=int(_env("PORT", "8000") or "8000"),
        app_token=app_token,
        data_dir=data_dir,
        library_dir=Path(_env("LIBRARY_DIR") or data_dir / "biblioteca"),
        max_upload_bytes=int(_env("MAX_UPLOAD_MB", "500") or "500") * 1024 * 1024,
        ffmpeg_path=_resolve_ffmpeg(),
        email=_load_email_settings(),
    )
