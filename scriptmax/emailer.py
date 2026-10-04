"""Envio do relatório por e-mail. O destinatário é fixo no .env (nunca vem do navegador)."""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from scriptmax.config import EmailSettings

SMTP_TIMEOUT_SECONDS = 30
IMPLICIT_TLS_PORT = 465
MAX_SUBJECT_CHARS = 150
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
ATTACHMENT_TYPES = {".pdf": ("application", "pdf"), ".html": ("text", "html")}


class EmailError(RuntimeError):
    """Falha ao enviar o e-mail."""


def safe_header_text(text: str) -> str:
    """Remove CR/LF e outros controles (bloqueia injeção de cabeçalho)."""
    return CONTROL_CHARACTERS.sub(" ", text).strip()[:MAX_SUBJECT_CHARS]


def build_message(settings: EmailSettings, subject: str, attachments: list[Path]) -> EmailMessage:
    clean_subject = safe_header_text(subject)
    message = EmailMessage()
    message["From"] = settings.user
    message["To"] = settings.recipient
    message["Subject"] = f"ScriptMax: {clean_subject}"
    message.set_content(f"Seu relatório \"{clean_subject}\" está pronto. Os arquivos seguem em anexo.\n\n— ScriptMax")
    for path in attachments:
        maintype, subtype = ATTACHMENT_TYPES.get(path.suffix.lower(), ("application", "octet-stream"))
        message.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)
    return message


class EmailSender:
    def __init__(self, settings: EmailSettings) -> None:
        self._settings = settings

    def send(self, subject: str, attachments: list[Path]) -> None:
        message = build_message(self._settings, subject, attachments)
        context = ssl.create_default_context()
        try:
            if self._settings.port == IMPLICIT_TLS_PORT:
                with smtplib.SMTP_SSL(self._settings.server, self._settings.port, timeout=SMTP_TIMEOUT_SECONDS, context=context) as smtp:
                    self._login_and_send(smtp, message)
            else:
                with smtplib.SMTP(self._settings.server, self._settings.port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
                    smtp.starttls(context=context)
                    self._login_and_send(smtp, message)
        except (smtplib.SMTPException, OSError) as error:
            raise EmailError(f"Falha ao enviar e-mail: {error}") from error

    def _login_and_send(self, smtp: smtplib.SMTP, message: EmailMessage) -> None:
        smtp.login(self._settings.user, self._settings.password)
        smtp.send_message(message)
