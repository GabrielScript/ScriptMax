"""Transcrição via API da Groq (Whisper large-v3, endpoint compatível com OpenAI).

Cada bloco transcrito é gravado em cache (chave = hash do áudio + modelo), então
uma falha no meio do caminho ou um reprocessamento não paga o áudio de novo.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai

from scriptmax import audio

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LANGUAGE = "pt"
# O prompt do Whisper aceita até 224 tokens; ~500 caracteres de PT ficam bem abaixo.
PROMPT_TAIL_CHARS = 500
CONTEXT_HINT_CHARS = 160
# Limiares padrão do próprio Whisper para descartar segmentos sem fala/alucinados.
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
CACHE_FORMAT_VERSION = 1

ProgressCallback = Callable[[int, int], None]


class TranscriptionError(RuntimeError):
    """Falha ao transcrever o áudio."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _Chunk:
    index: int
    samples: np.ndarray
    offset_seconds: float


@dataclass(frozen=True)
class Transcript:
    segments: list[Segment]
    duration_seconds: float

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()

    def to_timestamped_text(self) -> str:
        return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in self.segments)

    def to_json(self) -> str:
        payload = {"duration_seconds": self.duration_seconds, "segments": [asdict(s) for s in self.segments]}
        return json.dumps(payload, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, raw: str) -> Transcript:
        payload = json.loads(raw)
        segments = [Segment(**item) for item in payload["segments"]]
        return cls(segments=segments, duration_seconds=float(payload["duration_seconds"]))


def format_timestamp(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_reliable_segment(segment: Any) -> bool:
    """Replica o critério do Whisper: silêncio provável OU loop de repetição = descarta."""
    no_speech = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    compression = getattr(segment, "compression_ratio", None)
    if no_speech is not None and avg_logprob is not None:
        if no_speech > NO_SPEECH_THRESHOLD and avg_logprob < LOGPROB_THRESHOLD:
            return False
    if compression is not None and compression > COMPRESSION_RATIO_THRESHOLD:
        return False
    return bool(str(getattr(segment, "text", "")).strip())


def build_prompt(context_hint: str, previous_text: str) -> str:
    """Contexto para o Whisper: tema (vocabulário) + fim do bloco anterior (continuidade)."""
    hint = context_hint.strip()[:CONTEXT_HINT_CHARS]
    hint = f"{hint}." if hint else ""
    tail = previous_text[-PROMPT_TAIL_CHARS:].strip()
    return f"{hint} {tail}".strip()


class GroqTranscriber:
    def __init__(self, client: openai.OpenAI, model: str, ffmpeg_path: str, cache_dir: Path) -> None:
        self._client = client
        self._model = model
        self._ffmpeg_path = ffmpeg_path
        self._cache_dir = cache_dir

    @classmethod
    def from_api_key(cls, api_key: str, model: str, ffmpeg_path: str, cache_dir: Path) -> GroqTranscriber:
        # max_retries com backoff respeita o retry-after dos 429 do plano gratuito.
        client = openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=5, timeout=300)
        return cls(client, model, ffmpeg_path, cache_dir)

    def transcribe(self, audio_path: Path, context_hint: str, on_progress: ProgressCallback | None = None) -> Transcript:
        samples = audio.decode_to_pcm(self._ffmpeg_path, audio_path)
        chunk_dir = self._chunk_cache_dir(audio_path)
        chunks = audio.plan_chunks(samples)
        segments: list[Segment] = []
        previous_text = ""

        for index, (start, end) in enumerate(chunks):
            chunk = _Chunk(index=index, samples=samples[start:end], offset_seconds=start / audio.SAMPLE_RATE)
            cache_file = chunk_dir / f"chunk_{index:04d}.json"
            chunk_segments = self._cached_or_transcribe(cache_file, chunk, build_prompt(context_hint, previous_text))
            segments.extend(chunk_segments)
            chunk_text = " ".join(segment.text for segment in chunk_segments)
            previous_text = chunk_text or previous_text
            if on_progress:
                on_progress(index + 1, len(chunks))

        transcript = Transcript(segments=segments, duration_seconds=len(samples) / audio.SAMPLE_RATE)
        if not transcript.text:
            raise TranscriptionError("Nenhuma fala inteligível foi detectada no áudio.")
        return transcript

    def _chunk_cache_dir(self, audio_path: Path) -> Path:
        key_source = f"v{CACHE_FORMAT_VERSION}|{file_sha256(audio_path)}|{self._model}|{LANGUAGE}|{audio.CHUNK_SECONDS}"
        return self._cache_dir / hashlib.sha256(key_source.encode()).hexdigest()

    def _cached_or_transcribe(self, cache_file: Path, chunk: _Chunk, prompt: str) -> list[Segment]:
        if cache_file.exists():
            return [Segment(**item) for item in json.loads(cache_file.read_text(encoding="utf-8"))]

        if audio.is_silent(chunk.samples):
            logger.info("Bloco %d é silêncio; pulando.", chunk.index)
            chunk_segments: list[Segment] = []
        else:
            chunk_segments = self._transcribe_chunk(chunk, prompt)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps([asdict(s) for s in chunk_segments], ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_file)
        return chunk_segments

    def _transcribe_chunk(self, chunk: _Chunk, prompt: str) -> list[Segment]:
        flac_bytes = audio.encode_flac(self._ffmpeg_path, chunk.samples)
        request: dict[str, Any] = {
            "model": self._model,
            "file": ("chunk.flac", flac_bytes, "audio/flac"),
            "language": LANGUAGE,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if prompt:
            request["prompt"] = prompt
        try:
            response = self._client.audio.transcriptions.create(**request)
        except openai.APIError as error:
            raise TranscriptionError(f"Groq recusou a transcrição: {error}") from error
        return self._to_segments(response, chunk.offset_seconds, chunk_seconds=len(chunk.samples) / audio.SAMPLE_RATE)

    @staticmethod
    def _to_segments(response: Any, offset_seconds: float, chunk_seconds: float) -> list[Segment]:
        raw_segments = getattr(response, "segments", None)
        if not raw_segments:
            text = str(getattr(response, "text", "") or "").strip()
            return [Segment(offset_seconds, offset_seconds + chunk_seconds, text)] if text else []

        kept = [segment for segment in raw_segments if is_reliable_segment(segment)]
        dropped = len(raw_segments) - len(kept)
        if dropped:
            logger.info("Descartados %d segmento(s) sem fala/repetitivos.", dropped)
        return [
            Segment(
                start=round(offset_seconds + float(segment.start), 2),
                end=round(offset_seconds + float(segment.end), 2),
                text=str(segment.text).strip(),
            )
            for segment in kept
        ]
