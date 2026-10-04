"""Decodificação e fatiamento de áudio via ffmpeg.

O áudio é normalizado para PCM mono 16 kHz (o formato que o Whisper usa
internamente) e cortado em blocos de ~10 min, sempre num trecho de silêncio,
para respeitar o limite de 25 MB por requisição da Groq.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 600
SPLIT_SEARCH_SECONDS = 15
QUIET_WINDOW_SECONDS = 0.5
SILENCE_RMS_DBFS = -55.0
INT16_FULL_SCALE = 32768.0
# Pior caso de um bloco: (600 + 15) s * 32 000 B/s ≈ 19,7 MB em PCM; o FLAC fica abaixo disso.
MAX_CHUNK_SECONDS = CHUNK_SECONDS + SPLIT_SEARCH_SECONDS


class AudioError(RuntimeError):
    """Falha ao decodificar ou codificar áudio."""


def _run_ffmpeg(command: list[str], stdin_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(command, input=stdin_bytes, capture_output=True, check=False)
    except OSError as error:
        raise AudioError(f"Não foi possível executar o ffmpeg: {error}") from error
    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace").strip()[-400:]
        raise AudioError(f"ffmpeg falhou (código {result.returncode}): {stderr_tail}")
    return result.stdout


def decode_to_pcm(ffmpeg_path: str, source: Path) -> np.ndarray:
    """Decodifica qualquer formato suportado pelo ffmpeg para int16 mono 16 kHz."""
    command = [
        ffmpeg_path, "-nostdin", "-v", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    samples = np.frombuffer(_run_ffmpeg(command), dtype=np.int16)
    if samples.size == 0:
        raise AudioError("O arquivo não contém áudio decodificável.")
    return samples


def encode_flac(ffmpeg_path: str, samples: np.ndarray) -> bytes:
    """Codifica PCM int16 mono 16 kHz em FLAC (sem perdas)."""
    command = [
        ffmpeg_path, "-nostdin", "-v", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
        "-c:a", "flac", "-f", "flac", "-",
    ]
    return _run_ffmpeg(command, stdin_bytes=samples.astype(np.int16, copy=False).tobytes())


def find_quiet_point(samples: np.ndarray, target: int, search_radius: int) -> int:
    """Índice de menor energia (janela de 0,5 s) dentro de target ± search_radius."""
    window = int(QUIET_WINDOW_SECONDS * SAMPLE_RATE)
    low = max(0, target - search_radius)
    high = min(len(samples), target + search_radius)
    if high - low <= window:
        return target

    magnitudes = np.abs(samples[low:high].astype(np.int32))
    cumulative = np.concatenate(([0], np.cumsum(magnitudes, dtype=np.int64)))
    window_energy = cumulative[window:] - cumulative[:-window]
    quietest_start = int(np.argmin(window_energy))
    return low + quietest_start + window // 2


def plan_chunks(
    samples: np.ndarray,
    chunk_seconds: int = CHUNK_SECONDS,
    search_seconds: int = SPLIT_SEARCH_SECONDS,
) -> list[tuple[int, int]]:
    """Intervalos [início, fim) em amostras; cada um dura no máximo chunk + search segundos."""
    total = len(samples)
    chunk_length = chunk_seconds * SAMPLE_RATE
    search_radius = search_seconds * SAMPLE_RATE
    boundaries = [0]
    while total - boundaries[-1] > chunk_length:
        target = boundaries[-1] + chunk_length
        boundaries.append(find_quiet_point(samples, target, search_radius))
    boundaries.append(total)
    return list(zip(boundaries, boundaries[1:]))


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms == 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms / INT16_FULL_SCALE))


def is_silent(samples: np.ndarray) -> bool:
    """Bloco praticamente mudo: não vale pagar transcrição (e o Whisper alucina em silêncio)."""
    return rms_dbfs(samples) < SILENCE_RMS_DBFS
