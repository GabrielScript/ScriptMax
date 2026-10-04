from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scriptmax import audio
from scriptmax.transcription import GroqTranscriber, build_prompt, is_reliable_segment
from tests.conftest import FFMPEG, requires_ffmpeg

RATE = audio.SAMPLE_RATE


def tone(seconds: float, amplitude: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def test_short_audio_is_single_chunk() -> None:
    assert audio.plan_chunks(tone(30)) == [(0, 30 * RATE)]


def test_chunks_cut_at_silence_and_respect_max_length() -> None:
    samples = tone(1300)
    silence_at = 605 * RATE
    samples[silence_at : silence_at + RATE] = 0
    chunks = audio.plan_chunks(samples)
    first_end = chunks[0][1]
    assert abs(first_end - (silence_at + RATE // 2)) < RATE // 2
    assert all(end - start <= audio.MAX_CHUNK_SECONDS * RATE for start, end in chunks)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(samples)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_silence_detection() -> None:
    assert audio.is_silent(np.zeros(RATE, dtype=np.int16))
    assert not audio.is_silent(tone(1))


def test_segment_filter_uses_whisper_thresholds() -> None:
    hallucination = SimpleNamespace(text="Legendas pela comunidade", no_speech_prob=0.9, avg_logprob=-1.5, compression_ratio=1.0)
    loop = SimpleNamespace(text="a a a a a", no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=3.0)
    good = SimpleNamespace(text="Produto interno", no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.2)
    assert not is_reliable_segment(hallucination)
    assert not is_reliable_segment(loop)
    assert is_reliable_segment(good)


def test_prompt_is_bounded() -> None:
    prompt = build_prompt("Aula sobre Álgebra", "x" * 5000)
    assert prompt.startswith("Aula sobre Álgebra.")
    assert len(prompt) < 700


class FakeGroqClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        segment = SimpleNamespace(start=1.0, end=2.0, text=" olá turma", no_speech_prob=0.01, avg_logprob=-0.1, compression_ratio=1.1)
        return SimpleNamespace(text="olá turma", segments=[segment])


def write_wav(path: Path, samples: np.ndarray) -> None:
    subprocess.run(
        [FFMPEG, "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "-", str(path)],
        input=samples.tobytes(), check=True,
    )


@requires_ffmpeg
def test_transcriber_uses_cache_and_offsets(tmp_path: Path) -> None:
    wav = tmp_path / "aula.wav"
    write_wav(wav, tone(3))
    client = FakeGroqClient()
    transcriber = GroqTranscriber(client, "whisper-large-v3", FFMPEG, tmp_path / "cache")

    first = transcriber.transcribe(wav, "Aula sobre teste")
    second = transcriber.transcribe(wav, "Aula sobre teste")

    assert len(client.requests) == 1  # segunda vez veio do cache
    assert first == second
    assert first.text == "olá turma"
    assert client.requests[0]["response_format"] == "verbose_json"
    assert client.requests[0]["language"] == "pt"
    assert client.requests[0]["file"][1][:4] == b"fLaC"


@requires_ffmpeg
def test_decode_rejects_non_audio(tmp_path: Path) -> None:
    bogus = tmp_path / "nao_audio.mp3"
    bogus.write_text("isto não é áudio")
    try:
        audio.decode_to_pcm(FFMPEG, bogus)
    except audio.AudioError:
        return
    raise AssertionError("esperava AudioError")
