from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from scriptmax.categories import Category
from scriptmax.summarization import SummaryError, SummaryRequest, Summarizer, build_system_prompt, split_text


def completion(content: str, finish_reason: str = "stop"):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, prompt_cache_hit_tokens=80)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)], usage=usage)


class ScriptedClient:
    """Responde em ordem; uma Exception na fila é lançada."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def api_error() -> openai.APIError:
    return openai.APIError("falhou", request=SimpleNamespace(), body=None)  # type: ignore[arg-type]


def test_split_text_without_punctuation_hard_wraps() -> None:
    chunks = split_text("palavra " * 5000, max_chars=1000)
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert " ".join(chunks).split() == ("palavra " * 5000).split()


def test_split_text_keeps_sentences_together() -> None:
    text = "Primeira frase. Segunda frase! Terceira?"
    assert split_text(text, max_chars=20) == ["Primeira frase.", "Segunda frase!", "Terceira?"]


def test_non_academic_skips_classification_and_disables_thinking() -> None:
    client = ScriptedClient([completion("# Ata")])
    result = Summarizer(client, "deepseek-flash").summarize("Reunião curta.", SummaryRequest("Sprint", Category.WORK))
    assert result.markdown == "# Ata"
    assert len(client.calls) == 1
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "CATEGORIA: TRABALHO" in client.calls[0]["messages"][0]["content"]


def test_academic_classifies_then_uses_math_rules() -> None:
    client = ScriptedClient([completion("A"), completion("# Integrais")])
    result = Summarizer(client, "deepseek-flash").summarize("Integral de x.", SummaryRequest("Cálculo", Category.ACADEMIC))
    assert result.approach == "A"
    assert "exatas detectado" in client.calls[1]["messages"][0]["content"]


def test_truncated_part_is_continued() -> None:
    client = ScriptedClient([completion("Começo ", "length"), completion("fim.")])
    result = Summarizer(client, "m").summarize("Texto.", SummaryRequest("t", Category.MEDIA))
    assert result.markdown == "Começo fim."
    assert result.is_complete
    assert client.calls[1]["messages"][-2] == {"role": "assistant", "content": "Começo "}


def test_failed_part_is_marked_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scriptmax.summarization.MAX_PARALLEL_PARTS", 1)  # ordem determinística
    client = ScriptedClient([completion("# Parte 1"), api_error()])
    text = "Frase de teste. " * 1000  # 16 000 caracteres -> 2 trechos
    result = Summarizer(client, "m").summarize(text, SummaryRequest("t", Category.PERSONAL_GROWTH))
    assert result.failed_parts == [2]
    assert not result.is_complete
    assert result.markdown.startswith("# Parte 1")
    assert "não pôde ser gerado" in result.markdown


def test_all_parts_failing_raises() -> None:
    client = ScriptedClient([api_error()])
    with pytest.raises(SummaryError):
        Summarizer(client, "m").summarize("Texto.", SummaryRequest("t", Category.WORK))


def test_system_prompt_is_identical_across_parts() -> None:
    assert build_system_prompt(Category.MEDIA, "") == build_system_prompt(Category.MEDIA, "")
    assert "spoilers" in build_system_prompt(Category.MEDIA, "")
