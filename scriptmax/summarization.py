"""Geração do relatório com DeepSeek (API compatível com OpenAI).

Custo: o prompt fixo vai inteiro na mensagem de sistema, idêntica em todas as
chamadas de uma aula -> a DeepSeek cobra esse prefixo como cache hit (~1/50 do preço).
O modo "thinking" é desligado: tokens de raciocínio seriam cobrados como saída
e consumiriam o max_tokens, truncando seções.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import openai

from scriptmax.categories import ACADEMIC_MATH_RULES, ACADEMIC_THEORY_RULES, Category, profile_for

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHUNK_CHARS = 12_000
MAX_OUTPUT_TOKENS = 8_000
CLASSIFY_SAMPLE_CHARS = 6_000
MAX_PARALLEL_PARTS = 3
DISABLE_THINKING = {"thinking": {"type": "disabled"}}
MATH_APPROACH = "A"
THEORY_APPROACH = "B"

BASE_RULES = """Você transforma transcrições automáticas (Whisper) de áudios em relatórios em português do Brasil.

## PRINCÍPIO CENTRAL: DENSO, NÃO LONGO
- Cubra TODOS os pontos do trecho, mas cada informação aparece UMA única vez.
- Sem enchimento: nada de introduções genéricas, transições vazias ou recapitulações.
- Use o número de palavras necessário para registrar o ponto com precisão — nem mais, nem menos.

## FIDELIDADE (OBRIGATÓRIO)
- Não invente fatos, exemplos, números ou fórmulas que não estejam na transcrição.
- A transcrição tem erros de reconhecimento. Corrija um termo só quando o contexto deixar a correção inequívoca; se houver dúvida, mantenha o termo e marque com [?].
- A transcrição NÃO identifica quem fala. Só atribua falas, decisões ou tarefas a uma pessoa se o nome for dito explicitamente.

## FORMATAÇÃO
- Markdown: # e ## para tópicos, **negrito** para conceitos-chave, > para definições, citações e artigos de lei.
- Toda expressão matemática usa LaTeX: inline $...$ e bloco $$...$$. Nunca use \\( \\) nem \\[ \\].
- O caractere $ é reservado para LaTeX. Valores monetários: escreva "R$ 100" (com espaço) ou "100 reais".
- Não escreva HTML."""

ACADEMIC_APPROACH_RULES = {MATH_APPROACH: ACADEMIC_MATH_RULES, THEORY_APPROACH: ACADEMIC_THEORY_RULES}

CLASSIFY_INSTRUCTION = (
    "Classifique o conteúdo da transcrição abaixo. Responda SOMENTE com uma letra:\n"
    "A = exatas, engenharia, computação ou qualquer conteúdo com fórmulas/cálculos;\n"
    "B = humanidades, direito, negócios ou conteúdo teórico sem cálculos.\n\n"
)

CONTINUE_INSTRUCTION = (
    "Sua resposta foi cortada pelo limite de tamanho. Continue exatamente de onde parou, "
    "sem repetir nada do que já escreveu e sem comentários sobre a continuação."
)

CARD_MAX_TOKENS = 2_000
CARD_INSTRUCTION = (
    "Escreva a ficha de memória do relatório acima, para servir de contexto aos próximos itens da mesma "
    "sequência ({label}). 300 a 600 palavras, em português do Brasil, em Markdown simples, sem introdução. "
    "Registre o que um item futuro precisaria saber: conceitos e definições, personagens, decisões, nomes e "
    "termos ditos, e pontas abertas ou pendências. Use apenas o que está no relatório; não invente."
)
CONNECTIONS_INSTRUCTION = (
    "Depois das seções finais da estrutura, acrescente a seção '# Conexões com os anteriores': o que evoluiu "
    "ou foi retomado em relação às fichas da memória e as pontas soltas, abertas ou resolvidas. "
    "Cite os itens pelo rótulo e marque inferências como 'Análise:'."
)

ProgressCallback = Callable[[int, int], None]


class SummaryError(RuntimeError):
    """Nenhuma parte do relatório pôde ser gerada."""


@dataclass(frozen=True)
class SummaryRequest:
    subject: str
    category: Category
    memory: str = ""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.cached_prompt_tokens += int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def merge(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class PartResult:
    markdown: str
    usage: TokenUsage
    truncated: bool = False
    error: str | None = None


@dataclass
class SummaryResult:
    markdown: str
    approach: str
    failed_parts: list[int] = field(default_factory=list)
    truncated_parts: list[int] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def is_complete(self) -> bool:
        return not self.failed_parts and not self.truncated_parts


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Blocos <= max_chars cortados em fim de frase (ou em espaço, se não houver pontuação)."""
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        pieces.extend(_hard_wrap(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = sentence
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def build_system_prompt(category: Category, approach: str, memory: str = "") -> str:
    """Prefixo idêntico em todas as partes de um mesmo áudio (maximiza o cache da DeepSeek).

    A memória fica por último: é a única parte que cresce de um item para o próximo.
    """
    sections = [BASE_RULES, profile_for(category).rules]
    if category is Category.ACADEMIC:
        sections.append(ACADEMIC_APPROACH_RULES.get(approach, ACADEMIC_THEORY_RULES))
    if memory:
        sections.append(memory)
    return "\n\n".join(sections)


def build_part_instruction(subject: str, part: int, total: int, has_memory: bool = False) -> str:
    header = f"Título informado pelo usuário: {subject}\n"
    connections = f" {CONNECTIONS_INSTRUCTION}" if has_memory and part == total else ""
    if total == 1:
        return header + "Gere o relatório completo da transcrição abaixo." + connections
    intro = "Comece pelas seções iniciais da estrutura." if part == 1 else "Não repita seções introdutórias: continue o conteúdo."
    outro = "Feche com as seções finais da estrutura." if part == total else "Não escreva seções finais: haverá mais partes."
    return (
        header
        + f"Este é o trecho {part} de {total} de uma transcrição longa, em ordem. "
        + f"Gere o relatório APENAS deste trecho. {intro} {outro} "
        + "Não escreva 'Parte N' nos títulos."
        + connections
    )


class Summarizer:
    def __init__(self, client: openai.OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_api_key(cls, api_key: str, model: str) -> Summarizer:
        return cls(openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, max_retries=4, timeout=600), model)

    def summarize(
        self, transcript_text: str, request: SummaryRequest, on_progress: ProgressCallback | None = None
    ) -> SummaryResult:
        chunks = split_text(transcript_text)
        if not chunks:
            raise SummaryError("Transcrição vazia.")

        # Só a categoria acadêmica precisa decidir entre exatas e teórico.
        approach = self.classify(transcript_text) if request.category is Category.ACADEMIC else ""
        system_prompt = build_system_prompt(request.category, approach, request.memory)
        subject = request.subject
        has_memory = bool(request.memory)
        total = len(chunks)
        results: list[PartResult | None] = [None] * total
        completed = 0
        progress_lock = threading.Lock()

        def generate(index: int) -> None:
            nonlocal completed
            instruction = build_part_instruction(subject, index + 1, total, has_memory)
            results[index] = self._generate_part(system_prompt, instruction, chunks[index])
            with progress_lock:
                completed += 1
                done = completed
            if on_progress:
                on_progress(done, total)

        with ThreadPoolExecutor(max_workers=min(total, MAX_PARALLEL_PARTS)) as executor:
            list(executor.map(generate, range(total)))

        return self._assemble([result for result in results if result is not None], approach)

    def write_memory_card(self, request: SummaryRequest, approach: str, report_markdown: str) -> str:
        """Ficha curta do relatório. Reusa o system prompt dos trechos (mesmo prefixo no cache);
        só o que muda vai no user, com a instrução por último."""
        label = profile_for(request.category).label
        user = (
            f"Título: {request.subject}\n\nRELATÓRIO:\n\"\"\"\n{report_markdown}\n\"\"\"\n\n"
            + CARD_INSTRUCTION.format(label=label)
        )
        messages = [
            {"role": "system", "content": build_system_prompt(request.category, approach, request.memory)},
            {"role": "user", "content": user},
        ]
        try:
            card, _ = self._complete(messages, TokenUsage(), max_tokens=CARD_MAX_TOKENS)
        except openai.APIError as error:
            raise SummaryError(f"ficha de memória: {error}") from error
        if not card.strip():
            raise SummaryError("ficha de memória: resposta vazia")
        return card.strip()

    def classify(self, transcript_text: str) -> str:
        sample = transcript_text[:CLASSIFY_SAMPLE_CHARS]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1,
                temperature=0.0,
                messages=[{"role": "user", "content": CLASSIFY_INSTRUCTION + sample}],
                extra_body=DISABLE_THINKING,
            )
        except openai.APIError as error:
            logger.warning("Classificação falhou (%s); usando modo teórico.", error)
            return THEORY_APPROACH
        answer = (response.choices[0].message.content or "").strip().upper()[:1]
        return answer if answer in ACADEMIC_APPROACH_RULES else THEORY_APPROACH

    def _generate_part(self, system_prompt: str, instruction: str, chunk: str) -> PartResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\nTRANSCRIÇÃO:\n\"\"\"\n{chunk}\n\"\"\""},
        ]
        usage = TokenUsage()
        try:
            text, finish_reason = self._complete(messages, usage)
            if finish_reason == "length":
                messages += [{"role": "assistant", "content": text}, {"role": "user", "content": CONTINUE_INSTRUCTION}]
                continuation, finish_reason = self._complete(messages, usage)
                text = f"{text}{continuation}"
        except openai.APIError as error:
            logger.error("DeepSeek falhou num trecho: %s", error)
            return PartResult(markdown="", usage=usage, error=str(error))
        if not text.strip():
            return PartResult(markdown="", usage=usage, error=f"resposta vazia (finish_reason={finish_reason})")
        return PartResult(markdown=text.strip(), usage=usage, truncated=finish_reason == "length")

    def _complete(
        self, messages: list[dict[str, str]], usage: TokenUsage, max_tokens: int = MAX_OUTPUT_TOKENS
    ) -> tuple[str, str | None]:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=messages,
            extra_body=DISABLE_THINKING,
        )
        usage.add(response.usage)
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason

    @staticmethod
    def _assemble(parts: list[PartResult], approach: str) -> SummaryResult:
        result = SummaryResult(markdown="", approach=approach)
        sections: list[str] = []
        for number, part in enumerate(parts, start=1):
            result.usage.merge(part.usage)
            if part.error is not None:
                result.failed_parts.append(number)
                sections.append(f"> ⚠️ O trecho {number} não pôde ser gerado ({part.error}). Use \"Regerar\".")
                continue
            if part.truncated:
                result.truncated_parts.append(number)
                part.markdown += f"\n\n> ⚠️ O trecho {number} foi cortado pelo limite de tamanho."
            sections.append(part.markdown)

        if len(result.failed_parts) == len(parts):
            first_error = next(part.error for part in parts if part.error)
            raise SummaryError(f"Nenhum trecho do relatório foi gerado: {first_error}")
        result.markdown = "\n\n---\n\n".join(sections)
        return result
