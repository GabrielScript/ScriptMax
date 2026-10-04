"""Proteção de LaTeX e de valores monetários durante a conversão Markdown -> HTML.

O Markdown destrói LaTeX (\\( vira (, _ vira itálico). Antes de converter,
cada fórmula vira um marcador alfanumérico; depois de sanitizar o HTML, o
marcador volta como texto escapado, que o MathJax renderiza.

"R$ 100 ... R$ 200" não pode virar fórmula: moedas viram um <span> que o
MathJax ignora (classe tex2jax_ignore, padrão do MathJax 3).
"""
from __future__ import annotations

import html
import re

PLACEHOLDER_PATTERN = re.compile(r"ZZMATHPH(\d+)ZZ")
CURRENCY_PATTERN = re.compile(r"\b(?:R|US|U\$S)\$(?=\s?\d)")
DISPLAY_PATTERNS = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
)
# Regra do Pandoc: $ de abertura colado em não-espaço; $ de fechamento colado
# em não-espaço e não seguido de dígito ("$100 e $200" não é fórmula).
INLINE_PATTERNS = (
    re.compile(r"(?<![\\$])\$(?=[^\s$])(.+?)(?<![\s\\])\$(?![\d$])"),
    re.compile(r"\\\((.+?)\\\)"),
)


def protect(markdown_text: str) -> tuple[str, list[str]]:
    """Substitui moedas e fórmulas por marcadores. Retorna (texto, HTML de cada marcador)."""
    replacements: list[str] = []

    def stash(fragment_html: str) -> str:
        replacements.append(fragment_html)
        return f"ZZMATHPH{len(replacements) - 1}ZZ"

    text = CURRENCY_PATTERN.sub(
        lambda match: stash(f'<span class="tex2jax_ignore">{html.escape(match.group(0))}</span>'),
        markdown_text,
    )
    for pattern in (*DISPLAY_PATTERNS, *INLINE_PATTERNS):
        text = pattern.sub(lambda match: stash(html.escape(match.group(0), quote=True)), text)
    return text, replacements


def restore(rendered_html: str, replacements: list[str]) -> str:
    def lookup(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return replacements[index] if index < len(replacements) else match.group(0)

    return PLACEHOLDER_PATTERN.sub(lookup, rendered_html)
