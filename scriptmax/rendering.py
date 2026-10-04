"""Markdown -> HTML sanitizado (com MathJax) -> PDF via Chromium headless."""
from __future__ import annotations

import base64
import hashlib
import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import markdown
import nh3
from fpdf import FPDF
from markdown.extensions.tables import TableExtension

from scriptmax import latex

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html"
FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
MATHJAX_TIMEOUT_MS = 30_000
MATHJAX_CONFIG = (
    "window.mathjaxDone=false;"
    "window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],"
    "processEscapes:true,tags:'ams'},startup:{pageReady:()=>MathJax.startup.defaultPageReady()"
    ".then(()=>{window.mathjaxDone=true;})}};"
)
MATHJAX_CONFIG_HASH = "sha256-" + base64.b64encode(hashlib.sha256(MATHJAX_CONFIG.encode()).digest()).decode()
# Servido com sandbox: o relatório roda numa origem opaca, sem acesso ao cookie/API do app.
REPORT_CSP = (
    "default-src 'none'; "
    f"script-src '{MATHJAX_CONFIG_HASH}' https://cdn.jsdelivr.net; "
    "style-src 'unsafe-inline'; font-src https://cdn.jsdelivr.net data:; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; sandbox allow-scripts allow-popups"
)

ALLOWED_TAGS = {
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "b", "i", "u", "s", "del",
    "blockquote", "ul", "ol", "li", "code", "pre", "table", "thead", "tbody", "tr", "th", "td", "a", "sup", "sub", "span",
}
ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "th": {"align"}, "td": {"align"}, "ol": {"start"}}


@dataclass(frozen=True)
class ReportPage:
    title: str
    subtitle: str
    markdown_text: str


LIST_ITEM = r"(?:[-*+]|\d+\.)\s"
# LLMs costumam colar a lista na linha anterior ("**Itens:**\n- a"); o Markdown exige linha em branco.
LIST_WITHOUT_BLANK_LINE = re.compile(rf"(?m)^(?!\s*{LIST_ITEM})(\S.*)\n(?=\s*{LIST_ITEM})")


def markdown_to_safe_html(markdown_text: str) -> str:
    """Converte Markdown gerado por IA em HTML sem scripts/atributos perigosos, preservando LaTeX."""
    markdown_text = LIST_WITHOUT_BLANK_LINE.sub(r"\1\n\n", markdown_text)
    protected_text, replacements = latex.protect(markdown_text)
    raw_html = markdown.markdown(protected_text, extensions=[TableExtension(use_align_attribute=True), "fenced_code", "sane_lists"])
    # Restaura ANTES de sanitizar: o nh3 vê o HTML final (marcador dentro de atributo não escapa dele).
    return nh3.clean(
        latex.restore(raw_html, replacements),
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        allowed_classes={"span": {"tex2jax_ignore"}},
        url_schemes={"http", "https", "mailto"},
        link_rel="noopener noreferrer",
    )


def render_html(page: ReportPage) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    head, tail = template.split("{{BODY}}")
    head = (
        head.replace("{{MATHJAX_CONFIG}}", MATHJAX_CONFIG)
        .replace("{{TITLE}}", html.escape(page.title))
        .replace("{{SUBTITLE}}", html.escape(page.subtitle))
    )
    return head + markdown_to_safe_html(page.markdown_text) + tail


def write_pdf(html_path: Path, pdf_path: Path, page: ReportPage) -> bool:
    """Gera o PDF. Retorna True se as fórmulas foram renderizadas pelo MathJax."""
    try:
        return _write_pdf_with_chromium(html_path, pdf_path)
    except Exception as error:  # noqa: BLE001 - Playwright lança tipos variados (binário ausente, timeout, crash)
        logger.warning("Chromium indisponível (%s); gerando PDF simples.", error)
        _write_plain_pdf(pdf_path, page)
        return False


def _write_pdf_with_chromium(html_path: Path, pdf_path: Path) -> bool:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            math_rendered = True
            try:
                page.wait_for_function("() => window.mathjaxDone === true", timeout=MATHJAX_TIMEOUT_MS)
            except PlaywrightTimeout:
                math_rendered = False
                logger.warning("MathJax não terminou em %d ms (sem internet?).", MATHJAX_TIMEOUT_MS)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "18mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
            )
            return math_rendered
        finally:
            browser.close()


def _write_plain_pdf(pdf_path: Path, page: ReportPage) -> None:
    """Fallback legível (com acentos) usando DejaVu; fórmulas ficam em LaTeX cru."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", str(FONTS_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONTS_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 15)
    pdf.multi_cell(0, 9, page.title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("DejaVu", "", 9)
    pdf.multi_cell(0, 6, "PDF simplificado: fórmulas não renderizadas. Abra o HTML para a versão completa.",
                   new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 5.5, page.markdown_text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))
