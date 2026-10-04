from scriptmax import latex
from scriptmax.rendering import ReportPage, markdown_to_safe_html, render_html


def test_inline_and_display_math_survive_markdown() -> None:
    html = markdown_to_safe_html("A soma $a_1 + a_2$ e\n\n$$\\int_0^1 x^2 \\, dx$$")
    assert "$a_1 + a_2$" in html  # sem o _ virar <em>
    assert "$$\\int_0^1 x^2 \\, dx$$" in html


def test_currency_is_not_treated_as_math() -> None:
    html = markdown_to_safe_html("Custou R$ 100 e depois R$ 200.")
    assert html.count('<span class="tex2jax_ignore">R$</span>') == 2


def test_dollar_amounts_follow_pandoc_rule() -> None:
    _, replacements = latex.protect("Paguei $100 e $200 no total.")
    assert replacements == []


def test_paren_delimiters_are_protected() -> None:
    html = markdown_to_safe_html("Vetor \\(\\vec{v}\\) e bloco \\[x^2\\]")
    assert "\\(\\vec{v}\\)" in html
    assert "\\[x^2\\]" in html


def test_math_with_angle_brackets_is_escaped() -> None:
    html = markdown_to_safe_html("Se $a<b$ então ok")
    assert "$a&lt;b$" in html


def test_script_and_event_handlers_are_removed() -> None:
    html = markdown_to_safe_html('Olá <script>alert(1)</script><img src=x onerror=alert(1)> [x](javascript:alert(1))')
    assert "<script" not in html
    assert "onerror" not in html
    assert "javascript:" not in html


def test_placeholder_inside_attribute_cannot_break_out() -> None:
    html = markdown_to_safe_html('[x](http://a.com/$x"onmouseover=alert(1)$) e [y](http://b.com/R$ 1)')
    assert "onmouseover=" not in html.replace("&quot;onmouseover=", "")
    assert '<a href="http://a.com/$x"' not in html


def test_list_glued_to_previous_line_becomes_list() -> None:
    html = markdown_to_safe_html("**Propriedades:**\n- Bilinear\n- Simétrico")
    assert "<li>Bilinear</li>" in html


def test_table_alignment_kept() -> None:
    html = markdown_to_safe_html("| a | b |\n|:--|--:|\n| 1 | 2 |")
    assert '<td align="left">1</td>' in html


def test_title_is_escaped_in_template() -> None:
    page = ReportPage(title="<b>x</b>", subtitle="s", markdown_text="corpo")
    html = render_html(page)
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<b>x</b>" not in html
