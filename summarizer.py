import os
import re
import markdown
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()


class Summarizer:
    def __init__(self):
        # Using DeepSeek API
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("WARNING: DEEPSEEK_API_KEY not found in .env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def summarize(self, transcription_text):
        """Send the full transcription to DeepSeek and ask for a detailed, math-aware summary."""
        prompt = f"""
Você é um especialista em educação e sumarização de conteúdos complexos, com domínio total de notação matemática.

Abaixo está a transcrição de uma aula ou palestra. Sua tarefa é gerar um relatório EXTREMAMENTE extenso, longo e detalhado.

## REGRAS OBRIGATÓRIAS PARA CONTEÚDO MATEMÁTICO:

1. **TODAS** as fórmulas, equações e expressões matemáticas devem usar notação LaTeX:
   - Inline: `$formula$` (ex: $x^2 + y^2 = r^2$)
   - Bloco centralizado: `$$formula$$` (ex: $$\\int_0^1 x^2 \\, dx = \\frac{{1}}{{3}}$$)

2. Para **matrizes**, use:
   $$\\begin{{bmatrix}} a & b \\\\ c & d \\end{{bmatrix}}$$

3. Para **sistemas de equações**, use:
   $$\\begin{{cases}} x + y = 5 \\\\ 2x - y = 1 \\end{{cases}}$$

4. Para **derivadas**: $\\frac{{d}}{{dx}}f(x)$, $f'(x)$, $\\frac{{\\partial f}}{{\\partial x}}$
5. Para **integrais**: $\\int_a^b f(x)\\,dx$, $\\iint$, $\\oint$
6. Para **limites**: $\\lim_{{x \\to \\infty}} f(x)$
7. Para **somatórios**: $\\sum_{{i=1}}^{{n}} a_i$, $\\prod_{{i=1}}^{{n}} a_i$
8. Para **vetores**: $\\vec{{v}}$, $\\hat{{u}}$
9. Para **símbolos**: $\\forall$, $\\exists$, $\\in$, $\\subset$, $\\cup$, $\\cap$, $\\infty$, $\\neq$, $\\leq$, $\\geq$, $\\approx$
10. Para **frações**: $\\frac{{numerador}}{{denominador}}$
11. Para **raízes**: $\\sqrt{{x}}$, $\\sqrt[n]{{x}}$

## REGRAS DE FORMATAÇÃO:

- Organize por grandes tópicos e subtópicos usando # e ##
- Use bullet points para listas
- Use **negrito** para conceitos-chave
- Use > (blockquote) para teoremas e definições importantes
- Não omita detalhes cruciais
- Aja como se estivesse criando a "Apostila Definitiva" sobre o assunto

## IMPORTANTE:
- Mesmo que o professor tenha DITO a fórmula por extenso (ex: "integral de x ao quadrado dx"), você DEVE convertê-la para notação LaTeX ($\\int x^2 \\, dx$)
- Sempre forneça o contexto e explicação ao redor da fórmula
- Se houver demonstrações ou provas, reproduza os passos com as fórmulas em LaTeX

Transcrição:
-------------------------
{transcription_text}
-------------------------

Por favor, forneça o relatório final bem estruturado em português, com toda a matemática em LaTeX.
"""
        print("Enviando texto para o modelo DeepSeek...")
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Você é um assistente educacional especialista em criar resumos super detalhados e didáticos. Quando o conteúdo envolve matemática, você SEMPRE usa notação LaTeX para fórmulas, equações, matrizes e expressões matemáticas. Use $...$ para inline e $$...$$ para blocos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000,
                temperature=0.3
            )
            report_text = response.choices[0].message.content
            return report_text
        except Exception as e:
            print(f"Erro ao acessar DeepSeek API: {e}")
            return f"Erro ao gerar resumo: {e}"

    def generate_html_report(self, report_text, output_filename="relatorio_aula.html"):
        """Generate a beautiful HTML report with MathJax for perfect LaTeX rendering."""

        # Convert markdown to HTML (preserving LaTeX delimiters)
        # We need to protect LaTeX from being mangled by the markdown parser
        protected_text, latex_map = self._protect_latex(report_text)
        html_body = markdown.markdown(
            protected_text,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        html_body = self._restore_latex(html_body, latex_map)

        html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Detalhado da Aula</title>

    <!-- MathJax para renderização de fórmulas LaTeX -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: true,
                processEnvironments: true,
                tags: 'ams'
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --border: #334155;
            --success: #22c55e;
            --warning: #f59e0b;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.8;
            font-size: 16px;
            padding: 0;
        }}

        .header {{
            background: linear-gradient(135deg, var(--accent), #8b5cf6, #ec4899);
            padding: 3rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext x='10' y='50' font-size='40' opacity='0.1'%3E∫∑√π%3C/text%3E%3C/svg%3E");
            opacity: 0.3;
        }}

        .header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}

        .header p {{
            color: rgba(255,255,255,0.85);
            margin-top: 0.5rem;
            font-size: 1rem;
            position: relative;
            z-index: 1;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .content {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2.5rem;
            margin-top: -2rem;
            position: relative;
            z-index: 2;
            border: 1px solid var(--border);
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary);
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }}

        h1 {{ font-size: 1.8rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; color: var(--accent-light); }}
        h3 {{ font-size: 1.25rem; color: var(--text-secondary); }}

        p {{
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }}

        strong {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        ul, ol {{
            margin: 1rem 0;
            padding-left: 1.5rem;
        }}

        li {{
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }}

        li::marker {{
            color: var(--accent);
        }}

        blockquote {{
            border-left: 4px solid var(--accent);
            background: var(--accent-glow);
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 0 8px 8px 0;
        }}

        blockquote p {{
            color: var(--text-primary);
            margin-bottom: 0;
        }}

        code {{
            background: var(--bg-primary);
            color: var(--accent-light);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }}

        pre {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            text-align: left;
        }}

        th {{
            background: var(--bg-primary);
            color: var(--accent-light);
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: rgba(99, 102, 241, 0.05);
        }}

        /* MathJax styling overrides */
        .MathJax {{
            font-size: 1.1em !important;
        }}

        mjx-container[display="true"] {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin: 1.5rem 0 !important;
            overflow-x: auto;
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }}

        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        /* Print styles */
        @media print {{
            :root {{
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-card: #ffffff;
                --text-primary: #1e293b;
                --text-secondary: #475569;
                --border: #e2e8f0;
                --accent-glow: rgba(99, 102, 241, 0.08);
            }}

            body {{
                background: white;
                color: #1e293b;
                font-size: 11pt;
            }}

            .header {{
                background: #1e293b !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .content {{
                box-shadow: none;
                border: 1px solid #e2e8f0;
            }}

            mjx-container[display="true"] {{
                background: #f8fafc !important;
                border-color: #e2e8f0 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Relatório Detalhado da Aula</h1>
        <p>Gerado automaticamente por MaxClass PDF Generator</p>
    </div>

    <div class="container">
        <div class="content">
            {html_body}
        </div>
    </div>

    <div class="footer">
        <p>Gerado por MaxClass PDF Generator — Powered by DeepSeek & MathJax</p>
    </div>
</body>
</html>"""

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_template)

        print(f"✅ Relatório HTML gerado: {output_filename}")
        return output_filename

    def _protect_latex(self, text):
        """Protect LaTeX expressions from markdown parser by replacing with placeholders."""
        latex_map = {}
        counter = [0]

        def replace_match(match):
            key = f"LATEXPLACEHOLDER{counter[0]}ENDPLACEHOLDER"
            latex_map[key] = match.group(0)
            counter[0] += 1
            return key

        # Protect block math first ($$...$$), then inline ($...$)
        text = re.sub(r'\$\$.+?\$\$', replace_match, text, flags=re.DOTALL)
        text = re.sub(r'\$(?!\$).+?\$', replace_match, text)

        return text, latex_map

    def _restore_latex(self, html, latex_map):
        """Restore LaTeX expressions after markdown processing."""
        for key, value in latex_map.items():
            html = html.replace(key, value)
        return html

    def clean_text_for_pdf(self, text):
        """Prepares text for FPDF, removing LaTeX notation for plain PDF."""
        # Remove LaTeX delimiters but keep the content
        text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$', r'\1', text)
        # Clean up LaTeX commands for plain text
        text = text.replace('\\frac', '')
        text = text.replace('\\int', '∫')
        text = text.replace('\\sum', 'Σ')
        text = text.replace('\\prod', 'Π')
        text = text.replace('\\sqrt', '√')
        text = text.replace('\\infty', '∞')
        text = text.replace('\\leq', '≤')
        text = text.replace('\\geq', '≥')
        text = text.replace('\\neq', '≠')
        text = text.replace('\\approx', '≈')
        text = text.replace('\\times', '×')
        text = text.replace('\\cdot', '·')
        text = text.replace('\\pm', '±')
        text = text.replace('\\to', '→')
        text = text.replace('\\rightarrow', '→')
        text = text.replace('\\leftarrow', '←')
        text = text.replace('\\forall', '∀')
        text = text.replace('\\exists', '∃')
        text = text.replace('\\in', '∈')
        text = text.replace('\\subset', '⊂')
        text = text.replace('\\cup', '∪')
        text = text.replace('\\cap', '∩')
        text = text.replace('\\vec', '')
        text = text.replace('\\hat', '')
        text = text.replace('\\begin{bmatrix}', '[')
        text = text.replace('\\end{bmatrix}', ']')
        text = text.replace('\\begin{cases}', '{')
        text = text.replace('\\end{cases}', '}')
        text = text.replace('\\\\', '\n')
        text = text.replace('\\,', ' ')
        text = re.sub(r'\{|\}', '', text)
        return text

    def generate_pdf(self, report_text, output_filename="relatorio_aula.pdf"):
        """Generate a PDF with full Unicode support (DejaVu Sans from fpdf2)."""
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # fpdf2 inclui DejaVu Sans embutida com suporte completo a Unicode
        # (inclui ∈, ℝ, ℂ, ∫, Σ, ∞, →, ∀, ∃, etc.)
        font_name = "DejaVu"
        try:
            pdf.add_font("DejaVu", "", os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"))
            pdf.add_font("DejaVu", "B", os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf"))
        except Exception:
            # Tentar caminhos do sistema Windows
            try:
                pdf.add_font("DejaVu", "", "C:/Windows/Fonts/DejaVuSans.ttf")
                pdf.add_font("DejaVu", "B", "C:/Windows/Fonts/DejaVuSans-Bold.ttf")
            except Exception:
                # Último recurso: Arial com sanitização agressiva
                font_name = "Helvetica"
                print("⚠️ DejaVu Sans não encontrada, usando Helvetica (sem suporte Unicode completo)")

        cleaned_text = self.clean_text_for_pdf(report_text)

        # Sanitizar caracteres que a fonte pode não suportar
        def safe_text(text):
            """Remove caracteres que a fonte não consegue renderizar."""
            try:
                text.encode('latin-1')
                return text
            except UnicodeEncodeError:
                if font_name == "Helvetica":
                    # Para Helvetica, manter apenas ASCII + Latin-1
                    return ''.join(c if ord(c) < 256 else '?' for c in text)
                return text  # DejaVu suporta Unicode

        # Title
        pdf.set_font(font_name, style="B", size=16)
        pdf.cell(0, 10, safe_text("Relatório Detalhado da Aula"), ln=True, align='C')
        pdf.ln(5)
        pdf.set_font(font_name, size=9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, safe_text("(Para fórmulas perfeitas, abra o arquivo HTML no navegador)"), ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        # Body — com tratamento robusto de erros
        pdf.set_font(font_name, size=11)
        try:
            pdf.multi_cell(0, 7, txt=safe_text(cleaned_text))
        except Exception as e:
            print(f"⚠️ Erro no PDF com texto completo: {e}")
            try:
                # Fallback: remover todos os caracteres não-ASCII
                ascii_text = cleaned_text.encode('ascii', errors='replace').decode('ascii')
                pdf.multi_cell(0, 7, txt=ascii_text)
            except Exception as e2:
                print(f"⚠️ Fallback final do PDF: {e2}")
                pdf.multi_cell(0, 7, txt="Erro ao gerar PDF. Por favor, use o arquivo HTML.")

        try:
            pdf.output(output_filename)
            print(f"✅ PDF gerado: {output_filename}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar PDF: {e}")
            # Criar um PDF mínimo para não quebrar o fluxo
            pdf_min = FPDF()
            pdf_min.add_page()
            pdf_min.set_font("Helvetica", size=12)
            pdf_min.multi_cell(0, 10, txt="Relatorio gerado. Para formulas, abra o arquivo HTML.")
            pdf_min.output(output_filename)

        return output_filename
