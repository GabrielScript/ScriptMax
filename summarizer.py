import os
import re
from concurrent.futures import ThreadPoolExecutor

import markdown
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()


class Summarizer:
    # DeepSeek V4 Flash — rápido e barato, forte em PT/LaTeX. API compatível
    # com OpenAI (base_url apontando para a DeepSeek).
    MODEL = "deepseek-v4-flash"
    BASE_URL = "https://api.deepseek.com"

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("WARNING: DEEPSEEK_API_KEY not found in .env")

        # Cliente reaproveitado entre threads (httpx é thread-safe).
        self.client = (
            OpenAI(api_key=api_key, base_url=self.BASE_URL) if api_key else None
        )

    # Tamanho-alvo de cada bloco da transcrição (em caracteres) na geração em
    # múltiplas passagens. ~15000 chars ≈ ~3700 palavras de fala por bloco.
    # Bloco grande = menos partes geradas = relatório final mais enxuto.
    # Áudios pequenos/médios cabem numa passagem única.
    CHUNK_CHARS = 15000

    def summarize(self, transcription_text):
        """Gera um relatório IMENSO via geração em múltiplas passagens (chunking).

        A transcrição é dividida em blocos sequenciais; cada bloco vira uma seção
        detalhada do relatório. Assim o tamanho final não fica preso no teto de
        tokens de uma única chamada do modelo.
        """
        text = (transcription_text or "").strip()
        if not text:
            return "Erro: transcrição vazia."

        chunks = self._split_text(text, self.CHUNK_CHARS)
        total = len(chunks)

        if total == 1:
            section = self._summarize_chunk(chunks[0], part=1, total=1)
            if section and not section.startswith("Erro"):
                return section
            return f"Erro ao gerar resumo: {section}"

        # Os blocos são independentes (a continuidade é só instrução de prompt),
        # então geramos todas as partes EM PARALELO. O tempo total cai de
        # N×latência para ~1×latência. Resultado remontado na ordem original.
        print(f"Gerando relatório em {total} passagens (paralelo)...")
        parts = [None] * total
        max_workers = min(total, 3)  # teto baixo evita estourar o rate limit da API

        def _work(idx):
            i = idx + 1
            print(f"  → Enviando parte {i}/{total} para o DeepSeek...")
            section = self._summarize_chunk(chunks[idx], part=i, total=total)
            if section and not section.startswith("Erro"):
                parts[idx] = section
            else:
                parts[idx] = f"> ⚠️ Falha ao gerar a parte {i}: {section}"

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_work, range(total)))

        produced = [p for p in parts if p]
        if not produced:
            return "Erro ao gerar resumo: nenhuma parte foi produzida."

        return "\n\n---\n\n".join(produced)

    def _split_text(self, text, max_chars):
        """Divide o texto em blocos <= max_chars respeitando fronteiras de frase."""
        if len(text) <= max_chars:
            return [text]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _summarize_chunk(self, transcription_text, part=1, total=1):
        """Send one transcription chunk to DeepSeek and ask for a detailed section."""
        if total > 1:
            intro_rule = ("Inclua uma introdução geral do conteúdo."
                          if part == 1 else
                          "NÃO repita introduções gerais — continue de onde a parte anterior parou.")
            outro_rule = ("Inclua uma conclusão final consolidando tudo."
                          if part == total else
                          "NÃO escreva conclusão final ainda (haverá mais partes a seguir).")
            context_note = f"""

## CONTEXTO DE CONTINUIDADE (IMPORTANTE)
Esta é a **PARTE {part} de {total}** de uma transcrição longa, dividida em blocos sequenciais.
- Gere o relatório detalhado APENAS do conteúdo deste bloco.
- {intro_rule}
- {outro_rule}
- Não escreva "Parte {part}" no título; apenas continue o conteúdo de forma fluida e contínua.
"""
        else:
            context_note = ""

        prompt = f"""
Você é um especialista em educação estruturada, elaboração de atas executivas e formatação avançada de conteúdos complexos.

Abaixo está a transcrição (gerada por IA) de uma gravação. Sua tarefa é analisar o contexto dessa gravação e gerar um relatório organizando perfeitamente a transcrição de forma fluida.

## PRINCÍPIO CENTRAL: DENSO, NÃO LONGO
O relatório deve ser COMPLETO, mas NÃO inflado. Cubra TODOS os pontos abordados no áudio, porém:
- **Cada informação aparece UMA única vez.** Proibido repetir a mesma ideia, conceito ou dado em seções diferentes ou reformulado com outras palavras.
- **Sem enchimento.** Nada de introduções genéricas, frases de transição vazias, recapitulações ou conclusões que apenas repetem o que já foi dito.
- **Prefira o conteúdo à extensão.** Use o número de palavras necessário para registrar o ponto com precisão — nem mais, nem menos. Um relatório curto que cobre tudo é melhor que um longo e redundante.
- **Nada de "água".** Se uma frase não acrescenta informação nova, remova-a.
{context_note}
## PASSO 1: ANÁLISE DE CONTEXTO
Antes de iniciar o relatório, identifique sobre o que a transcrição se trata e escolha UMA das duas abordagens abaixo:

### ABORDAGEM A: Exatas, Engenharia e Computação (Modo Matemático)
Se o áudio contiver equações, matemática de nível médio/superior, programação, física ou teorias exatas:
1. **TODAS** as fórmulas, equações e expressões matemáticas devem MANDATORIAMENTE usar notação LaTeX:
   - Inline: `$formula$` (ex: $x^2 + y^2 = r^2$)
   - Bloco centralizado: `$$formula$$` (ex: $$\\int_0^1 x^2 \\, dx = \\frac{{1}}{{3}}$$)
2. Identifique e converta matrizes (`\\begin{{bmatrix}}`), sistemas (`\\begin{{cases}}`), derivadas, integrais, limites e símbolos matemáticos espalhados pela fala em LaTeX.
3. Se o professor disser uma fórmula por extenso, converta-a para a fórmula matemática formal.
4. Estruture como a "Apostila Definitiva" sobre a matéria.

### ABORDAGEM B: Humanidades, Teoria, Direito e Reuniões corporativas (Modo Orgânico)
Se o áudio for sobre ciências sociais, negócios, literatura, leis ou uma reunião de trabalho diária:
1. **NÃO FORCE fórmulas matemáticas**. Não tente criar matrizes/equações artificiais (ex: não escreva "Receita = Lucro + Despesas" em LaTeX, escreva naturalmente no texto).
2. Escreva com foco gigantesco em **fluidez de leitura**. Use parágrafos muito bem escritos, conectivos lógicos e encadeamento de ideias.
3. Se for uma reunião: Estruture as metas, defina as responsabilidades distribuídas (Action Items), os acordos, os prazos e as deliberações.
4. Se for aula teórica: Foque nos argumentos do professor, nos fatos históricos/legais, exemplos práticos discutidos e filosofias, como um ensaio ou ata executiva perfeitamente polida.

## REGRAS GERAIS E OBRIGATÓRIAS (Para ambas as abordagens):
- Organize por grandes tópicos e subtópicos lógicos usando # e ##.
- Use **negrito** para destacar conceitos-chave, pessoas citadas, regras de negócio ou termos técnicos vitais.
- Use > (blockquote) para citações diretas, regras universais, artigos de leis ou definições irrefutáveis.
- NÃO omita detalhes cruciais ou conversas de rodapé que construíram a base lógica da decisão.
- Mas TAMBÉM não invente, não estenda artificialmente e não reescreva o mesmo ponto de formas diferentes só para parecer mais completo.

Transcrição Bruta:
-------------------------
{transcription_text}
-------------------------

Por favor, forneça o relatório final bem estruturado em português, baseado na Abordagem que melhor se adequar ao texto acima.
"""
        if self.client is None:
            return "Erro: DEEPSEEK_API_KEY não configurada no .env."

        system_prompt = (
            "Você é um assistente educacional especialista em criar relatórios "
            "completos, densos e didáticos — cobrindo todos os pontos sem redundância "
            "nem enchimento. Cada informação aparece uma única vez. Quando o conteúdo "
            "envolve matemática, você SEMPRE usa notação LaTeX para fórmulas, equações, "
            "matrizes e expressões matemáticas. Use $...$ para inline e $$...$$ para blocos."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            text_out = response.choices[0].message.content or ""
            if not text_out.strip():
                reason = getattr(response.choices[0], "finish_reason", None) or "resposta vazia"
                return f"Erro: o modelo não retornou conteúdo (finish_reason={reason})."
            return text_out
        except Exception as e:
            print(f"Erro ao acessar a API da DeepSeek: {e}")
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
        window.mathjaxDone = false;
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
            }},
            startup: {{
                pageReady: () => {{
                    return MathJax.startup.defaultPageReady().then(() => {{
                        window.mathjaxDone = true;
                    }});
                }}
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

    def generate_pdf(self, report_text, output_filename="relatorio_aula.pdf", html_source=None):
        """
        Gera PDF a partir do HTML renderizado com MathJax (Playwright/Chromium).

        Otimizações:
          - Reusa o HTML já salvo (`html_source`) em vez de gerar um 2º temporário.
          - Sem sleeps fixos: espera as fontes (`document.fonts.ready`) e, SÓ quando
            há fórmulas, espera o MathJax sinalizar conclusão (`window.mathjaxDone`).
          - `wait_until="load"` em vez de `networkidle` (não trava esperando CDN ocioso).
        """
        import time as _time

        # Reusa o HTML já escrito por generate_html_report; só cria temp se não veio.
        temp_html = None
        if html_source and os.path.exists(html_source):
            html_path = html_source
        else:
            html_path = output_filename.replace(".pdf", "_temp_for_pdf.html")
            self.generate_html_report(report_text, html_path)
            temp_html = html_path

        # Relatório sem LaTeX (Abordagem B) não precisa esperar o MathJax.
        has_formulas = "$" in (report_text or "")

        try:
            from playwright.sync_api import sync_playwright

            print("📄 Gerando PDF (Playwright + MathJax)...")
            start = _time.time()

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                abs_path = os.path.abspath(html_path)
                page.goto(f"file:///{abs_path}", wait_until="load")

                # Garante que as web fonts carregaram antes de imprimir.
                try:
                    page.evaluate("async () => { await document.fonts.ready; }")
                except Exception:
                    pass

                # Espera o MathJax SÓ se houver fórmulas — sinal real, sem sleep fixo.
                if has_formulas:
                    try:
                        page.wait_for_function(
                            "() => window.mathjaxDone === true", timeout=20000
                        )
                    except Exception:
                        pass  # timeout: segue com o que já renderizou

                page.pdf(
                    path=output_filename,
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "20mm",
                        "bottom": "20mm",
                        "left": "15mm",
                        "right": "15mm"
                    }
                )

                browser.close()

            elapsed = _time.time() - start
            print(f"✅ PDF gerado em {elapsed:.1f}s: {output_filename}")

        except Exception as e:
            print(f"⚠️ Erro ao gerar PDF via Playwright: {e}")
            print("   Gerando PDF simples como fallback...")
            self._generate_pdf_fallback(report_text, output_filename)

        finally:
            # Limpa só o temporário que NÓS criamos (nunca o HTML salvo do usuário).
            if temp_html and os.path.exists(temp_html):
                try:
                    os.remove(temp_html)
                except Exception:
                    pass

        return output_filename

    def _generate_pdf_fallback(self, report_text, output_filename):
        """Fallback: gera PDF básico com texto puro (sem fórmulas renderizadas)."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(0, 10, "Relatorio da Aula", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "(Formulas nao renderizadas - instale Playwright para PDF completo)", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        pdf.set_font("Helvetica", size=10)

        cleaned = self.clean_text_for_pdf(report_text)
        safe = cleaned.encode('ascii', errors='replace').decode('ascii')
        try:
            pdf.multi_cell(0, 6, txt=safe)
        except:
            pdf.multi_cell(0, 6, txt="Erro ao gerar conteudo. Abra o HTML.")

        pdf.output(output_filename)
        print(f"✅ PDF fallback gerado: {output_filename}")
