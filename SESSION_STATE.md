# SESSION_STATE — ScriptMax

Atualizado em 2026-10-03. Documento de passagem de contexto entre sessões.

## Resumo

A aplicação foi reescrita: saiu Streamlit + faster-whisper no Kaggle e entrou um backend FastAPI enxuto com frontend HTML/JS.

- **Transcrição:** Groq `whisper-large-v3`.
- **Relatório:** DeepSeek `deepseek-flash`, com o modo thinking desligado.
- **Testes:** 52 passando (`python -m pytest`).
- **Navegador:** validado no Chromium headless, em desktop e celular, sem erros no console.
- **Commit:** nenhum. As remoções dos arquivos antigos estão staged.

## Arquitetura

```
scriptmax/
  __main__.py      monta os serviços e sobe o uvicorn (python -m scriptmax)
  config.py        Settings via .env (load_dotenv sem override); exige APP_TOKEN se HOST não for local
  audio.py         ffmpeg -> PCM mono 16 kHz; blocos de 10 min cortados no silêncio; FLAC; detecção de silêncio
  transcription.py GroqTranscriber: verbose_json, filtro no_speech/compression, prompt de continuidade,
                   cache por bloco em data/transcripts/<hash>/
  summarization.py Summarizer: split em 12k chars, até 3 partes em paralelo, continua se finish_reason=length,
                   marca partes falhas, classificação A/B só para a categoria acadêmica
  categories.py    4 categorias + prompts: desenvolvimento-pessoal, filmes-series, academico, trabalho
  latex.py         protege $..$, $$..$$, \(..\), \[..\] e R$/US$ (span tex2jax_ignore)
  rendering.py     markdown -> restore latex -> nh3.clean (ordem importa: fix de XSS) -> template;
                   PDF via Playwright, com fallback fpdf2 + DejaVu
  templates/report.html  estilo papel/editorial; MathJax 3.2.2 com SRI; CSP com hash + sandbox
  storage.py       data/reports/<uuid hex>/ (meta.json, report.md/html/pdf, transcript.json/txt); escrita atômica
  library.py       espelha PDFs em data/biblioteca/<Categoria>/<Pasta>/...; saneia nomes; respeita MAX_PATH 240
  pipeline.py      process / regenerate / move / delete; ReportBuildError quando a transcrição foi salva mas o relatório falhou
  jobs.py          fila com 1 worker; retry; purge de uploads com mais de 3 dias
  security.py      SessionAuth (cookie HMAC httpOnly SameSite=Strict), modo só-local sem token,
                   checagem de Origin, rate limit no login
  server.py        rotas /api/*; checa auth e tamanho ANTES do parse do multipart; handler 500 com request_id
static/
  index.html, styles.css, fonts/ (Fraunces, Atkinson Hyperlegible, Martian Mono — auto-hospedadas)
  js/app.js, api.js, dom.js, jobs.js, library.js, recorder.js, recording-panel.js
tests/  conftest (fakes), audio/transcrição, latex/render, summarization, library/storage, server/segurança
```

## Funcionalidades entregues

- **Gravação no navegador:** microfone, áudio do PC (getDisplayMedia; só Chrome/Edge desktop) ou os dois mixados. Tem VU, pausa, wake lock, revisão e download antes do envio.
- **Anexos:** vários arquivos, com arrastar e soltar.
- **Categorias:** 4, cada uma com estrutura de relatório própria.
- **Pastas:** até 3 níveis, mover entre pastas, espelho real em disco e botão "Abrir no Explorer" (só local).
- **Regerar:** gera o relatório de novo sem pagar a transcrição outra vez.
- **E-mail:** opcional, com destinatário fixo no `.env`.

## Pendências do usuário

1. **Criar `GROQ_API_KEY`** e colocar no `.env`, onde a linha já existe vazia. A chamada real à Groq **nunca foi testada**.
2. **Mover o projeto para fora do OneDrive e rotacionar as chaves.** O `.env` e o `firebase-credentials*.json` foram sincronizados. Trocar a chave da DeepSeek e a senha de app do Gmail. O Firebase não é usado pelo app.
3. **Testar com microfone real e com áudio do PC.** No teste headless não havia dispositivo.
4. **Commit:** ainda não foi feito.

## Pontos em aberto / ideias

- O relatório legado em `relatorios/` (1 item) não foi migrado.
- O MathJax depende de CDN. Sem internet, o PDF sai sem fórmulas renderizadas, e o meta registra `math_rendered=false`.
- Possível: testar a xAI Grok STT (com diarização) para a categoria Trabalho.
- Para cortar custo: `TRANSCRIPTION_MODEL=whisper-large-v3-turbo`.

## Custos de referência

| Item | Preço |
|---|---|
| Groq whisper-large-v3 | US$ 0,111 por hora de áudio; grátis até 8 h por dia |
| DeepSeek flash (pico) | entrada US$ 0,30/M; saída US$ 1,20/M; metade fora do pico |
| Aula de 90 min | cerca de US$ 0,20 no total, ou cerca de US$ 0,03 dentro do plano grátis da Groq |

## Comandos

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m scriptmax          # http://127.0.0.1:8000
python -m pytest
```
