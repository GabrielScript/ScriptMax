# SESSION_STATE — ScriptMax

Atualizado em 2026-10-04. Documento de passagem de contexto entre sessões.

## Resumo

O ScriptMax está em produção no Google Cloud Run. Desde 2026-10-03 a stack é: backend FastAPI, frontend HTML/JS puro,
transcrição na Groq com `whisper-large-v3` e relatórios no DeepSeek com `deepseek-flash`, que a API confirma ser o
**DeepSeek-V4.1-Flash**.

- **Link:** https://scriptmax-572877712489.us-central1.run.app. A revisão `scriptmax-00002-hjz` recebe 100% do tráfego
  (criada só para carregar os segredos rotacionados; mesma imagem da `00001-fcn`).
- **Login:** pelo `APP_TOKEN`, guardado no Secret Manager. Para ver:
  `gcloud secrets versions access latest --secret=app-token --project=scriptmax-app`
- **Repositório:** https://github.com/GabrielScript/ScriptMax. Último commit: `eb2ce55`. Só o `SESSION_STATE.md` está fora de commit.
  **Commit e push são feitos pelo usuário**, não pelo Claude.
- **Testes:** 57 passando (`python -m pytest`).
- **Local do projeto:** `D:\Projetos\ScriptMax`. Saiu do OneDrive nesta sessão porque o C: estava 100% cheio.

## O que foi feito nesta sessão (2026-10-04)

1. **Commit e push** da reescrita (`3456572`). Antes disso, chequei que nenhum segredo está versionado.
2. **Adaptação para o Cloud Run** (`1f6e048`):
   - Com `BEHIND_PROXY=1`, o app sobe no **Hypercorn** com HTTP/2 em texto puro (h2c). Isso remove o limite de
     32 MB por requisição do Cloud Run. `ProxyFixMiddleware(trusted_hops=1)` dá o esquema https e o IP real do
     cliente, que o cookie `Secure` e o rate limit usam.
   - Sem a flag, continua rodando com uvicorn, como antes, para uso local e ngrok.
   - `Dockerfile`: `python:3.12-slim-bookworm` com ffmpeg, `chromium-headless-shell` e usuário não-root.
   - `.dockerignore` em lista de permissão; o `.gcloudignore` só o inclui.
3. **Limpeza com /simplify** (`eb2ce55`):
   - Os imports dos servidores agora são lazy.
   - `requirements-dev.txt` separado do `requirements.txt`.
   - Lista única de ignore e `compileall` no build.
4. **Infraestrutura no GCP:**
   - Projeto novo `scriptmax-app`. A conta de faturamento tinha atingido o limite de 5 projetos, e **o
     faturamento do `resolveja-jp-8347` foi desvinculado** por decisão do usuário. O Cloud Run desse projeto está parado.
   - APIs ativadas, segredos criados, bucket, conta de serviço, permissões IAM mínimas, deploy, alerta de orçamento e
     limpeza automática no Artifact Registry.
5. **Teste ponta a ponta em produção:**
   - Um envio de 40 MB chegou ao app.
   - Login e cookie `Secure` funcionaram.
   - Um áudio de teste passou por Groq, DeepSeek, gerou o PDF (77 KB, com MathJax) e foi gravado no bucket.
6. **Skills do Google Cloud** instaladas em `.claude/skills`, só as essenciais: `gcloud`, `cloud-run-basics`,
   `cloud-build-basics`, `google-cloud-storage-basics`, `google-cloud-storage-fuse` e `google-cloud-waf-cost-optimization`.
7. **Migração para o D:** o projeto foi copiado e conferido (559 de 559 arquivos, 9,4 GB), e a cópia do OneDrive foi
   apagada. A memória do Claude foi copiada para a chave do novo caminho.

## Sessão seguinte (2026-10-04, noite): manutenção
1. **Chaves rotacionadas:** DeepSeek e senha de app do Gmail.
   - Testadas antes de publicar: DeepSeek respondeu 200; login SMTP OK com e sem espaços.
   - Versão 2 criada em `deepseek-api-key` e `email-password`, com a senha do Gmail gravada sem espaços.
   - A versão 1 dos dois foi **desativada**, não destruída. O usuário já apagou as chaves antigas nos provedores.
   - Nova revisão `scriptmax-00002-hjz`, criada com `--update-labels=secrets-rotated=2026-10-04`, sem rebuild. Responde 200 e não tem avisos no log.
   - Ainda falta um relatório e um envio de e-mail reais em produção com as chaves novas.
2. **Docker Desktop movido para `D:\DockerData`.**
   - O motor tinha travado (`docker ps` não respondia) e foi preciso forçar o fechamento com `wsl --shutdown`.
   - Depois a mudança deu certo. Os containers `alepha-db` e `terraiq-postgis` estão de pé.
   - O C: passou de 9,1 para 19,7 GB livres.
3. **Pastas apagadas no D:** `D:\Docker`, que estava vazia, e `D:\DockerDesktopWSL`, com o vhdx de 52,8 GB de abril. O D: ficou com 345 GB livres.
4. **MCP do GitHub com token novo:** fine-grained, válido até 2027-01-02, com permissão de admin no ScriptMax.
   - Trocado via `claude mcp remove` e `claude mcp add-json` no escopo user (`~/.claude.json`). A cópia no `~/.claude/settings.json` também foi atualizada, mas o Claude Code não lê esse arquivo.
   - O servidor roda via Docker (`ghcr.io/github/github-mcp-server`), então só conecta com o Docker de pé.
   - Precisa reiniciar o Claude Code para reconectar.
   - Ficou a pasta vazia `D:\tmp_keys`, para o usuário apagar pelo Explorer.
5. **Método de troca de segredos:** o usuário salva a chave num arquivo fora do repositório (ou no `.env`). O Claude testa a chave sem exibi-la, grava com `gcloud secrets versions add --data-file`, cria uma revisão nova e apaga o arquivo.

## Infraestrutura (GCP)

| Recurso | Valor |
|---|---|
| Projeto / região | `scriptmax-app` / `us-central1` |
| Serviço Cloud Run | `scriptmax`: 1 vCPU, 2 GiB, mín. 0 e máx. 1 instância, `--no-cpu-throttling`, `--use-http2`, timeout de 3600 s, gen2 |
| Conta de serviço de runtime | `scriptmax-run@scriptmax-app.iam.gserviceaccount.com` |
| Dados | bucket `gs://scriptmax-app-data` montado em `/data` (GCS FUSE); privado e sem soft delete |
| Segredos | `groq-api-key`, `deepseek-api-key`, `email-password`, `app-token` (réplica em us-central1) |
| Variáveis de ambiente | `EMAIL_USER`, `EMAIL_RECIPIENT`, `MAX_UPLOAD_MB=300`; no Dockerfile: `HOST=0.0.0.0`, `BEHIND_PROXY=1`, `DATA_DIR=/data` |
| Imagens | Artifact Registry `cloud-run-source-deploy` (~476 MB); a limpeza mantém as 2 mais recentes |
| Orçamento | alerta de R$ 5 por mês, só para este projeto (50%, 90%, 100% e previsão) |

**Por que esses flags:**
- `--no-cpu-throttling`: os jobs rodam numa thread depois da resposta HTTP. Sem o flag, a CPU congela.
- `--max-instances=1`: o estado dos jobs fica em memória.
- `--use-http2`: libera uploads acima de 32 MB. Depende do Hypercorn, ligado por `BEHIND_PROXY=1`.

**Deploy:** use o comando que está no README, rodando no **PowerShell**. O Git Bash converte `/data` em caminho do
Windows, e `MSYS_NO_PATHCONV=1` quebra o wrapper do gcloud. Comandos de billing ou budget precisam de
`--billing-project=scriptmax-app`, porque o projeto padrão do gcloud é o `incorpodata-app`.

## Custos estimados por mês

| Item | Custo |
|---|---|
| Cloud Run | R$ 0 até cerca de 62 h de instância ligada por mês. Essa cota grátis é da conta de faturamento e é dividida com incorpodata, clinica-voice e hv-dashboards. Depois disso, cerca de R$ 0,43 por hora. |
| Cloud Storage, Secret Manager, Cloud Build, Logging | R$ 0, dentro do plano grátis |
| Artifact Registry | R$ 0 a cerca de R$ 0,30 |
| Groq | R$ 0, grátis até 8 h de áudio por dia |
| DeepSeek | cerca de R$ 0,16 por aula de 90 min, ou cerca de R$ 3 com 20 aulas |
| **Total** | **cerca de R$ 0 a R$ 4 por mês** |

Decisão do usuário: **manter o DeepSeek**. Ficou avaliada e não implementada a opção de um modo híbrido: modelo
grátis da Groq com o DeepSeek como reserva.

## Pendências

Feitas em 2026-10-04 (noite): disco do Docker no D:, vhdx antigo apagado, PAT do GitHub novo e chaves rotacionadas.

1. **Testar no app publicado** com microfone real e com áudio do PC, inclusive pelo celular. Isso também valida em produção o DeepSeek e o e-mail com as chaves novas.
2. **Confirmar o MCP do GitHub** depois de reiniciar o Claude Code, com o Docker de pé.
3. O relatório de teste "Teste Cloud Run (pode apagar)", na pasta Testes, pode ser apagado pela interface.
4. **Apagar a pasta vazia antiga** `C:\Users\gabri\OneDrive\Área de Trabalho\Aplicativos\ScriptMax` e esvaziar a
   lixeira do OneDrive.
5. **Decidir o destino do `resolveja-jp-8347`**, que ficou sem faturamento.
6. Apagar a pasta vazia `D:\tmp_keys`.
7. Commit do `SESSION_STATE.md`, que o usuário faz.

## Limitações conhecidas

- Se a instância reiniciar no meio de um job, o job se perde. Use o botão de tentar de novo.
- Cold start de alguns segundos depois de um tempo ocioso.
- O botão "Abrir no Explorer" só aparece no uso local, no Windows.
- O MathJax depende de CDN. Sem acesso a ela, o PDF sai sem fórmulas renderizadas.
- Ideias que ficaram de fora do /simplify:
  - usar só o Hypercorn também no uso local;
  - centralizar a confiança em proxy, hoje em `__main__`, `security.py` e no README;
  - upload direto para o GCS por URL assinada;
  - ffmpeg estático na imagem.

## Arquitetura

```
scriptmax/
  __main__.py      monta os serviços; uvicorn (local) ou Hypercorn h2c + ProxyFix (BEHIND_PROXY=1)
  config.py        Settings via .env; exige APP_TOKEN se HOST não for local; BEHIND_PROXY, DATA_DIR
  audio.py         ffmpeg -> PCM mono 16 kHz; blocos de 10 min cortados no silêncio; FLAC
  transcription.py GroqTranscriber com cache por bloco em data/transcripts/<hash>/
  summarization.py Summarizer: partes em paralelo, continua se a resposta for cortada
  categories.py    4 categorias + prompts
  latex.py / rendering.py  Markdown seguro (nh3) + MathJax; PDF via Playwright, com fallback fpdf2
  storage.py / library.py  data/reports/<id>/ e espelho de PDFs em data/biblioteca/<Categoria>/<Pasta>/
  pipeline.py / jobs.py    process/regenerate/move/delete; fila com 1 worker e retry
  security.py / server.py  sessão por cookie HMAC, CSRF por Origin, rate limit; rotas /api/*
static/            index.html, styles.css, fontes auto-hospedadas, js/*
tests/             57 testes (inclui test_config.py)
Dockerfile, .dockerignore, .gcloudignore, requirements.txt, requirements-dev.txt
```

## Comandos

```powershell
# Local
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python -m scriptmax          # http://127.0.0.1:8000
python -m pytest

# Git
git add -A; git commit -m "mensagem"; git push origin main

# Produção (comando completo de deploy no README)
gcloud run services describe scriptmax --project=scriptmax-app --region=us-central1 --format="value(status.url)"
gcloud run services logs read scriptmax --project=scriptmax-app --region=us-central1 --limit=50
```
