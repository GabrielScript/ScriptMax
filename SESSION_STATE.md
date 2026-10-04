# SESSION_STATE — ScriptMax

Atualizado em 2026-10-04. Documento de passagem de contexto entre sessões.

## Resumo

O ScriptMax está em produção no Google Cloud Run. Desde 2026-10-03 a stack é: backend FastAPI, frontend HTML/JS puro,
transcrição na Groq com `whisper-large-v3` e relatórios no DeepSeek com `deepseek-flash`, que a API confirma ser o
**DeepSeek-V4.1-Flash**.

- **Link:** https://scriptmax-572877712489.us-central1.run.app. A revisão `scriptmax-00001-fcn` recebe 100% do tráfego.
- **Login:** pelo `APP_TOKEN`, guardado no Secret Manager. Para ver:
  `gcloud secrets versions access latest --secret=app-token --project=scriptmax-app`
- **Repositório:** https://github.com/GabrielScript/ScriptMax. Último commit: `eb2ce55`. Árvore limpa.
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

1. **Docker Desktop:** mover o disco para `D:\DockerData` em Settings → Resources → Advanced → Disk image location.
   - A primeira tentativa, para `D:\Docker`, falhou com "já em uso". Essa pasta está vazia e pode ser apagada pelo Explorer.
   - O armazenamento do Docker pode ter corrompido quando o C: encheu. Se acontecer, use Troubleshoot → Clean / Purge data.
2. **`D:\DockerDesktopWSL`** tem um `docker_data.vhdx` antigo de **52,8 GB**, de abril de 2026. Avaliar se dá para
   apagar, depois de confirmar que o Docker não usa mais esse caminho.
3. **Apagar a pasta vazia antiga** `C:\Users\gabri\OneDrive\Área de Trabalho\Aplicativos\ScriptMax` e esvaziar a
   lixeira do OneDrive, o que libera espaço na nuvem.
4. **Token do MCP do GitHub inválido** (erro 401). Gerar um PAT novo; por enquanto o push é feito pelo `git`.
5. **Rotacionar chaves:** DeepSeek e senha de app do Gmail, porque passaram pelo OneDrive. Depois, atualizar os
   segredos com `gcloud secrets versions add <nome> --data-file=... --project=scriptmax-app`.
6. **Testar no app publicado** com microfone real e com áudio do PC, inclusive pelo celular.
7. O relatório de teste "Teste Cloud Run (pode apagar)", na pasta Testes, pode ser apagado pela interface.
8. **Decidir o destino do `resolveja-jp-8347`**, que ficou sem faturamento.

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
