# ScriptMax

Grave (microfone, áudio do PC ou os dois) ou anexe um áudio. O ScriptMax transcreve com **Whisper large-v3 (Groq)**,
gera um relatório com **DeepSeek** seguindo instruções específicas da categoria e arquiva o PDF em pastas.

Sem GPU, sem Kaggle: tudo roda no seu PC e as partes pesadas são APIs baratas.

## Categorias

| Categoria | O relatório traz |
|---|---|
| Desenvolvimento pessoal / Psicologia | ideia central, conceitos, mecanismos, ferramentas passo a passo, perguntas de reflexão, plano de ação |
| Filmes / Séries / Documentários | ficha, sinopse, desenvolvimento, personagens, fatos e argumentos, temas, falas marcantes (com aviso de spoiler) |
| Acadêmico / Conhecimento | apostila; detecta exatas e usa LaTeX; "pontos de atenção para prova" e glossário |
| Trabalho | reunião → resumo executivo, decisões, itens de ação; treinamento → procedimentos e checklist |
| Tech / Tecnologia | visão geral, arquitetura, tabela da stack, passo a passo com código, trade-offs, armadilhas e glossário |

## Instalação

Requisitos: Python 3.12, [ffmpeg](https://ffmpeg.org) no PATH (`winget install Gyan.FFmpeg`).

```powershell
python -m pip install -r requirements-dev.txt   # inclui pytest; produção usa só requirements.txt
python -m playwright install chromium
copy .env.example .env   # preencha GROQ_API_KEY e DEEPSEEK_API_KEY
python -m scriptmax       # abre em http://127.0.0.1:8000
```

## Como funciona

1. **Áudio** → ffmpeg normaliza para mono 16 kHz e corta blocos de ~10 min **no silêncio** (limite de 25 MB da Groq).
2. **Transcrição** → Groq `whisper-large-v3`, com o fim do bloco anterior como contexto. Segmentos de silêncio/alucinação
   são descartados com os limiares do próprio Whisper. Cada bloco fica em cache: falhar no meio não cobra de novo.
3. **Relatório** → DeepSeek com prompt da categoria (modo *thinking* desligado; prefixo fixo aproveita o cache da API).
   Trechos cortados por limite de tamanho são continuados; trechos que falham ficam marcados e podem ser **regerados**
   sem transcrever de novo.
4. **Saída** → HTML sanitizado com MathJax + PDF via Chromium, copiado para
   `data/biblioteca/<Categoria>/<Pasta>/<Subpasta>/`.

### Memória entre relatórios (séries, cursos, stacks)

Relatórios na **mesma categoria e pasta exata** formam uma sequência, ordenada pela data de geração. Ao gerar o item N,
o relatório recebe uma **ficha** (300–600 palavras) de cada item anterior, cita-os no texto quando há relação real e
termina com a seção **Conexões com os anteriores**. Cada relatório pronto guarda a sua ficha em
`data/reports/<id>/memory.md`.

- A memória é recalculada a cada geração: mover, apagar ou regerar um item nunca a deixa desatualizada. Regerar o item 3
  enxerga só os itens 1 e 2 (sem spoilers).
- Itens antigos sem ficha ganham uma na hora. Pasta na raiz da categoria não tem memória.
- Limite de ~30 mil caracteres (≈10 fichas): acima disso ficam as mais recentes e o prompt informa quantas foram omitidas.
- A ficha nunca derruba o relatório: se falhar, o status avisa e ela é refeita no próximo item.
- Custo extra: uma chamada curta por relatório (~US$ 0,002). A memória vai no prompt de sistema, do mais antigo ao mais
  novo, para o cache da DeepSeek aproveitar o prefixo (veja `.claude/skills/deepseek-cache-economy`).

## Custos (outubro/2026)

| Etapa | Preço | Aula de 90 min |
|---|---|---|
| Groq whisper-large-v3 | US$ 0,111/h (grátis até 8 h/dia no plano free) | ~US$ 0,17 ou grátis |
| DeepSeek flash | US$ 0,30/M entrada, 1,20/M saída (metade fora do pico) | ~US$ 0,03 |

Para gastar menos: `TRANSCRIPTION_MODEL=whisper-large-v3-turbo` (US$ 0,04/h, um pouco menos preciso) e processe fora
do horário de pico da DeepSeek (01–04h e 06–10h UTC são pico).

## Gravar pelo celular

O navegador só libera microfone em HTTPS. Defina `APP_TOKEN` (16+ caracteres) no `.env` e use um túnel:

```powershell
ngrok http 8000
```

Abra o link `https://...ngrok-free.app` no celular e entre com o token. Sem `APP_TOKEN`, o app recusa qualquer acesso
que não seja do próprio PC.

## Deploy no Cloud Run

Produção: projeto `scriptmax-app`, região `us-central1`, serviço `scriptmax`. Chaves e `APP_TOKEN` ficam no Secret
Manager; relatórios e biblioteca no bucket `gs://scriptmax-app-data`, montado em `/data` (Cloud Storage FUSE).
Com `BEHIND_PROXY=1` o app sobe no Hypercorn em HTTP/2 (h2c), o que libera uploads acima de 32 MB.

Rode no PowerShell (o Git Bash converte `/data` em caminho do Windows):

```powershell
gcloud run deploy scriptmax --source . --project=scriptmax-app --region=us-central1 `
  --service-account=scriptmax-run@scriptmax-app.iam.gserviceaccount.com --allow-unauthenticated `
  --use-http2 --no-cpu-throttling --cpu=1 --memory=2Gi --min-instances=0 --max-instances=1 `
  --timeout=3600 --execution-environment=gen2 `
  "--add-volume=name=data,type=cloud-storage,bucket=scriptmax-app-data,mount-options=implicit-dirs" `
  "--add-volume-mount=volume=data,mount-path=/data" `
  "--set-secrets=GROQ_API_KEY=groq-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest,EMAIL_PASSWORD=email-password:latest,APP_TOKEN=app-token:latest" `
  --quiet
```

- `--no-cpu-throttling`: os jobs rodam em thread depois da resposta HTTP; sem isso a CPU congela.
- `--max-instances=1`: o estado dos jobs fica em memória; mais de uma instância perderia o acompanhamento.
- `EMAIL_USER`, `EMAIL_RECIPIENT` e `MAX_UPLOAD_MB=300` já estão no serviço e são mantidos em novos deploys.
- Ver o token de login: `gcloud secrets versions access latest --secret=app-token --project=scriptmax-app`.

## Segurança

- Chaves de API só no servidor; o navegador nunca as vê.
- Login troca o token por cookie `httpOnly` + `SameSite=Strict`; tentativas limitadas (10 a cada 15 min).
- CSP restritiva no app; relatórios abertos em *sandbox* (origem isolada) e com HTML sanitizado (nh3).
- Destinatário de e-mail fixo no `.env`; nomes de pasta saneados contra path traversal.

## Testes

```powershell
python -m pytest
```
