# SESSION_STATE — ScriptMax

Atualizado em 2026-10-09. Documento de passagem de contexto entre sessões.
**Leia primeiro a seção "Sessão de 2026-10-09" abaixo: há trabalho local não commitado e não publicado.**

## Resumo

O ScriptMax está em produção no Google Cloud Run. Desde 2026-10-03 a stack é: backend FastAPI, frontend HTML/JS puro,
transcrição na Groq com `whisper-large-v3` e relatórios no DeepSeek com `deepseek-flash`, que a API confirma ser o
**DeepSeek-V4.1-Flash**.

- **Link:** https://scriptmax-572877712489.us-central1.run.app. A revisão `scriptmax-00002-hjz` recebe 100% do tráfego
  (criada só para carregar os segredos rotacionados; mesma imagem da `00001-fcn`).
- **Login:** pelo `APP_TOKEN`, guardado no Secret Manager. Para ver:
  `gcloud secrets versions access latest --secret=app-token --project=scriptmax-app`
- **Repositório:** https://github.com/GabrielScript/ScriptMax. Último commit: `31b53af`. As features de 2026-10-09 estão fora de commit.
  **Commit e push são feitos pelo usuário**, não pelo Claude.
- **Testes:** 91 passando localmente em 2026-10-09 (`python -m pytest`; eram 57 em 2026-10-04). A produção ainda roda a versão antiga.
- **Local do projeto:** `D:\Projetos\ScriptMax`. Saiu do OneDrive nesta sessão porque o C: estava 100% cheio.

## Sessão de 2026-10-09 (noite): teste pelo celular e viabilidade comercial

**Estado:** nenhum código mudou. Nada commitado, nada publicado. Servidor local **desligado**.

### Testar pelo celular
- **O link do Cloud Run funciona, mas roda a versão antiga** (sem Tech, sugestões no título e memória). Não houve deploy.
  Commit e push não são pré-requisito do deploy: `gcloud run deploy --source .` sobe o código **local**. Publicar sem
  commit deixa a produção à frente do GitHub.
- **O usuário não quer usar o ngrok.** O caminho escolhido é a rede local:
  - `$env:APP_TOKEN = '<16+ caracteres>'; $env:HOST = '0.0.0.0'; python -m scriptmax`
  - No celular, na mesma rede: `http://192.168.0.2:8000` (IP do PC pelo cabo Ethernet em 2026-10-09; pode mudar).
  - O `.env` **não tem `APP_TOKEN` nem `HOST`**. Sem token, o app recusa acesso que não seja local, e com `HOST` fora de
    localhost ele nem sobe. Nesta sessão o token foi só variável de ambiente; nada foi gravado no `.env`.
  - O firewall do Windows já libera o `python.exe` no perfil Público, que é o perfil da rede Ethernet.
  - **Limitação:** em `http` (sem HTTPS) o navegador do celular bloqueia o microfone (`isSecureContext` falso em
    `static/js/recorder.js`). Upload de arquivo, biblioteca, categorias e memória funcionam. A gravação ao vivo pelo
    celular só funciona com HTTPS (ngrok ou Cloud Run).
  - Cookie de sessão: `secure` só quando o esquema é https (`server.py:141`), então o login funciona em http na LAN.
- Antes de subir, havia um `python -m scriptmax` antigo, sem token, na porta 8000. Ele foi encerrado após conferir que
  `/api/jobs` estava vazio.
- O servidor da LAN foi **morto pelo Claude Code por falta de memória** no PC (sessão ociosa). Não foi falha do app. Para
  evitar isso: iniciar o Claude Code com `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.

### Por que testar localmente antes do deploy
1. Os prompts da memória (`CARD_INSTRUCTION`, `CONNECTIONS_INSTRUCTION`) nunca foram vistos por um modelo real.
2. A produção grava no bucket: fichas ruins ficariam salvas e entrariam no prompt dos itens seguintes da pasta.
3. Sem commit não há ponto de volta no GitHub.

### Viabilidade comercial (conversa, sem decisão)
- O mercado existe: apps de aula para resumo (Coconote, TurboLearn, StudyFetch, Mindgrasp) e de reunião (Otter,
  Fireflies, Granola, Plaud). Os preços não foram pesquisados nesta sessão.
- **Diferenciais:**
  - PT-BR nativo com prompts por categoria;
  - memória entre relatórios da mesma pasta;
  - PDF com LaTeX;
  - biblioteca em pastas;
  - custo de ~R$ 0,16 por aula.
- **Riscos:**
  - concorrência grátis (NotebookLM, ChatGPT, Gemini e recursos nativos do celular);
  - distribuição;
  - taxa de 15–30% das lojas;
  - Apple recusa app que é só um site embrulhado;
  - gravação longa com a tela bloqueada exige app nativo;
  - LGPD e consentimento para gravar aulas;
  - a categoria Filmes/Séries é zona cinzenta de direito autoral.
- **Arquitetura atual é de usuário único** (`APP_TOKEN`, jobs em memória, `max-instances=1`). Um produto pago exige:
  contas de usuário, banco de dados, fila de jobs, cotas e cobrança.
- **Sugestão dada:**
  - escolher um nicho (universitários de exatas e medicina, ou concurseiros);
  - landing page e lista de espera;
  - 10 a 20 usuários na versão web antes das lojas;
  - plano grátis com 2–3 aulas por mês e pago entre R$ 19,90 e R$ 29,90.
- Ficou oferecido e não feito: pesquisar os preços atuais dos concorrentes.

## Sessão de 2026-10-09: categoria Tech, sugestões no título e memória entre relatórios

**Estado:** tudo abaixo está só no disco local. Nada commitado, nada em produção (o Cloud Run segue com a versão antiga).
Commit e push são do usuário. `scriptmax/library.py` e `tests/test_library_storage.py` já estavam modificados antes
desta sessão e não são desta feature.

### O que foi feito

1. **Categoria "Tech"** (`Category.TECH = "tech"`, pasta `Tech/` na biblioteca): regras próprias em `categories.py`
   (visão geral, arquitetura, tabela da stack com versão, passo a passo com código, trade-offs, armadilhas, glossário;
   proíbe inventar comando/flag/versão). Card roxo na UI (claro e escuro); com 5 cards o último ocupa a linha inteira.
2. **Sugestões no campo Título/assunto** (`<datalist id="subject-options">`): lista as pastas da categoria escolhida com a
   contagem. Escolher uma pasta da lista preenche o campo Pasta, limpa o título e devolve o foco a ele; digitar à mão o nome
   de uma pasta não faz nada. Detecta a escolha por `inputType === 'insertReplacementText'` (ou sem `inputType`).
   `foldersOf` virou `folderCounts` em `static/js/library.js`. Fechar o diálogo "Mover" restaura as sugestões do formulário.
3. **Memória entre relatórios** (abordagem 1: ficha por relatório). Novo `scriptmax/memory.py`; ficha em
   `data/reports/<id>/memory.md`; `ReportMeta.memory_items`; `ReportStore.remove_file`; `SummaryRequest.memory`;
   `Summarizer.write_memory_card`; `_collect_memory` e `_write_card` em `pipeline.py`. Regras:
   - Sequência = **mesma categoria e pasta exata**, ordenada por `created_at`. Raiz da categoria (pasta vazia) não tem memória.
   - Só entram itens anteriores com `report_ready`; irmão sem ficha ganha uma na hora (backfill, system sem memória).
   - Memória no **prompt de sistema**, depois das regras da categoria, **do mais antigo ao mais novo**.
   - Limite `MEMORY_MAX_CHARS` = 30 mil (~10 fichas, ~9 mil tokens): acima disso ficam as **mais novas** e o prompt avisa
     quantas antigas foram omitidas. `MAX_CARD_CHARS` = 6 mil por ficha.
   - A instrução "# Conexões com os anteriores" vai no `user` e **só no último trecho**; zero irmãos = sem instrução.
   - A ficha do item atual é gerada logo após o relatório e **antes** da renderização, com o mesmo system prompt dos trechos.
   - Falha da ficha ou da memória **nunca derruba o relatório**: aviso no status final. Regerar apaga a ficha velha antes.
   - Subtítulo do PDF mostra "memória: N itens".
4. **Skills do projeto** em `.claude/skills/`: `deepseek-cache-economy` (criada e testada), `dsh-error-handling` e
   `dsh-ci-test-reliability` (copiadas do repositório oficial `deepseek-ai/deepseek-harness`, MIT, com a licença).
5. README (categoria Tech e seção "Memória entre relatórios") e esta seção atualizados.

### Decisões do usuário (não reabrir sem motivo)

- Relação com os anteriores: referências no texto + seção "Conexões" no fim (opção A).
- Ordem dos itens: pela data de geração. Quem é "anterior": só a mesma pasta exata.
- Modelo: **manter a DeepSeek V4.1 Flash** (`deepseek-flash`). Projeto fica no `D:` (HDD); o `C:` (SSD) tem só ~13 GB livres.
- Pular spec e plano em arquivo para acelerar: as seções 1 e 2 do design, aprovadas no chat, valem como design.

### Insights e achados

- **Custo real medido** (Ep. 1 de 64 min): 8.235 tokens de entrada, 5.572 de saída = ~US$ 0,009 no pico, ~US$ 0,005 fora do
  pico. 73% do custo é saída. Pico (tarifa dobrada): 01–04 e 06–10 UTC em dias úteis = 22h–01h e 03h–07h em Brasília.
- **Cache hit medido: 3%** (256 de 8.235). Os 3 trechos rodam em paralelo e chegam juntos antes de existir cache; o
  comentário em `summarization.py` promete mais do que ocorre. Cache hit custa US$ 0,003/M contra 0,15/M do miss (50x).
  Não existe garantia de que requests simultâneos se ajudem (a documentação não cobre). **Ainda não medimos a taxa depois da memória.**
- **Ordem das fichas importa pro cache:** do mais antigo ao mais novo, o item novo entra no fim e o começo do prompt fica
  igual ao do episódio anterior. Do mais novo ao mais antigo, o bloco inteiro perde o cache a cada item.
- Regras de cache da DeepSeek (guia oficial e notas do time): casa por prefixo desde o primeiro token; unidade só conta se
  coincidir inteira; chamada auxiliar deve reusar o system prompt e pôr a instrução nova por último; nada variável no começo.
- **Teste da skill com subagentes (4 amostras, Sonnet):** sem a skill, sob pressão, o agente aceitou "fichas da mais nova
  à mais antiga" e system próprio pra ficha; com a skill recusou os 3 pedidos. A skill foi corrigida depois (a ficha reusa o
  system, não "o mesmo conteúdo"; estouro de limite corta as mais antigas). **Essa correção não foi retestada.**
- **Jev (TypeSafe)** pesquisado: modelo de decisões tipadas (Noul/sim-não, Choice, Score), US$ 0,042/M de entrada, saída
  grátis, sem português confirmado, cadastro pausado em 22/09/2026. **Não vale integrar:** o custo está em texto gerado, e o
  ganho máximo seria ~4% de menos de 1 centavo. Só faria sentido pra priorizar fichas se uma pasta estourar o limite.
- Concorrentes na faixa barata (preços de agregadores, não confirmados nas páginas oficiais exceto Gemini e DeepSeek):
  GPT-5.6 Luna ~US$ 0,20/1,20, Qwen3.8-Flash ~0,15/0,47 (verboso), Gemini 3.1 Flash-Lite 0,25/1,50. Nenhum compensa trocar.
- `deepseek-recipe` (tokenizador oficial em Python) existe mas exige Rust/OpenCV; descartado por ora. Limite por caracteres basta.
- Repositório `deepseek-harness`: das ~16 skills, só `dsh-error-handling` e `dsh-ci-test-reliability` servem aqui. As demais
  são específicas do Harness (Cordis, docs bilíngues, PRs empilhados, Office, sandbox).

### Erros e correções desta sessão

- Apaguei sem querer `build_system_prompt` e `build_part_instruction` ao editar com um parâmetro errado
  (`new_str` em vez de `new_string`); restaurei e conferi pelo diff.
- A contagem das sugestões incluía subfolders, mas a memória usa a pasta exata: corrigido (`folderCounts` conta só a pasta;
  pasta que só tem subpastas mostra "só subpastas").
- **Testes instáveis:** `created_at` tinha resolução de segundos; itens no mesmo segundo empatavam e a ordem virava o uuid
  aleatório. Passou a `timespec="milliseconds"`. Em produção nada muda. Cinco rodadas seguidas passaram, o que é teste de
  estresse, não prova.
- `ACADEMIC_APPROACH_RULES[approach]` virou `.get(approach, teórico)`: antes, abordagem desconhecida quebrava.
- Clone completo do `deepseek-harness` travou; usei a API do GitHub e `raw.githubusercontent.com` (arquivos individuais).
- Playwright travou clicando num `<label>` coberto pelo `<input type=radio>` (do próprio script de teste, não do app).
- Parte do trabalho foi feita com o modelo Sonnet por engano; o certo era Opus. A revisão final foi feita no Opus.
- `bash` recusa heredoc com aspas aninhadas contendo `'''`; use as ferramentas de edição.

### O que NÃO foi verificado

- **Nenhuma chamada real à DeepSeek** com a memória: `CARD_INSTRUCTION`, `CONNECTIONS_INSTRUCTION` e `MEMORY_HEADER` são
  rascunho que nenhum modelo real viu (a seção 3 do design, a redação dos prompts, foi pulada a pedido).
- O clique real numa sugestão do datalist: a automação não abre a lista nativa. O usuário disse que testou e pediu para
  derrubar o servidor, mas não relatou o resultado.
- JavaScript não tem testes automatizados; foi verificado só no navegador.
- Cloud Run com a feature nova: não houve deploy.

### Próximos passos

1. **Teste de aceitação (~US$ 0,007):** `python -m scriptmax`, biblioteca, Ep. 1 (pasta `Series`, categoria Filmes) →
   **Regerar** → ler `data/reports/bb61f…/memory.md`. Se a ficha ficar ruim, ajustar `CARD_INSTRUCTION` em `summarization.py`.
2. Enviar o Ep. 2 na mesma pasta: conferir "memória: 1 item" no subtítulo e a seção "Conexões com os anteriores" no PDF.
3. Ler `cached_prompt_tokens` / `prompt_tokens` no `meta.json` do Ep. 2 e comparar com os 3% do Ep. 1 (usar a skill
   `deepseek-cache-economy`). Se o cache continuar baixo, avaliar rodar o primeiro trecho antes dos outros (hoje descartado
   por falta de medição).
4. Decidir sobre `.claude/skills/deepseek-chat/deepseek-chat/SKILL.md`: está aninhada errado, sem o script, e com o modelo
   `deepseek-chat` que não consta mais nos preços. Consertar (mover, usar `deepseek-flash`, script Python com o pacote
   `openai`) ou apagar.
5. Commit e push (usuário), depois deploy no Cloud Run pelo comando do README.
6. Limitações conhecidas da memória: a ficha derivada de um áudio entra no prompt de sistema dos itens seguintes, então
   instruções embutidas no áudio poderiam persistir (app de usuário único; risco baixo). Rótulos "Item N" podem pular
   número se uma ficha falhar. Itens fora de ordem de envio ficam na ordem da data de geração.

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

## Sessão de 2026-10-04 (madrugada → 19h40): verificações

- Tráfego: 100% na `scriptmax-00002-hjz`, HTTP 200.
- MCP do GitHub conectado (leu os commits até `5fc1a5e`).
- **Teste real em produção via API:** 5 min da aula "Algebra Linear (Aula 01)", categoria Acadêmico, pasta Testes,
  com e-mail. Groq, DeepSeek, PDF e e-mail OK em 32 s. Ou seja, as chaves novas do DeepSeek e do Gmail funcionam.
  Gerou o relatório "Teste producao chaves novas (pode apagar)".
- **`resolveja-jp-8347` excluído** (`DELETE_REQUESTED`). Dá para desfazer por ~30 dias com
  `gcloud projects undelete resolveja-jp-8347`.
- **Pasta antiga do OneDrive apagada.** Ela estava travada por 6 processos órfãos que tinham a pasta como diretório
  de trabalho: servidores MCP (googlemaps e mongodb) de sessões antigas do Claude e um `python3` de plugin. O usuário
  autorizou, e eles e mais 29 `node.exe` órfãos de 2 e 3/out foram encerrados.

## Pendências

0. **Ver "Próximos passos" da sessão de 2026-10-09** (teste de aceitação da memória, commit e deploy).
1. **Testar no app publicado** com microfone real e com áudio do PC, inclusive pelo celular (o caminho de upload,
   DeepSeek e e-mail já foi validado).
2. Commit do `SESSION_STATE.md`, que o usuário faz.

Feitos pelo usuário em 2026-10-04: os 2 relatórios de teste apagados pela interface (o bucket foi conferido), lixeira
do OneDrive esvaziada e `D:\tmp_keys` apagada.

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
  categories.py    5 categorias + prompts
  latex.py / rendering.py  Markdown seguro (nh3) + MathJax; PDF via Playwright, com fallback fpdf2
  storage.py / library.py  data/reports/<id>/ e espelho de PDFs em data/biblioteca/<Categoria>/<Pasta>/
  memory.py                fichas por relatório (memory.md) e bloco de memória da pasta exata
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
