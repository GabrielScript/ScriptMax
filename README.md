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

## Instalação

Requisitos: Python 3.12, [ffmpeg](https://ffmpeg.org) no PATH (`winget install Gyan.FFmpeg`).

```powershell
python -m pip install -r requirements.txt
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

## Segurança

- Chaves de API só no servidor; o navegador nunca as vê.
- Login troca o token por cookie `httpOnly` + `SameSite=Strict`; tentativas limitadas (10 a cada 15 min).
- CSP restritiva no app; relatórios abertos em *sandbox* (origem isolada) e com HTML sanitizado (nh3).
- Destinatário de e-mail fixo no `.env`; nomes de pasta saneados contra path traversal.

## Testes

```powershell
python -m pytest
```
