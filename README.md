# MaxClass PDF Generator (ScriptMax) 🎓🎤

Uma aplicação web construída com **Streamlit** que utiliza Inteligência Artificial para transcrever aulas gravadas ou enviadas (MP3, WAV, etc.), melhorar a qualidade do áudio e gerar resumos e apostilas detalhadas automaticamente usando o modelo **DeepSeek**.

A plataforma suporta perfeitamente aulas de exatas (Matemática, Física) renderizando equações complexas via LaTeX e MathJax.

## ✨ Funcionalidades

- **🎙️ Gravação ao Vivo ou Upload**: Grave sua aula diretamente no navegador ou envie um arquivo de áudio (`.mp3`, `.wav`, `.m4a`, etc.).
- **🔧 Melhoramento de Áudio (Denoising)**: Pipeline avançado de áudio que aplica filtro passa-banda, redução de ruído espectral e normalização para garantir a melhor qualidade antes da transcrição.
- **📝 Transcrição Ultrarrápida**: Utiliza `faster-whisper` otimizado para CPU, ignorando silêncios (VAD) e processando em lote para máxima velocidade.
- **🤖 Summarização Didática (DeepSeek)**: A IA estrutura o conhecimento em tópicos, conceitos-chave e exemplos focados em aprendizado.
- **📐 Suporte Avançado a Fórmulas**: Renderização perfeita de expressões matemáticas em LaTeX através de relatórios HTML gerados dinamicamente com MathJax.
- **📁 Organização Automática**: Todos os relatórios são salvos e organizados automaticamente em pastas baseadas no assunto da aula.
- **📄 Exportação Dual**: Baixe o resultado em HTML interativo (ideal para exatas) ou PDF de texto simples.

## 🚀 Como Executar

### 1. Pré-requisitos
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) (necessário para processamento de áudio pelo Whisper). Certifique-se de que o FFmpeg está no seu `PATH` (variáveis de ambiente do Windows).
- Uma chave de API (DeepSeek ou OpenAI compatível).

### 2. Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Configuração da API

Defina sua chave de API nas variáveis de ambiente do seu sistema antes de rodar o aplicativo.
No Windows (PowerShell):
```powershell
$env:DEEPSEEK_API_KEY="sua_chave_aqui"
```

*Obs: O arquivo `summarizer.py` atualmente busca a chave `DEEPSEEK_API_KEY` por padrão.*

### 4. Iniciando a aplicação

Execute o comando do Streamlit na pasta do projeto:

```bash
streamlit run app.py
```

O aplicativo abrirá automaticamente no seu navegador padrão (`http://localhost:8501`).

## 📁 Estrutura do Projeto

- `app.py`: O frontend em Streamlit e orquestrador principal do pipeline.
- `audio_recorder.py`: Lida com a gravação de áudio do microfone pelo navegador.
- `audio_enhancer.py`: Pipeline de DSP (Digital Signal Processing) para diminuir o ruído do áudio.
- `transcriber.py`: Integração com `faster-whisper` com otimizações de velocidade.
- `summarizer.py`: Comunicação com a API do DeepSeek, com prompts embutidos e geração de HTML/PDF.
- `relatorios/`: Diretório gerado automaticamente onde aulas processadas, organizadas por assunto, são salvas.

## ⚠️ Observações de Desempenho (Uso em CPU)

A aplicação foi rigorosamente otimizada para rodar rápido mesmo **sem GPU**, utilizando:
- VAD (Voice Activity Detection) para pular silêncios.
- Decodificação Greedy (`beam_size=1`).
- Processamento em lote (`batch_size=16`).
- Multithreading (`cpu_threads=os.cpu_count()`).
