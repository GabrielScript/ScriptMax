import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio


class Transcriber:
    """
    Transcrição com faster-whisper large-v3 — 100% otimizada para 2x GPU (T4 x2).

    Estratégia:
      - GPU: carrega 'large-v3' (FP16) em CADA GPU disponível (até 2).
        Para um único áudio longo, ele é dividido em 2 metades (corte feito num
        ponto de silêncio) e cada metade é transcrita EM PARALELO, uma por GPU.
        Isso satura as duas T4 ao mesmo tempo → ~2x mais rápido por arquivo.
      - Sem GPU: fallback para CPU ('small', INT8) sequencial.
    """

    SAMPLE_RATE = 16000

    def __init__(self):
        print("Iniciando varredura de hardware para o Whisper...")
        self.sample_rate = self.SAMPLE_RATE
        self.device = "cpu"
        self.pipelines = []            # lista de (pipeline_ou_modelo, label)
        self.use_batched_pipeline = False

        gpu_count = self._detect_gpus()
        if gpu_count >= 1:
            try:
                self._init_gpu(gpu_count)
            except Exception as e:
                print(f"⚠️ Falha ao inicializar GPU(s): {e}")
                self._init_cpu()
        else:
            print("⚠️ Nenhuma GPU CUDA detectada.")
            self._init_cpu()

    # ------------------------------------------------------------------ setup
    def _detect_gpus(self):
        """Conta GPUs CUDA disponíveis (via torch, se presente)."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    def _init_gpu(self, gpu_count):
        """Carrega large-v3 em até 2 GPUs (T4 x2)."""
        n = min(gpu_count, 2)  # otimizado para até 2 GPUs
        print(f"Detectadas {gpu_count} GPU(s). Alocando 'large-v3' (FP16) em {n} GPU(s)...")

        pipelines = []
        for idx in range(n):
            try:
                model = WhisperModel(
                    "large-v3", device="cuda",
                    device_index=[idx], compute_type="float16",
                )
                pipelines.append((BatchedInferencePipeline(model=model), f"cuda:{idx}"))
                print(f"  ✅ large-v3 carregado na GPU {idx}")
            except Exception as e:
                print(f"  ⚠️ GPU {idx} não pôde carregar o large-v3: {e}")

        if not pipelines:
            raise RuntimeError("Nenhuma GPU conseguiu carregar o large-v3.")

        self.pipelines = pipelines
        self.device = "cuda"
        self.use_batched_pipeline = True
        modo = "T4x2 paralelo" if len(pipelines) >= 2 else "GPU única"
        print(f"✅ Pronto: large-v3 em {len(pipelines)} GPU(s) — modo {modo}.")

    def _init_cpu(self):
        """Fallback definitivo: CPU com modelo 'small' quantizado (INT8)."""
        cpu_cores = os.cpu_count() or 4
        print(f"🔄 Usando CPU ('small', INT8, {cpu_cores} threads)...")
        model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=cpu_cores)
        self.pipelines = [(model, "cpu")]
        self.device = "cpu"
        self.use_batched_pipeline = False
        print("✅ Fallback para CPU concluído.")

    # ------------------------------------------------------------ parâmetros
    def _decode_params(self):
        """Parâmetros de decodificação — MODO MÁXIMA VELOCIDADE.

        beam_size=1 (greedy): com VAD + large-v3, a perda de precisão é mínima e
        o ganho é grande. O VAD já remove o silêncio onde o greedy mais erra.
        """
        return {
            "beam_size": 1,
            "language": "pt",
            "vad_filter": True,
            "without_timestamps": True,
            "vad_parameters": {
                "threshold": 0.5,
                # 100ms: o português é denso em interjeições curtas e vitais ("é", "tá", "já").
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 1000,
            },
        }

    def _run_pipeline(self, pipeline, audio, batched):
        """Executa a transcrição de um áudio (path ou array numpy) num pipeline."""
        params = self._decode_params()
        if batched:
            # GPU: lotes grandes maximizam o throughput dos CUDA cores da T4.
            segments, info = pipeline.transcribe(audio, batch_size=32, **params)
        else:
            # CPU: sequencial. condition_on_previous_text=False corta alucinação
            # em cascata/repetição e ainda acelera um pouco.
            segments, info = pipeline.transcribe(
                audio, condition_on_previous_text=False, **params
            )
        text = " ".join(seg.text for seg in segments)
        return text, info

    # ------------------------------------------------------------ T4x2 split
    def _find_split_point(self, audio, window_s=10):
        """Acha um ponto de baixa energia (silêncio) perto do meio do áudio.

        Evita cortar uma palavra no meio ao dividir o áudio entre as 2 GPUs.
        """
        n = len(audio)
        mid = n // 2
        sr = self.sample_rate
        w = window_s * sr
        lo = max(0, mid - w)
        hi = min(n, mid + w)

        seg = np.abs(audio[lo:hi])
        win = max(1, int(0.5 * sr))  # janela de 0,5s
        if len(seg) <= win:
            return mid

        # Energia por janela deslizante via soma cumulativa (rápido).
        c = np.cumsum(seg)
        energy = c[win:] - c[:-win]
        rel = int(np.argmin(energy))
        return lo + rel + win // 2

    def _transcribe_split(self, audio):
        """Divide o áudio em 2 e transcreve uma metade por GPU, em paralelo."""
        split = self._find_split_point(audio)
        parts = [audio[:split], audio[split:]]
        print(f"⚡ T4x2: corte em {split / self.sample_rate:.1f}s — 1 metade por GPU (paralelo).")

        results = [None, None]

        def work(i):
            results[i] = self._run_pipeline(self.pipelines[i][0], parts[i], batched=True)[0]

        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(work, range(2)))

        return " ".join(r for r in results if r)

    # --------------------------------------------------------------- público
    def transcribe(self, audio_filepath):
        """Transcreve o áudio, usando 2 GPUs em paralelo quando disponíveis."""
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Erro Crítico de I/O: Arquivo não localizado em {audio_filepath}")

        try:
            print(f"Processando áudio: {audio_filepath}...")
            start_time = time.time()

            multi_gpu = self.use_batched_pipeline and len(self.pipelines) >= 2

            if multi_gpu:
                # Decodifica 1x (mono 16kHz) para poder dividir entre as GPUs.
                audio = decode_audio(audio_filepath, sampling_rate=self.sample_rate)
                duration = len(audio) / self.sample_rate
                if duration < 90:
                    # Áudio curto: dividir não compensa o overhead → 1 GPU.
                    text = self._run_pipeline(self.pipelines[0][0], audio, batched=True)[0]
                else:
                    text = self._transcribe_split(audio)
            else:
                # GPU única (batched) ou CPU (sequencial) — passa o path direto.
                pipeline = self.pipelines[0][0]
                text, info = self._run_pipeline(pipeline, audio_filepath, self.use_batched_pipeline)
                duration = info.duration or 0

            elapsed = time.time() - start_time
            print(f"✅ Decodificação finalizada em {elapsed:.2f}s")
            if duration > 0 and elapsed > 0:
                print(f"   Duração do áudio: {duration:.1f}s | Aceleração: {duration / elapsed:.2f}x tempo real")

            text = text.strip()
            if not text:
                return "Aviso: Nenhuma fala inteligível foi detectada pelo VAD."
            return text

        except Exception as e:
            print(f"Erro fatal na transcrição: {e}")
            return f"Erro de processamento: {e}"
