import os
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline

class Transcriber:
    def __init__(self):
        # Modelo 'base' com int8 na CPU — bom equilíbrio qualidade/velocidade
        print("Carregando modelo faster-whisper (small, CPU, int8)...")
        model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 8,  # Usa todas as threads disponíveis
        )
        # BatchedInferencePipeline: processa chunks de áudio em paralelo (~4-8x mais rápido)
        self.pipeline = BatchedInferencePipeline(model=model)
        print(f"Modelo carregado! (CPU threads: {os.cpu_count() or 4})")

    def transcribe(self, audio_filepath):
        """
        Transcreve o áudio usando faster-whisper com otimizações de velocidade:
        - VAD filter: pula trechos de silêncio (~30-50% mais rápido)
        - beam_size=1: decodificação greedy (~3-5x mais rápido)
        - batch_size=16: processamento em paralelo
        - language='pt': pula detecção automática de idioma
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_filepath}")

        try:
            print(f"Iniciando transcrição de {audio_filepath}...")
            start_time = time.time()

            segments, info = self.pipeline.transcribe(
                audio_filepath,
                beam_size=1,           # Greedy decoding — muito mais rápido, qualidade similar
                batch_size=16,         # Processa 16 chunks simultaneamente
                language="pt",         # Pula detecção automática de idioma
                vad_filter=True,       # Filtra silêncio com Silero VAD
                vad_parameters={
                    "threshold": 0.5,               # Threshold de detecção de fala
                    "min_speech_duration_ms": 250,   # Ignora falas < 250ms (ruído)
                    "min_silence_duration_ms": 1000, # Silêncio mínimo para separar chunks
                },
                without_timestamps=True,  # Mais rápido sem timestamps
            )

            full_text = []
            for segment in segments:
                full_text.append(segment.text)

            elapsed = time.time() - start_time
            duration = info.duration or 0
            duration_after_vad = info.duration_after_vad or duration

            print(f"✅ Transcrição concluída em {elapsed:.1f}s")
            print(f"   Duração do áudio: {duration:.1f}s")
            print(f"   Após VAD (sem silêncio): {duration_after_vad:.1f}s")
            print(f"   Silêncio removido: {duration - duration_after_vad:.1f}s")
            if duration > 0:
                print(f"   Velocidade: {duration / elapsed:.1f}x tempo real")

            transcription = " ".join(full_text)

            if not transcription.strip():
                return "Nenhuma voz inteligível foi detectada no áudio."

            return transcription

        except Exception as e:
            print(f"Erro durante a transcrição: {e}")
            return f"Erro ao transcrever: {e}"
