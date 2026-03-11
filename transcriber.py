import os
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline

class Transcriber:
    def __init__(self):
        print("Iniciando varredura de arquitetura de hardware para o Whisper...")
        
        self.device = "cpu"
        self.model = None
        self.use_batched_pipeline = False
        
        # Estratégia de Fallback em Cascata
        try:
            # Tentativa 1: Arquitetura Ideal (GPU CUDA + Modelo Grande)
            print("Tentando alocar 'large-v3' na GPU (FP16)...")
            self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            self.device = "cuda"
            self.use_batched_pipeline = True
            print("✅ Sucesso: GPU CUDA detectada. Modelo 'large-v3' carregado na VRAM.")
            
        except Exception as e:
            print(f"⚠️ Falha ao carregar 'large-v3' na GPU: {e}")
            try:
                # Tentativa 2: Restrição de VRAM (GPU CUDA + Modelo Menor)
                print("Tentando fallback para modelo 'small' na GPU (mantendo aceleração CUDA)...")
                self.model = WhisperModel("small", device="cuda", compute_type="float16")
                self.device = "cuda"
                self.use_batched_pipeline = True
                print("✅ Sucesso parcial: Modelo 'small' alocado na GPU. Aceleração mantida.")
                
            except Exception as e:
                # Tentativa 3: Ausência de GPU (Fallback definitivo para CPU)
                print("⚠️ Falha severa no CUDA ou GPU não encontrada.")
                print("🔄 Alternando para computação distribuída em CPU (INT8)...")
                
                # Na CPU, limitamos as threads ao número de núcleos físicos para evitar 
                # thrashing de cache L1/L2 pelo OpenMP/MKL.
                cpu_cores = os.cpu_count() or 4
                self.model = WhisperModel(
                    "small", 
                    device="cpu", 
                    compute_type="int8", 
                    cpu_threads=cpu_cores
                )
                self.device = "cpu"
                # Na CPU, o batching por cima do paralelismo nativo de threads pode degradar performance.
                self.use_batched_pipeline = False 
                print(f"✅ Fallback para CPU concluído. Utilizando {cpu_cores} threads com quantização INT8.")

        # O BatchedInferencePipeline é instanciado condicionalmente.
        # Só o ativamos na GPU, onde os CUDA cores devoram batches em paralelo de forma eficiente.
        if self.use_batched_pipeline:
            self.pipeline = BatchedInferencePipeline(model=self.model)
        else:
            self.pipeline = self.model

    def transcribe(self, audio_filepath):
        """
        Executa a transcrição baseada no hardware detectado.
        Os hiperparâmetros foram ajustados para resiliência semântica e evitar alucinações.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Erro Crítico de I/O: Arquivo não localizado em {audio_filepath}")

        try:
            print(f"Processando vetor de áudio: {audio_filepath}...")
            start_time = time.time()

            # Parâmetros de decodificação universais (Independentes de CPU/GPU)
            # Beam Size: Mantido em 5 (Padrão). Greedy (1) destrói a precisão em áudios ruidosos.
            decode_params = {
                "beam_size": 5,
                "language": "pt",
                "vad_filter": True,
                "without_timestamps": True,
                "vad_parameters": {
                    "threshold": 0.5,
                    # Reduzido de 250ms para 100ms. O português é denso em interjeições vitais ("é", "tá", "já").
                    # 250ms cortaria o contexto sintático dessas palavras curtas.
                    "min_speech_duration_ms": 100, 
                    "min_silence_duration_ms": 1000,
                }
            }

            # Injeção dinâmica de parâmetros dependendo do pipeline
            if self.use_batched_pipeline:
                # O processamento em lotes brilha na GPU
                segments, info = self.pipeline.transcribe(audio_filepath, batch_size=16, **decode_params)
            else:
                # A CPU brilha no processamento sequencial usando paralelismo de baixo nível (threads)
                segments, info = self.pipeline.transcribe(audio_filepath, **decode_params)

            # A inferência do faster-whisper é um Generator (Lazy Loading).
            # A execução matricial só acontece quando iteramos sobre 'segments'.
            full_text = []
            for segment in segments:
                full_text.append(segment.text)

            elapsed = time.time() - start_time
            
            # Análise Forense do Desempenho
            duration = info.duration or 0
            duration_after_vad = info.duration_after_vad or duration
            silence_removed = duration - duration_after_vad

            print(f"✅ Decodificação finalizada em {elapsed:.2f}s")
            print(f"   Espectro total analisado: {duration:.2f}s")
            print(f"   Espectro útil (após VAD): {duration_after_vad:.2f}s (Silêncio podado: {silence_removed:.2f}s)")
            
            if duration > 0 and elapsed > 0:
                print(f"   Coeficiente de Aceleração: {duration / elapsed:.2f}x tempo real")

            transcription = " ".join(full_text)

            if not transcription.strip():
                return "Aviso: Nenhuma densidade vocal inteligível foi extraída pelos tensores do VAD."

            return transcription

        except Exception as e:
            # Fallback limpo em caso de áudio corrompido ou erro de decodificação de FFmpeg
            print(f"Erro fatal na topologia de transcrição: {e}")
            return f"Erro de processamento: {e}"