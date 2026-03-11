import os
import time
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfilt
from faster_whisper import decode_audio


class AudioEnhancer:
    """
    Pipeline de melhoramento de áudio para transcrição:
    1. Carrega o áudio (MP3, WAV, M4A, OGG — via ffmpeg)
    2. Converte para mono 16kHz
    3. Aplica filtro passa-banda (80Hz–7500Hz) — remove ruído grave e agudo
    4. Aplica redução de ruído espectral (noisereduce)
    5. Normaliza volume
    6. Salva áudio melhorado como WAV
    """

    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def enhance(self, audio_filepath, output_path=None):
        """
        Recebe um arquivo de áudio ruidoso e retorna o caminho de um arquivo limpo.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_filepath}")

        if output_path is None:
            base, ext = os.path.splitext(audio_filepath)
            output_path = f"{base}_enhanced.wav"

        try:
            print(f"🔧 Iniciando melhoramento do áudio: {audio_filepath}")
            start_time = time.time()

            # --- 1. Carregar áudio usando ffmpeg (suporta MP3, M4A, OGG, WAV) ---
            audio = decode_audio(audio_filepath, sampling_rate=self.target_sr)
            sr = self.target_sr
            original_duration = len(audio) / sr
            print(f"   Áudio carregado: {original_duration:.1f}s, {sr}Hz, mono")

            # --- 2. Converter para mono se necessário ---
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                print("   Convertido para mono")

            # --- 3. Resample para 16kHz se necessário ---
            if sr != self.target_sr:
                audio = self._resample(audio, sr, self.target_sr)
                sr = self.target_sr
                print(f"   Reamostrado para {sr}Hz")

            # --- 4. Filtro passa-banda (80Hz – 7500Hz) ---
            # Remove ruído de baixa frequência (ar condicionado, tráfego)
            # e ruído de alta frequência (chiado, estática)
            audio = self._bandpass_filter(audio, sr, lowcut=80, highcut=7500)
            print("   Filtro passa-banda aplicado (80Hz–7500Hz)")

            # --- 5. Redução de ruído espectral (noisereduce) ---
            # Modo não-estacionário: melhor para ruído que varia (sala de aula, auditório)
            audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,       # Ruído que varia ao longo do tempo
                prop_decrease=0.75,     # 75% de redução (equilíbrio entre limpeza e naturalidade)
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            )
            print("   Redução de ruído espectral aplicada")

            # --- 6. Normalização de pico (-3dB) ---
            audio = self._normalize(audio, target_db=-3.0)
            print("   Volume normalizado (-3dB)")

            # --- 7. Salvar áudio melhorado ---
            sf.write(output_path, audio, sr)

            elapsed = time.time() - start_time
            print(f"✅ Áudio melhorado em {elapsed:.1f}s → {output_path}")
            print(f"   Duração: {original_duration:.1f}s | Processamento: {elapsed:.1f}s")

            return output_path

        except Exception as e:
            print(f"⚠️ Erro no melhoramento do áudio: {e}")
            print("   Continuando com áudio original...")
            return audio_filepath  # Fallback: retorna o áudio original

    def _bandpass_filter(self, audio, sr, lowcut=80, highcut=7500, order=5):
        """Filtro Butterworth passa-banda para manter apenas frequências de fala."""
        nyquist = sr / 2.0
        low = lowcut / nyquist
        high = min(highcut / nyquist, 0.99)  # Não pode exceder Nyquist
        sos = butter(order, [low, high], btype='bandpass', output='sos')
        return sosfilt(sos, audio).astype(np.float32)

    def _normalize(self, audio, target_db=-3.0):
        """Normalização de pico: ajusta o volume máximo para target_db."""
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        target_amplitude = 10 ** (target_db / 20.0)
        return (audio * (target_amplitude / peak)).astype(np.float32)

    def _resample(self, audio, orig_sr, target_sr):
        """Resample simples usando interpolação linear."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
