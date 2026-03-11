import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
import os
import queue

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        self._thread = None
        self.q = queue.Queue()
        
        # Silence detection parameters
        self.silence_threshold = 0.01  # Volume threshold
        self.silence_duration_limit = 300  # 5 minutes in seconds
        self.current_silence_duration = 0
        self.stop_reason = ""

    def _audio_callback(self, indata, frames, time_info, status):
        """This is called for each audio block by sounddevice."""
        if status:
            print(status)
        self.q.put(indata.copy())

    def _record_loop(self):
        """Main recording loop running in a separate thread."""
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback):
                while self.is_recording:
                    try:
                        data = self.q.get(timeout=0.5)
                        self.audio_data.append(data)
                        
                        # Calculate RMS for volume level
                        rms = np.sqrt(np.mean(np.square(data)))
                        
                        if rms < self.silence_threshold:
                            # Assuming chunk size represents about 1 chunk per get, we add elapsed time
                            # Usually sounddevice blocksize is around 1024 or based on default buffering
                            # It's cleaner to just measure real time since silence started
                            if not hasattr(self, '_silence_start_time') or self._silence_start_time is None:
                                self._silence_start_time = time.time()
                            else:
                                elapsed_silence = time.time() - self._silence_start_time
                                if elapsed_silence >= self.silence_duration_limit:
                                    self.stop_reason = "5 minutes of silence detected"
                                    self.is_recording = False
                        else:
                            self._silence_start_time = None
                            
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Error accessing microphone: {e}")
            self.stop_reason = f"Error: {e}"
            self.is_recording = False

    def start_recording(self):
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_data = []
        self.current_silence_duration = 0
        self.stop_reason = ""
        self._silence_start_time = None
        
        self._thread = threading.Thread(target=self._record_loop)
        self._thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self._thread:
            self._thread.join()

    def save_audio(self, filename="recorded_lecture.wav"):
        if not self.audio_data:
            return None
        
        audio_np = np.concatenate(self.audio_data, axis=0)
        wav.write(filename, self.sample_rate, audio_np)
        return filename
