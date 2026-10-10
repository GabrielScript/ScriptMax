// Gravação no navegador: microfone, áudio do PC (aba/tela) ou os dois mixados.
// O áudio do PC usa getDisplayMedia: Chrome/Edge exigem "video: true" e o usuário
// precisa marcar "Compartilhar áudio" na janela de escolha.

import { RecordingBackup } from './recording-backup.js';

export const SOURCE = Object.freeze({ MIC: 'mic', SYSTEM: 'system', BOTH: 'both' });

const MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm'];
const BITRATE = 64_000; // Opus 64 kbps: fala e trilha de filme com folga (~29 MB/h)
export const TIMESLICE_MS = 1000;
const LEVEL_GAIN = 4;

export function recordingSupport() {
  const devices = navigator.mediaDevices;
  return {
    secure: window.isSecureContext,
    microphone: Boolean(devices?.getUserMedia) && typeof MediaRecorder !== 'undefined',
    systemAudio: Boolean(devices?.getDisplayMedia) && typeof MediaRecorder !== 'undefined',
  };
}

export function extensionFor(mimeType) {
  if (mimeType.includes('ogg')) return 'ogg';
  if (mimeType.includes('mp4')) return 'mp4';
  return 'webm';
}

function pickMimeType() {
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}

function stopStream(stream) {
  stream.getTracks().forEach((track) => track.stop());
}

function explainCaptureError(error) {
  const messages = {
    NotAllowedError: 'Permissão negada. Libere o microfone/compartilhamento no cadeado da barra de endereço.',
    NotFoundError: 'Nenhum microfone encontrado neste dispositivo.',
    NotReadableError: 'O microfone está em uso por outro programa.',
    AbortError: 'A captura foi cancelada.',
  };
  if (error?.name in messages) return new Error(messages[error.name]);
  return error instanceof Error ? error : new Error(String(error));
}

async function captureSystemAudio() {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    systemAudio: 'include',
    selfBrowserSurface: 'exclude',
  });
  if (stream.getAudioTracks().length === 0) {
    stopStream(stream);
    throw new Error('Nenhum áudio foi compartilhado. Ao escolher a aba ou a tela, marque “Compartilhar áudio”.');
  }
  return stream;
}

function captureMicrophone(source) {
  return navigator.mediaDevices.getUserMedia({
    // Com o áudio do PC junto, o cancelamento de eco evita gravar o alto-falante duas vezes.
    audio: { echoCancellation: source === SOURCE.BOTH, noiseSuppression: true, autoGainControl: true },
  });
}

async function acquireStreams(source) {
  const streams = [];
  try {
    // A captura de tela precisa vir primeiro: exige o gesto do clique ainda "fresco".
    if (source === SOURCE.SYSTEM || source === SOURCE.BOTH) streams.push(await captureSystemAudio());
    if (source === SOURCE.MIC || source === SOURCE.BOTH) streams.push(await captureMicrophone(source));
  } catch (error) {
    streams.forEach(stopStream);
    throw explainCaptureError(error);
  }
  return streams;
}

export class AudioRecorder {
  #streams = [];
  #context = null;
  #analyser = null;
  #recorder = null;
  #chunks = [];
  #startedAt = 0;
  #accumulatedMs = 0;
  #levelFrame = 0;
  #wakeLock = null;
  #backup = null;
  #onLevel;
  #onSourceEnded;
  #onVisibilityChange = () => this.#reacquireWakeLock();

  constructor({ onLevel, onSourceEnded }) {
    this.#onLevel = onLevel;
    this.#onSourceEnded = onSourceEnded;
  }

  get state() {
    return this.#recorder?.state ?? 'inactive';
  }

  get elapsedMs() {
    if (this.state === 'recording') return this.#accumulatedMs + (performance.now() - this.#startedAt);
    return this.#accumulatedMs;
  }

  async start(source) {
    if (this.state !== 'inactive') throw new Error('Já existe uma gravação em andamento.');
    this.#streams = await acquireStreams(source);
    try {
      const mixedStream = await this.#buildMixer();
      const mimeType = pickMimeType();
      const options = mimeType ? { mimeType, audioBitsPerSecond: BITRATE } : { audioBitsPerSecond: BITRATE };
      this.#recorder = new MediaRecorder(mixedStream, options);
    } catch (error) {
      this.#release();
      throw explainCaptureError(error);
    }
    this.#chunks = [];
    const backup = new RecordingBackup();
    this.#backup = backup;
    this.#recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size === 0) return;
      this.#chunks.push(event.data);
      backup.append(event.data, this.elapsedMs);
    });
    this.#recorder.start(TIMESLICE_MS);
    this.#accumulatedMs = 0;
    this.#startedAt = performance.now();
    this.#watchLevel();
    await this.#keepScreenAwake();
  }

  pause() {
    if (this.state !== 'recording') return;
    this.#recorder.pause();
    this.#accumulatedMs += performance.now() - this.#startedAt;
  }

  resume() {
    if (this.state !== 'paused') return;
    this.#recorder.resume();
    this.#startedAt = performance.now();
  }

  async stop() {
    const recorder = this.#recorder;
    if (!recorder) return null;
    const durationMs = this.elapsedMs;
    if (recorder.state !== 'inactive') {
      const stopped = new Promise((resolve) => recorder.addEventListener('stop', resolve, { once: true }));
      recorder.stop();
      await stopped;
    }
    const blob = new Blob(this.#chunks, { type: recorder.mimeType || 'audio/webm' });
    const backup = this.#backup;
    this.#backup = null;
    this.#chunks = [];
    this.#release();
    // A cópia de segurança fica até a gravação ser enviada ou descartada (quem decide é o painel).
    return { blob, durationMs, backup };
  }

  cancel() {
    if (this.#recorder && this.#recorder.state !== 'inactive') this.#recorder.stop();
    this.#backup?.discard();
    this.#backup = null;
    this.#chunks = [];
    this.#release();
  }

  async #buildMixer() {
    this.#context = new AudioContext();
    await this.#context.resume();
    const mixer = this.#context.createGain();
    const destination = this.#context.createMediaStreamDestination();
    this.#analyser = this.#context.createAnalyser();
    this.#analyser.fftSize = 1024;
    for (const stream of this.#streams) {
      for (const track of stream.getAudioTracks()) {
        track.addEventListener('ended', () => this.#onSourceEnded?.(), { once: true });
      }
      this.#context.createMediaStreamSource(stream).connect(mixer);
    }
    mixer.connect(destination);
    mixer.connect(this.#analyser);
    return destination.stream;
  }

  #watchLevel() {
    const samples = new Float32Array(this.#analyser.fftSize);
    const tick = () => {
      if (!this.#analyser) return;
      this.#analyser.getFloatTimeDomainData(samples);
      let sumOfSquares = 0;
      for (const sample of samples) sumOfSquares += sample * sample;
      const level = this.state === 'recording' ? Math.min(1, Math.sqrt(sumOfSquares / samples.length) * LEVEL_GAIN) : 0;
      this.#onLevel?.(level);
      this.#levelFrame = requestAnimationFrame(tick);
    };
    tick();
  }

  async #keepScreenAwake() {
    // Gravação longa no celular: impede a tela de apagar (e o navegador de pausar a aba).
    document.addEventListener('visibilitychange', this.#onVisibilityChange);
    await this.#reacquireWakeLock();
  }

  async #reacquireWakeLock() {
    if (!('wakeLock' in navigator) || document.visibilityState !== 'visible' || this.state === 'inactive') return;
    try {
      this.#wakeLock = await navigator.wakeLock.request('screen');
    } catch {
      this.#wakeLock = null; // bateria fraca ou política do navegador: segue sem wake lock
    }
  }

  #release() {
    cancelAnimationFrame(this.#levelFrame);
    document.removeEventListener('visibilitychange', this.#onVisibilityChange);
    this.#wakeLock?.release().catch(() => {});
    this.#wakeLock = null;
    this.#streams.forEach(stopStream);
    this.#streams = [];
    this.#context?.close().catch(() => {});
    this.#context = null;
    this.#analyser = null;
    this.#recorder = null;
    this.#onLevel?.(0);
  }
}
