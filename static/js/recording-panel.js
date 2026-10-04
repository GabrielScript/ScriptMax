// Console de gravação: botão REC, pausa, VU-meter, timer e revisão antes de enviar.

import { byId, el, formatClock } from './dom.js';
import { AudioRecorder, SOURCE, extensionFor, recordingSupport } from './recorder.js';

const VU_SEGMENTS = 24;
const WARM_FROM = 16;
const HOT_FROM = 21;
const SILENCE_WARNING_MS = 6000;
const SIGNAL_THRESHOLD = 0.02;
const SOURCE_HINTS = {
  [SOURCE.MIC]: 'Capta sua voz ou a sala (aulas presenciais).',
  [SOURCE.SYSTEM]: 'Chrome/Edge: escolha a aba ou a tela e marque “Compartilhar áudio”. Ideal para filmes e vídeos.',
  [SOURCE.BOTH]: 'Sua voz + o som do PC. Ideal para reuniões online (Meet, Teams, Zoom). Use fone para evitar eco.',
};

export class RecordingPanel {
  #recorder;
  #onSubmit;
  #take = null;
  #tickTimer = 0;
  #lastSignalAt = 0;
  #vuSegments = [];
  #elements = {
    start: byId('record-start'),
    startLabel: byId('record-start-label'),
    pause: byId('record-pause'),
    timer: byId('record-timer'),
    state: byId('record-state'),
    vu: byId('vu'),
    sourceHint: byId('source-hint'),
    sourceFieldset: byId('source-fieldset'),
    review: byId('recording-review'),
    preview: byId('recording-preview'),
    download: byId('recording-download'),
    submit: byId('recording-submit'),
    discard: byId('recording-discard'),
  };

  constructor({ onSubmit }) {
    this.#onSubmit = onSubmit;
    this.#recorder = new AudioRecorder({
      onLevel: (level) => this.#showLevel(level),
      onSourceEnded: () => this.#stop('A fonte de áudio foi encerrada. Revise a gravação.'),
    });
    this.#buildVu();
    this.#bindEvents();
    this.#applySupport();
  }

  get hasUnsavedAudio() {
    return this.#recorder.state !== 'inactive' || this.#take !== null;
  }

  #buildVu() {
    this.#vuSegments = Array.from({ length: VU_SEGMENTS }, (_, index) => {
      const className = index >= HOT_FROM ? 'hot' : index >= WARM_FROM ? 'warm' : '';
      return el('span', { className });
    });
    this.#elements.vu.replaceChildren(...this.#vuSegments);
  }

  #bindEvents() {
    const { start, pause, submit, discard, sourceFieldset } = this.#elements;
    start.addEventListener('click', () => {
      if (this.#recorder.state === 'inactive') this.#start();
      else this.#stop('Gravação encerrada. Ouça, baixe ou processe.');
    });
    pause.addEventListener('click', () => this.#togglePause());
    submit.addEventListener('click', () => this.#submit());
    discard.addEventListener('click', () => this.#discard());
    sourceFieldset.addEventListener('change', () => {
      this.#elements.sourceHint.textContent = SOURCE_HINTS[this.#selectedSource()];
    });
  }

  #applySupport() {
    const support = recordingSupport();
    if (!support.secure) {
      this.#disableRecording('Gravação exige HTTPS ou localhost. Use o link https do túnel (ngrok) no celular.');
      return;
    }
    if (!support.microphone) {
      this.#disableRecording('Este navegador não grava áudio. Use Chrome, Edge, Firefox ou Safari atualizados.');
      return;
    }
    if (!support.systemAudio) {
      for (const id of ['source-system', 'source-both']) byId(id).disabled = true;
      this.#elements.sourceHint.textContent = `${SOURCE_HINTS[SOURCE.MIC]} (Áudio do PC só no Chrome/Edge de computador.)`;
    }
  }

  #disableRecording(message) {
    this.#elements.start.disabled = true;
    this.#elements.state.textContent = message;
  }

  #selectedSource() {
    return document.querySelector('input[name="audio-source"]:checked')?.value ?? SOURCE.MIC;
  }

  async #start() {
    if (this.#take && !window.confirm('Descartar a gravação anterior e começar outra?')) return;
    this.#discard();
    this.#elements.start.disabled = true;
    this.#elements.state.textContent = 'Pedindo permissão…';
    try {
      await this.#recorder.start(this.#selectedSource());
    } catch (error) {
      this.#elements.state.textContent = error.message;
      this.#elements.start.disabled = false;
      return;
    }
    this.#lastSignalAt = performance.now();
    this.#setMode('recording');
    this.#elements.state.textContent = 'Gravando.';
    this.#tickTimer = window.setInterval(() => this.#tick(), 250);
  }

  #togglePause() {
    if (this.#recorder.state === 'recording') {
      this.#recorder.pause();
      this.#setMode('paused');
      this.#elements.state.textContent = 'Pausado.';
    } else if (this.#recorder.state === 'paused') {
      this.#recorder.resume();
      this.#lastSignalAt = performance.now();
      this.#setMode('recording');
      this.#elements.state.textContent = 'Gravando.';
    }
  }

  async #stop(message) {
    if (this.#recorder.state === 'inactive') return;
    window.clearInterval(this.#tickTimer);
    const result = await this.#recorder.stop();
    this.#setMode('idle');
    if (!result || result.blob.size === 0) {
      this.#elements.state.textContent = 'Nada foi gravado.';
      return;
    }
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
    this.#take = {
      blob: result.blob,
      filename: `gravacao-${stamp}.${extensionFor(result.blob.type)}`,
      url: URL.createObjectURL(result.blob),
    };
    this.#elements.timer.textContent = formatClock(result.durationMs);
    this.#elements.preview.src = this.#take.url;
    this.#elements.download.href = this.#take.url;
    this.#elements.download.download = this.#take.filename;
    this.#elements.review.hidden = false;
    this.#elements.state.textContent = message;
    this.#elements.submit.focus();
  }

  async #submit() {
    if (!this.#take) return;
    this.#elements.submit.disabled = true;
    try {
      const accepted = await this.#onSubmit(this.#take.blob, this.#take.filename);
      if (accepted) this.#discard();
    } finally {
      this.#elements.submit.disabled = false;
    }
  }

  #discard() {
    if (this.#take) URL.revokeObjectURL(this.#take.url);
    this.#take = null;
    this.#elements.review.hidden = true;
    this.#elements.preview.removeAttribute('src');
    this.#elements.timer.textContent = formatClock(0);
  }

  #tick() {
    this.#elements.timer.textContent = formatClock(this.#recorder.elapsedMs);
    const silentFor = performance.now() - this.#lastSignalAt;
    if (this.#recorder.state === 'recording' && silentFor > SILENCE_WARNING_MS) {
      this.#elements.state.textContent = 'Nenhum som detectado — confira a fonte ou o volume.';
    } else if (this.#recorder.state === 'recording') {
      this.#elements.state.textContent = 'Gravando.';
    }
  }

  #showLevel(level) {
    if (level > SIGNAL_THRESHOLD) this.#lastSignalAt = performance.now();
    const lit = Math.round(level * VU_SEGMENTS);
    this.#vuSegments.forEach((segment, index) => segment.classList.toggle('on', index < lit));
  }

  #setMode(mode) {
    const live = mode !== 'idle';
    document.body.classList.toggle('recording', mode === 'recording');
    document.body.classList.toggle('paused', mode === 'paused');
    const { start, startLabel, pause, sourceFieldset } = this.#elements;
    start.disabled = false;
    start.classList.toggle('is-live', live);
    startLabel.textContent = live ? 'Parar' : 'Gravar';
    start.setAttribute('aria-label', live ? 'Parar gravação e revisar' : 'Iniciar gravação');
    pause.hidden = !live;
    pause.textContent = mode === 'paused' ? 'Continuar' : 'Pausar';
    sourceFieldset.disabled = live;
  }
}
