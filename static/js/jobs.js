// Painel "Na mesa de edição": acompanha os processamentos por polling.
// Cada job tem um nó DOM estável (atualizado no lugar), então o foco do
// teclado não se perde a cada atualização.

import { api, reportFileUrl } from './api.js';
import { el } from './dom.js';

const POLL_INTERVAL_MS = 2000;
const ACTIVE_STAGES = new Set(['queued', 'transcribing', 'summarizing', 'rendering', 'emailing']);
const STAGE_LABELS = {
  queued: 'Na fila',
  transcribing: 'Transcrevendo',
  summarizing: 'Escrevendo',
  rendering: 'Diagramando',
  emailing: 'Enviando',
  done: 'Pronto',
  failed: 'Falhou',
};

function badgeClass(stage) {
  if (stage === 'done') return 'badge done';
  if (stage === 'failed') return 'badge failed';
  if (stage === 'queued') return 'badge queued';
  return 'badge running';
}

export class JobBoard {
  #list;
  #announcer;
  #onFinished;
  #onError;
  #nodes = new Map();
  #lastStages = new Map();
  #timer = 0;

  constructor({ listElement, announcer, onFinished, onError }) {
    this.#list = listElement;
    this.#announcer = announcer;
    this.#onFinished = onFinished;
    this.#onError = onError;
  }

  async refresh() {
    let jobs;
    try {
      jobs = await api.jobs();
    } catch (error) {
      this.#onError(error);
      this.#schedule(true);
      return;
    }
    this.#render(jobs);
    this.#schedule(jobs.some((job) => ACTIVE_STAGES.has(job.stage)));
  }

  track(job) {
    this.#render([job, ...this.#currentJobs().filter((existing) => existing.id !== job.id)]);
    this.#schedule(true);
  }

  #currentJobs() {
    return [...this.#nodes.values()].map((entry) => entry.job);
  }

  #schedule(active) {
    window.clearTimeout(this.#timer);
    if (active) this.#timer = window.setTimeout(() => this.refresh(), POLL_INTERVAL_MS);
  }

  #render(jobs) {
    let finishedSomething = false;
    for (const job of jobs) {
      const previous = this.#lastStages.get(job.id);
      if (previous && previous !== job.stage) {
        this.#announcer.textContent = `${job.title}: ${STAGE_LABELS[job.stage] ?? job.stage}.`;
        if (job.stage === 'done' || job.stage === 'failed') finishedSomething = true;
      }
      this.#lastStages.set(job.id, job.stage);
      this.#upsert(job);
    }
    const order = jobs.map((job) => this.#nodes.get(job.id).node);
    order.forEach((node, index) => {
      if (this.#list.children[index] !== node) this.#list.insertBefore(node, this.#list.children[index] ?? null);
    });
    if (finishedSomething) this.#onFinished();
  }

  #upsert(job) {
    let entry = this.#nodes.get(job.id);
    if (!entry) {
      entry = this.#build();
      this.#nodes.set(job.id, entry);
    }
    entry.job = job;
    const { parts } = entry;
    parts.title.textContent = job.title;
    parts.badge.textContent = STAGE_LABELS[job.stage] ?? job.stage;
    parts.badge.className = badgeClass(job.stage);
    parts.message.textContent = job.message;
    parts.progress.value = job.progress;
    parts.progress.hidden = !ACTIVE_STAGES.has(job.stage) || job.stage === 'queued';
    parts.error.textContent = job.error ?? '';
    parts.error.hidden = !job.error;
    parts.actions.replaceChildren(...this.#actionsFor(job));
  }

  #build() {
    const parts = {
      title: el('span', { className: 'job-title' }),
      badge: el('span', { className: 'badge' }),
      message: el('p', { className: 'job-message' }),
      progress: el('progress', { max: '1', value: '0', 'aria-label': 'Progresso da etapa' }),
      error: el('p', { className: 'job-error' }),
      actions: el('div', { className: 'report-actions' }),
    };
    const node = el('li', { className: 'job' },
      el('div', { className: 'job-head' }, parts.title, parts.badge),
      parts.message, parts.progress, parts.error, parts.actions);
    return { node, parts, job: null };
  }

  #actionsFor(job) {
    const actions = [];
    if (job.stage === 'done' && job.report_id) {
      actions.push(el('a', { href: reportFileUrl(job.report_id, 'html'), target: '_blank', rel: 'noopener', text: 'Ler relatório' }));
      actions.push(el('a', { href: reportFileUrl(job.report_id, 'pdf'), target: '_blank', rel: 'noopener', text: 'PDF' }));
    }
    if (job.stage === 'failed' && job.retryable) {
      actions.push(el('button', { type: 'button', text: 'Tentar de novo', onClick: () => this.#retry(job.id) }));
    }
    return actions;
  }

  async #retry(jobId) {
    try {
      this.track(await api.retryJob(jobId));
    } catch (error) {
      this.#onError(error);
    }
  }
}
