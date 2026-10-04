// Biblioteca em árvore: gaveta por categoria -> pastas -> relatórios.
// Usa <details>/<summary> nativos: teclado e leitor de tela funcionam sem ARIA extra.

import { reportFileUrl } from './api.js';
import { el, formatDate, formatDuration } from './dom.js';

function emptyNode(name, path) {
  return { name, path, folders: new Map(), reports: [] };
}

function countReports(node) {
  let total = node.reports.length;
  for (const child of node.folders.values()) total += countReports(child);
  return total;
}

function matchesFilter(report, filter) {
  if (!filter) return true;
  return `${report.subject} ${report.folder} ${report.source_name}`.toLocaleLowerCase('pt-BR').includes(filter);
}

export function foldersOf(reports, categoryId) {
  const folders = new Set();
  for (const report of reports) {
    if (report.category !== categoryId || !report.folder) continue;
    const segments = report.folder.split('/');
    segments.forEach((_, index) => folders.add(segments.slice(0, index + 1).join('/')));
  }
  return [...folders].sort((a, b) => a.localeCompare(b, 'pt-BR'));
}

export class Library {
  #container;
  #categories;
  #handlers;
  #reports = [];
  #filter = '';
  #openKeys = new Set();
  #knownKeys = new Set();

  constructor({ container, categories, handlers }) {
    this.#container = container;
    this.#categories = categories;
    this.#handlers = handlers;
  }

  get reports() {
    return this.#reports;
  }

  setReports(reports) {
    this.#reports = reports;
    this.#render();
  }

  setFilter(text) {
    this.#filter = text.trim().toLocaleLowerCase('pt-BR');
    this.#render();
  }

  #render() {
    const visible = this.#reports.filter((report) => matchesFilter(report, this.#filter));
    const drawers = this.#categories
      .map((category) => this.#renderDrawer(category, visible.filter((report) => report.category === category.id)))
      .filter(Boolean);
    this.#container.replaceChildren(...drawers);
  }

  #buildTree(reports) {
    const root = emptyNode('', '');
    for (const report of reports) {
      let node = root;
      for (const segment of (report.folder || '').split('/').filter(Boolean)) {
        const path = node.path ? `${node.path}/${segment}` : segment;
        if (!node.folders.has(segment)) node.folders.set(segment, emptyNode(segment, path));
        node = node.folders.get(segment);
      }
      node.reports.push(report);
    }
    return root;
  }

  #renderDrawer(category, reports) {
    if (reports.length === 0) return null;
    const tree = this.#buildTree(reports);
    return this.#details({
      key: category.id,
      className: `drawer cat-${category.id}`,
      label: category.label,
      count: reports.length,
      openByDefault: true,
      content: this.#renderContents(tree, category.id),
    });
  }

  #renderContents(node, categoryId) {
    const folders = [...node.folders.values()]
      .sort((a, b) => a.name.localeCompare(b.name, 'pt-BR'))
      .map((child) => this.#details({
        key: `${categoryId}/${child.path}`,
        className: 'folder',
        label: `📁 ${child.name}`,
        count: countReports(child),
        openByDefault: Boolean(this.#filter),
        content: this.#renderContents(child, categoryId),
      }));
    const reports = node.reports.length
      ? [el('ul', { className: 'report-list' }, node.reports.map((report) => this.#renderReport(report)))]
      : [];
    return [...reports, ...folders];
  }

  #details({ key, className, label, count, openByDefault, content }) {
    if (!this.#knownKeys.has(key)) {
      this.#knownKeys.add(key);
      if (openByDefault) this.#openKeys.add(key);
    }
    const details = el('details', { className, open: this.#openKeys.has(key) || Boolean(this.#filter) },
      el('summary', {}, el('span', { text: label }), el('span', { className: 'count', text: String(count) })),
      content);
    details.addEventListener('toggle', () => {
      if (details.open) this.#openKeys.add(key);
      else this.#openKeys.delete(key);
    });
    return details;
  }

  #renderReport(report) {
    const metaParts = [formatDate(report.created_at), formatDuration(report.duration_seconds), report.source_name].filter(Boolean);
    const flags = [];
    if (!report.report_ready) flags.push('Só transcrição');
    else if (!report.complete) flags.push('Parcial');
    if (report.report_ready && !report.math_rendered) flags.push('Fórmulas sem render');

    return el('li', { className: 'report' },
      el('p', { className: 'report-title', text: report.subject }),
      el('p', { className: 'report-meta' }, metaParts.map((part) => el('span', { text: part }))),
      flags.length ? el('div', { className: 'report-flags' }, flags.map((flag) => el('span', { className: 'badge', text: flag }))) : null,
      el('div', { className: 'report-actions' }, this.#actionsFor(report)));
  }

  #actionsFor(report) {
    const link = (kind, text, options) => el('a', {
      href: reportFileUrl(report.id, kind, options), target: options?.download ? null : '_blank', rel: 'noopener', text,
    });
    const button = (text, handler, className) => el('button', {
      type: 'button', className, text, 'aria-label': `${text}: ${report.subject}`, onClick: () => handler(report),
    });
    const actions = [];
    if (report.report_ready) {
      actions.push(link('html', 'Ler'), link('pdf', 'PDF'), link('pdf', 'Baixar PDF', { download: true }));
    }
    actions.push(link('transcript', 'Transcrição'));
    actions.push(button(report.report_ready ? 'Regerar' : 'Gerar relatório', this.#handlers.onRegenerate));
    actions.push(button('Mover', this.#handlers.onMove));
    actions.push(button('Excluir', this.#handlers.onDelete, 'danger-link'));
    return actions;
  }
}
