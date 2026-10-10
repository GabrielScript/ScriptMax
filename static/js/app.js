// Ponto de entrada do frontend: login, formulário, envio e ligação entre os painéis.

import { api, uploadAudio } from './api.js';
import { byId, el, formatBytes, storage } from './dom.js';
import { JobBoard } from './jobs.js';
import { Library, folderCounts } from './library.js';
import { RecordingPanel } from './recording-panel.js';

const LAST_CATEGORY_KEY = 'scriptmax.category';
const LAST_FOLDER_KEY = 'scriptmax.folder';
const CATEGORY_BLURBS = {
  'desenvolvimento-pessoal': 'Podcasts, palestras e conteúdos de psicologia',
  'filmes-series': 'Filmes, episódios e documentários',
  academico: 'Aulas e cursos — fórmulas em LaTeX automático',
  trabalho: 'Reuniões, calls e treinamentos',
  tech: 'Stacks, ferramentas, arquitetura e tutoriais',
};

const state = { config: null, files: [], uploading: false };
let library;
let jobBoard;
let recordingPanel;

// ---------- Boot e login ----------

async function boot() {
  const bootStatus = byId('boot-status');
  try {
    state.config = await api.config();
  } catch (error) {
    bootStatus.textContent = `Não foi possível conectar ao servidor: ${error.message}`;
    return;
  }
  bootStatus.hidden = true;
  if (state.config.token_required && !state.config.authenticated) showLogin();
  else startApp();
}

function showLogin() {
  const section = byId('login-section');
  const form = byId('login-form');
  const input = byId('login-token');
  const error = byId('login-error');
  section.hidden = false;
  input.focus();
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.textContent = '';
    try {
      await api.login(input.value);
    } catch (loginError) {
      error.textContent = loginError.message;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      return;
    }
    input.value = '';
    section.hidden = true;
    state.config = await api.config();
    startApp();
  });
}

// ---------- Aplicação ----------

function startApp() {
  const { config } = state;
  byId('app').hidden = false;
  renderCategories(config.categories);
  byId('max-upload').textContent = String(config.max_upload_mb);
  byId('library-dir').textContent = config.library_dir;
  byId('open-library').hidden = !config.can_open_library;
  byId('logout').hidden = !config.token_required;
  byId('folder').value = storage.get(LAST_FOLDER_KEY) ?? '';

  library = new Library({
    container: byId('library-tree'),
    categories: config.categories,
    handlers: { onRegenerate: regenerateReport, onMove: openMoveDialog, onDelete: deleteReport },
  });
  jobBoard = new JobBoard({
    listElement: byId('job-list'),
    announcer: byId('job-announcer'),
    onFinished: refreshLibrary,
    onError: (error) => setStatus('library-status', error.message),
  });
  recordingPanel = new RecordingPanel({ onSubmit: (blob, filename) => submitAudio([{ file: blob, filename }]) });

  bindFormEvents();
  bindUploadEvents();
  bindLibraryEvents();
  refreshLibrary();
  jobBoard.refresh();
}

function renderCategories(categories) {
  const saved = storage.get(LAST_CATEGORY_KEY);
  const cards = categories.map((category, index) => {
    const id = `category-${category.id}`;
    const input = el('input', {
      type: 'radio', name: 'category', id, value: category.id, required: index === 0,
      checked: saved ? saved === category.id : index === 0,
    });
    const label = el('label', { for: id },
      el('strong', { text: category.label }),
      el('span', { text: CATEGORY_BLURBS[category.id] ?? '' }));
    return el('div', { className: `category-card cat-${category.id}` }, input, label);
  });
  byId('category-options').replaceChildren(...cards);
  byId('move-category').replaceChildren(
    ...categories.map((category) => el('option', { value: category.id, text: category.label })));
}

function selectedCategory() {
  return document.querySelector('input[name="category"]:checked')?.value ?? '';
}

function updateFolderSuggestions(categoryId) {
  const folders = [...folderCounts(library?.reports ?? [], categoryId).keys()];
  byId('folder-options').replaceChildren(...folders.map((folder) => el('option', { value: folder })));
}

// O título sugere as pastas da categoria: escolher uma continua aquela sequência.
function updateSubjectSuggestions() {
  const counts = folderCounts(library?.reports ?? [], selectedCategory());
  const describe = (total) => (total === 0 ? 'só subpastas' : `${total} ${total === 1 ? 'relatório' : 'relatórios'}`);
  const options = [...counts].map(([folder, total]) => el('option', { value: folder, label: describe(total) }));
  byId('subject-options').replaceChildren(...options);
}

function updateFormSuggestions() {
  updateFolderSuggestions(selectedCategory());
  updateSubjectSuggestions();
}

// Escolha no <datalist> chega como insertReplacementText (ou Event sem inputType em navegadores
// antigos); digitação comum é insertText e não dispara nada, mesmo que coincida com uma pasta.
function isSuggestionPick(event) {
  return event.inputType === undefined || event.inputType === 'insertReplacementText';
}

function onSubjectInput(event) {
  const input = event.target;
  input.removeAttribute('aria-invalid');
  if (!isSuggestionPick(event)) return;
  const folder = input.value;
  if (!folderCounts(library?.reports ?? [], selectedCategory()).has(folder)) return;
  byId('folder').value = folder;
  storage.set(LAST_FOLDER_KEY, folder);
  input.value = '';
  input.focus();
  byId('subject-announcer').textContent = `Pasta “${folder}” selecionada. Digite o título do novo relatório.`;
}

function bindFormEvents() {
  byId('category-options').addEventListener('change', () => {
    storage.set(LAST_CATEGORY_KEY, selectedCategory());
    updateFormSuggestions();
  });
  byId('folder').addEventListener('change', (event) => storage.set(LAST_FOLDER_KEY, event.target.value));
  byId('subject').addEventListener('input', onSubjectInput);
  byId('logout').addEventListener('click', async () => {
    await api.logout().catch(() => {});
    window.location.reload();
  });
  window.addEventListener('beforeunload', (event) => {
    if (state.uploading || recordingPanel?.hasUnsavedAudio) event.preventDefault();
  });
}

function readDetails() {
  const subjectInput = byId('subject');
  const error = byId('details-error');
  const subject = subjectInput.value.trim();
  const category = selectedCategory();
  error.textContent = '';
  if (!category) {
    error.textContent = 'Escolha uma categoria.';
    document.querySelector('input[name="category"]')?.focus();
    return null;
  }
  if (!subject) {
    error.textContent = 'Informe o título ou assunto antes de enviar.';
    subjectInput.setAttribute('aria-invalid', 'true');
    subjectInput.focus();
    return null;
  }
  return { subject, category, folder: byId('folder').value.trim() };
}

// ---------- Envio ----------

function setStatus(elementId, message) {
  byId(elementId).textContent = message;
}

async function submitAudio(items) {
  const details = readDetails();
  if (!details || state.uploading) return false;
  state.uploading = true;
  const progress = byId('upload-progress');
  progress.hidden = false;
  let allAccepted = true;
  try {
    for (const [index, item] of items.entries()) {
      const subject = items.length > 1 ? `${details.subject} — ${item.filename.replace(/\.[^.]+$/, '')}` : details.subject;
      setStatus('upload-status', `Enviando ${index + 1} de ${items.length}: ${item.filename}`);
      progress.value = 0;
      try {
        const job = await uploadAudio({
          fields: { subject, category: details.category, folder: details.folder },
          file: item.file,
          filename: item.filename,
          onProgress: (fraction) => { progress.value = fraction; },
        });
        jobBoard.track(job);
      } catch (error) {
        allAccepted = false;
        setStatus('upload-status', `Falha ao enviar ${item.filename}: ${error.message}`);
        return false;
      }
    }
    setStatus('upload-status', items.length > 1 ? `${items.length} arquivos na fila.` : 'Enviado. Acompanhe em “Na mesa de edição”.');
    return allAccepted;
  } finally {
    state.uploading = false;
    progress.hidden = true;
  }
}

function bindUploadEvents() {
  const input = byId('audio-files');
  const dropZone = byId('drop-zone');
  input.addEventListener('change', () => setSelectedFiles([...input.files]));
  for (const type of ['dragenter', 'dragover']) {
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      dropZone.classList.add('dragging');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    dropZone.addEventListener(type, () => dropZone.classList.remove('dragging'));
  }
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    setSelectedFiles([...event.dataTransfer.files]);
  });
  byId('upload-submit').addEventListener('click', async () => {
    const accepted = await submitAudio(state.files.map((file) => ({ file, filename: file.name })));
    if (accepted) {
      input.value = '';
      setSelectedFiles([]);
    }
  });
}

function setSelectedFiles(files) {
  const maxBytes = state.config.max_upload_mb * 1024 * 1024;
  const tooBig = files.filter((file) => file.size > maxBytes);
  state.files = files.filter((file) => file.size <= maxBytes && file.size > 0);
  byId('selected-files').replaceChildren(...state.files.map((file) =>
    el('li', {}, el('span', { text: file.name }), el('span', { className: 'size', text: formatBytes(file.size) }))));
  byId('upload-submit').disabled = state.files.length === 0;
  setStatus('upload-status', tooBig.length ? `Ignorados (acima de ${state.config.max_upload_mb} MB): ${tooBig.map((f) => f.name).join(', ')}` : '');
}

// ---------- Biblioteca ----------

async function refreshLibrary() {
  try {
    library.setReports(await api.reports());
    updateFormSuggestions();
  } catch (error) {
    setStatus('library-status', `Não foi possível carregar a biblioteca: ${error.message}`);
  }
}

function bindLibraryEvents() {
  byId('library-filter').addEventListener('input', (event) => library.setFilter(event.target.value));
  byId('open-library').addEventListener('click', () => api.openLibrary().catch((error) => setStatus('library-status', error.message)));
  byId('move-form').addEventListener('submit', onMoveSubmit);
  // O diálogo troca as sugestões de pasta para a categoria do relatório; ao fechar, voltam às do formulário.
  byId('move-dialog').addEventListener('close', () => updateFolderSuggestions(selectedCategory()));
}

async function regenerateReport(report) {
  try {
    jobBoard.track(await api.regenerateReport(report.id));
    setStatus('library-status', `Regerando “${report.subject}”.`);
  } catch (error) {
    setStatus('library-status', error.message);
    // 404: o relatório foi apagado em outra aba/aparelho; tira-o da lista.
    if (error.status === 404) refreshLibrary();
  }
}

let reportBeingMoved = null;

function openMoveDialog(report) {
  reportBeingMoved = report;
  byId('move-category').value = report.category;
  byId('move-folder').value = report.folder;
  byId('move-error').textContent = '';
  updateFolderSuggestions(report.category);
  byId('move-dialog').showModal();
}

async function onMoveSubmit(event) {
  event.preventDefault();
  const dialog = byId('move-dialog');
  if (event.submitter?.value !== 'confirm' || !reportBeingMoved) {
    dialog.close();
    return;
  }
  try {
    await api.moveReport(reportBeingMoved.id, byId('move-category').value, byId('move-folder').value.trim());
  } catch (error) {
    if (error.status === 404) {
      // Apagado em outra aba/aparelho: não há mais o que mover.
      dialog.close();
      setStatus('library-status', error.message);
      reportBeingMoved = null;
      refreshLibrary();
      return;
    }
    byId('move-error').textContent = error.message;
    return;
  }
  dialog.close();
  setStatus('library-status', `“${reportBeingMoved.subject}” movido.`);
  reportBeingMoved = null;
  refreshLibrary();
}

function confirmDeletion(report) {
  const dialog = byId('confirm-dialog');
  byId('confirm-title').textContent = `Excluir “${report.subject}”?`;
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true });
  });
}

async function deleteReport(report) {
  if (!(await confirmDeletion(report))) return;
  try {
    await api.deleteReport(report.id);
    setStatus('library-status', `“${report.subject}” excluído.`);
  } catch (error) {
    setStatus('library-status', error.message);
  }
  refreshLibrary();
}

boot();
