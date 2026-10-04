// Cliente da API. Sessão via cookie httpOnly (o token nunca fica no JS/localStorage).

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function describeDetail(detail) {
  if (Array.isArray(detail)) return detail.map((item) => item.msg ?? String(item)).join('; ');
  return typeof detail === 'string' && detail ? detail : 'Erro inesperado no servidor.';
}

async function errorFrom(response) {
  try {
    const payload = await response.json();
    return new ApiError(response.status, describeDetail(payload.detail));
  } catch {
    return new ApiError(response.status, `Erro ${response.status}.`);
  }
}

async function request(method, url, body) {
  const options = { method, credentials: 'same-origin', headers: {} };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiError(0, 'Sem conexão com o servidor do ScriptMax.');
  }
  if (!response.ok) throw await errorFrom(response);
  return response.status === 204 ? null : response.json();
}

const reportPath = (id) => `/api/reports/${encodeURIComponent(id)}`;

export const api = {
  config: () => request('GET', '/api/config'),
  login: (token) => request('POST', '/api/login', { token }),
  logout: () => request('POST', '/api/logout'),
  jobs: () => request('GET', '/api/jobs'),
  retryJob: (id) => request('POST', `/api/jobs/${encodeURIComponent(id)}/retry`),
  reports: () => request('GET', '/api/reports'),
  moveReport: (id, category, folder) => request('PATCH', reportPath(id), { category, folder }),
  deleteReport: (id) => request('DELETE', reportPath(id)),
  regenerateReport: (id) => request('POST', `${reportPath(id)}/regenerate`),
  openLibrary: () => request('POST', '/api/library/open'),
};

export function reportFileUrl(id, kind, { download = false } = {}) {
  return `${reportPath(id)}/files/${kind}${download ? '?download=1' : ''}`;
}

// XHR (e não fetch) porque só ele informa o progresso do envio.
export function uploadAudio({ fields, file, filename, onProgress }) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) form.append(key, value);
    form.append('file', file, filename);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.responseType = 'json';
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new ApiError(xhr.status, describeDetail(xhr.response?.detail)));
    });
    xhr.addEventListener('error', () => reject(new ApiError(0, 'A conexão caiu durante o envio.')));
    xhr.send(form);
  });
}
