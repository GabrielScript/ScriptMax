// Cópia de segurança da gravação no IndexedDB, pedaço a pedaço.
// Se o celular matar a aba (tela bloqueada, economia de bateria), o áudio gravado até ali
// fica no navegador e é oferecido de volta na próxima vez que o site abrir.
//
// Só o primeiro pedaço tem o cabeçalho do contêiner (WebM/MP4), então a ordem é sagrada:
// as escritas rodam em fila e, na primeira falha, a cópia para. O que fica salvo é sempre
// um prefixo contínuo da gravação, que o ffmpeg do servidor lê normalmente.

const DB_NAME = 'scriptmax-gravacoes';
const DB_VERSION = 1;
const SESSIONS = 'sessions';
const CHUNKS = 'chunks';

let dbPromise = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        reject(new Error('IndexedDB indisponível.'));
        return;
      }
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore(SESSIONS, { keyPath: 'id' });
        db.createObjectStore(CHUNKS, { autoIncrement: true }).createIndex('session', 'session');
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    dbPromise.catch(() => { dbPromise = null; });
  }
  return dbPromise;
}

function done(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error('Transação abortada.'));
  });
}

function result(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export class RecordingBackup {
  #id = crypto.randomUUID();
  #session = null;
  #queue = Promise.resolve();
  #failed = false;

  get id() {
    return this.#id;
  }

  /** Enfileira um pedaço. Nunca lança: se falhar, a cópia para e a gravação segue só na memória. */
  append(blob, elapsedMs) {
    if (this.#failed) return;
    this.#queue = this.#queue
      .then(() => this.#write(blob, elapsedMs))
      .catch(() => { this.#failed = true; });
  }

  /** Apaga a cópia depois das escritas pendentes; pedaços que chegarem depois são ignorados. */
  discard() {
    this.#failed = true;
    this.#queue = this.#queue.then(() => RecordingBackup.remove(this.#id)).catch(() => {});
    return this.#queue;
  }

  async #write(blob, elapsedMs) {
    if (this.#failed) return;
    const db = await openDb();
    const now = Date.now();
    if (!this.#session) {
      this.#session = { id: this.#id, mimeType: blob.type || 'audio/webm', startedAt: now };
      navigator.storage?.persist?.().catch(() => {}); // pede para o navegador não apagar sob pouco espaço
    }
    const transaction = db.transaction([SESSIONS, CHUNKS], 'readwrite');
    // Pedaço e sessão na mesma transação: a sessão só existe se tiver ao menos o primeiro pedaço.
    transaction.objectStore(CHUNKS).add({ session: this.#id, blob });
    transaction.objectStore(SESSIONS).put({ ...this.#session, lastWriteAt: now, elapsedMs });
    await done(transaction);
  }

  /**
   * Gravação interrompida mais recente, ou null. Sessões escritas há menos de `staleMs`
   * são de outra aba ainda gravando e ficam de fora.
   */
  static async findOrphan(staleMs) {
    const db = await openDb();
    const sessions = await result(db.transaction(SESSIONS).objectStore(SESSIONS).getAll());
    const orphans = sessions
      .filter((session) => Date.now() - session.lastWriteAt > staleMs)
      .sort((a, b) => b.lastWriteAt - a.lastWriteAt);
    if (orphans.length === 0) return null;
    const session = orphans[0];
    const index = db.transaction(CHUNKS).objectStore(CHUNKS).index('session');
    // O índice devolve na ordem da chave primária (autoincremento), que é a ordem de gravação.
    const chunks = await result(index.getAll(session.id));
    if (chunks.length === 0) {
      await RecordingBackup.remove(session.id);
      return null;
    }
    return {
      id: session.id,
      blob: new Blob(chunks.map((chunk) => chunk.blob), { type: session.mimeType }),
      startedAt: session.startedAt,
      elapsedMs: session.elapsedMs,
    };
  }

  static async remove(id) {
    const db = await openDb();
    const transaction = db.transaction([SESSIONS, CHUNKS], 'readwrite');
    transaction.objectStore(SESSIONS).delete(id);
    const index = transaction.objectStore(CHUNKS).index('session');
    const keys = await result(index.getAllKeys(id));
    for (const key of keys) transaction.objectStore(CHUNKS).delete(key);
    await done(transaction);
  }
}
