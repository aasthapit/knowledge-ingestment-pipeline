import axios from "axios"

export const api = axios.create({ baseURL: "/api" })

// ── Stats / Status ─────────────────────────────────────────────────────────────
export const getStats = () => api.get("/stats").then(r => r.data)
export const getStatus = () => api.get("/status").then(r => r.data)

// ── Review ─────────────────────────────────────────────────────────────────────
export const listDocs = (status?: string) =>
  api.get("/review/", { params: status ? { status } : {} }).then(r => r.data.docs)
export const getDoc = (id: string) => api.get(`/review/${id}`).then(r => r.data)
export const approveDoc = (id: string) => api.post(`/review/${id}/approve`)
export const rejectDoc = (id: string, reason = "") =>
  api.post(`/review/${id}/reject`, { reason })
export const updateChunk = (docId: string, chunkId: string, updates: Record<string, unknown>) =>
  api.patch(`/review/${docId}/chunks/${chunkId}`, updates)
export const splitDoc = (docId: string, chunkIds: string[], newTitle: string) =>
  api.post(`/review/${docId}/split`, { chunk_ids: chunkIds, new_title: newTitle })
export const splitChunk = (docId: string, chunkId: string, parts: string[]) =>
  api.post(`/review/${docId}/chunks/${chunkId}/split`, { content_parts: parts })
export const pushDocs = (docId?: string, removeAfter = false) =>
  api.post("/review/push", { doc_id: docId ?? null, remove_after_push: removeAfter }).then(r => r.data)

// ── Search ─────────────────────────────────────────────────────────────────────
export const search = (payload: Record<string, unknown>) =>
  api.post("/search/", payload).then(r => r.data)
export const listUsecases = () => api.get("/search/usecases").then(r => r.data.usecases)
export const listAgents = (ucId: string) =>
  api.get(`/search/usecases/${ucId}/agents`).then(r => r.data.agents)

// ── Ledger ─────────────────────────────────────────────────────────────────────
export const listLedger = (kbName?: string, driftStatus?: string) =>
  api.get("/ledger/", { params: { kb_name: kbName, drift_status: driftStatus } }).then(r => r.data.docs)
export const listSnapshots = () => api.get("/ledger/snapshots").then(r => r.data.snapshots)
export const getSnapshot = (id: string) => api.get(`/ledger/snapshots/${id}`).then(r => r.data)
export const runDriftCheck = (kbName?: string) =>
  api.post("/ledger/drift-check", null, { params: { kb_name: kbName } }).then(r => r.data)
export const removeDoc = (docId: string) => api.delete(`/ledger/${docId}`)

// ── Manifests ──────────────────────────────────────────────────────────────────
export const listManifests = (params?: Record<string, string>) =>
  api.get("/manifests/", { params }).then(r => r.data.manifests)
export const getManifest = (id: string) => api.get(`/manifests/${id}`).then(r => r.data)
export const createManifest = (payload: Record<string, unknown>) =>
  api.post("/manifests/", payload).then(r => r.data)
export const snapshotManifest = (payload: Record<string, unknown>) =>
  api.post("/manifests/snapshot", payload).then(r => r.data)
export const diffManifests = (a: string, b: string) =>
  api.post("/manifests/diff", { manifest_id_a: a, manifest_id_b: b }).then(r => r.data)
export const freezeManifest = (id: string) => api.post(`/manifests/${id}/freeze`)
export const archiveManifest = (id: string) => api.post(`/manifests/${id}/archive`)
export const removeManifestDocs = (id: string, docIds?: string[]) =>
  api.delete(`/manifests/${id}/docs`, { data: { doc_ids: docIds ?? null } }).then(r => r.data)

// ── Corpus ─────────────────────────────────────────────────────────────────────
export const listCorpora = () => api.get("/corpus/").then(r => r.data.corpora)
export const getCorpus = (id: string) => api.get(`/corpus/${id}`).then(r => r.data)
export const createCorpus = (payload: Record<string, unknown>) =>
  api.post("/corpus/", payload).then(r => r.data)
export const updateCorpus = (id: string, payload: Record<string, unknown>) =>
  api.patch(`/corpus/${id}`, payload).then(r => r.data)
export const deleteCorpus = (id: string) => api.delete(`/corpus/${id}`)
export const addCorpusDocs = (id: string, payload: Record<string, unknown>) =>
  api.post(`/corpus/${id}/docs`, payload).then(r => r.data)
export const removeCorpusDocs = (id: string, payload: Record<string, unknown>) =>
  api.delete(`/corpus/${id}/docs`, { data: payload }).then(r => r.data)
export const getCorpusChangelog = (id: string) =>
  api.get(`/corpus/${id}/changelog`).then(r => r.data.changelog)
