import { useState, useEffect } from "react"
import {
  listCorpora, getCorpus, createCorpus, updateCorpus,
  deleteCorpus, removeCorpusDocs, getCorpusChangelog,
} from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { cn, formatDate } from "@/lib/utils"
import { Plus, Trash2, ChevronRight, RefreshCw } from "lucide-react"

type Corpus = {
  corpus_id: string; name: string; description: string
  kb_names: string[]; usecase_id: string; agent_filter: string
  sources: unknown[]; doc_ids: string[]
  doc_count: number; chunk_count: number
  last_updated: string; created_at: string
}

type ChangelogEntry = {
  corpus_id: string; action: "added" | "removed"
  doc_id: string; title: string; timestamp: string
}

function CreateCorpusDialog({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [kbNames, setKbNames] = useState("default")
  const [usecaseId, setUsecaseId] = useState("")
  const [agentFilter, setAgentFilter] = useState("")
  const [pending, setPending] = useState(false)
  const [error, setError] = useState("")

  async function submit() {
    setPending(true); setError("")
    try {
      await createCorpus({
        name: name.trim(), description,
        kb_names: kbNames.split(",").map(s => s.trim()).filter(Boolean),
        usecase_id: usecaseId.trim(), agent_filter: agentFilter.trim(),
      })
      setName(""); setDescription(""); setKbNames("default"); setUsecaseId(""); setAgentFilter("")
      onCreated(); onClose()
    } catch (e: unknown) {
      const axiosErr = e as { response?: { data?: { detail?: string } } }
      setError(axiosErr.response?.data?.detail ?? String(e))
    } finally {
      setPending(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={v => !v && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader><DialogTitle>New Corpus</DialogTitle></DialogHeader>
        <div className="space-y-4 pt-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Name *</label>
            <Input placeholder="e.g. support-kb-v2" value={name} onChange={e => setName(e.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Description</label>
            <Textarea rows={2} placeholder="What documents live here and why…" value={description} onChange={e => setDescription(e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">KB Names (comma-separated)</label>
              <Input placeholder="default" value={kbNames} onChange={e => setKbNames(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Use Case ID</label>
              <Input placeholder="support" value={usecaseId} onChange={e => setUsecaseId(e.target.value)} />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Agent Filter</label>
            <Input placeholder="support-agent-v1" value={agentFilter} onChange={e => setAgentFilter(e.target.value)} />
          </div>
          {error && <p className="text-xs text-red-600">{error}</p>}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={submit} disabled={!name.trim() || pending}>
              {pending ? "Creating…" : "Create Corpus"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function EditCorpusDialog({ corpus, onClose, onSaved }: { corpus: Corpus; onClose: () => void; onSaved: () => void }) {
  const [description, setDescription] = useState(corpus.description)
  const [kbNames, setKbNames] = useState(corpus.kb_names.join(", "))
  const [usecaseId, setUsecaseId] = useState(corpus.usecase_id)
  const [agentFilter, setAgentFilter] = useState(corpus.agent_filter)
  const [pending, setPending] = useState(false)

  async function submit() {
    setPending(true)
    try {
      await updateCorpus(corpus.corpus_id, {
        description,
        kb_names: kbNames.split(",").map(s => s.trim()).filter(Boolean),
        usecase_id: usecaseId.trim(), agent_filter: agentFilter.trim(),
      })
      onSaved(); onClose()
    } finally {
      setPending(false)
    }
  }

  return (
    <Dialog open onOpenChange={v => !v && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader><DialogTitle>Edit — {corpus.name}</DialogTitle></DialogHeader>
        <div className="space-y-4 pt-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Description</label>
            <Textarea rows={2} value={description} onChange={e => setDescription(e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">KB Names (comma-separated)</label>
              <Input value={kbNames} onChange={e => setKbNames(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Use Case ID</label>
              <Input value={usecaseId} onChange={e => setUsecaseId(e.target.value)} />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Agent Filter</label>
            <Input value={agentFilter} onChange={e => setAgentFilter(e.target.value)} />
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={submit} disabled={pending}>{pending ? "Saving…" : "Save"}</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function CorpusDetail({ corpusId, onDeleted }: { corpusId: string; onDeleted: () => void }) {
  const [corpus, setCorpus] = useState<Corpus | null>(null)
  const [changelog, setChangelog] = useState<ChangelogEntry[]>([])
  const [detailKey, setDetailKey] = useState(0)
  const [editOpen, setEditOpen] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set())
  const [removing, setRemoving] = useState(false)

  useEffect(() => {
    let cancelled = false
    getCorpus(corpusId).then(d => { if (!cancelled) setCorpus(d as Corpus) }).catch(() => {})
    getCorpusChangelog(corpusId).then(d => { if (!cancelled) setChangelog(d as ChangelogEntry[]) }).catch(() => {})
    return () => { cancelled = true }
  }, [corpusId, detailKey])

  async function doRemove() {
    if (!corpus) return
    setRemoving(true)
    try {
      await removeCorpusDocs(corpusId, { doc_ids: Array.from(selectedDocs), chunk_ids: [], titles: [] })
      setSelectedDocs(new Set())
      setDetailKey(k => k + 1)
    } finally {
      setRemoving(false)
    }
  }

  async function doDelete() {
    if (!corpus) return
    if (!confirm(`Delete corpus "${corpus.name}"? This removes the corpus record but does not delete documents.`)) return
    await deleteCorpus(corpusId)
    onDeleted()
  }

  if (!corpus) return <p className="text-sm text-muted-foreground p-4">Loading…</p>

  const toggleDoc = (id: string) => setSelectedDocs(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })

  return (
    <div className="space-y-6 min-w-0">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold">{corpus.name}</h2>
          {corpus.description && <p className="text-sm text-muted-foreground mt-0.5">{corpus.description}</p>}
        </div>
        <div className="flex gap-2 shrink-0">
          <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>Edit</Button>
          <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700" onClick={doDelete}>Delete</Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {corpus.kb_names.map(kb => (
          <span key={kb} className="inline-flex items-center rounded-md border px-2 py-1 font-mono bg-muted/50">{kb}</span>
        ))}
        {corpus.usecase_id && (
          <span className="inline-flex items-center rounded-md border px-2 py-1 bg-blue-50 text-blue-700 border-blue-200">uc: {corpus.usecase_id}</span>
        )}
        {corpus.agent_filter && (
          <span className="inline-flex items-center rounded-md border px-2 py-1 bg-violet-50 text-violet-700 border-violet-200">agent: {corpus.agent_filter}</span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Card><CardContent className="p-4 text-center">
          <p className="text-2xl font-bold">{corpus.doc_count}</p>
          <p className="text-xs text-muted-foreground">Documents</p>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <p className="text-2xl font-bold">{corpus.chunk_count}</p>
          <p className="text-xs text-muted-foreground">Chunks</p>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <p className="text-sm font-medium">{formatDate(corpus.last_updated)}</p>
          <p className="text-xs text-muted-foreground">Last updated</p>
        </CardContent></Card>
      </div>

      <Tabs defaultValue="docs">
        <TabsList>
          <TabsTrigger value="docs">Documents ({corpus.doc_count})</TabsTrigger>
          <TabsTrigger value="changelog">Changelog ({changelog.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="docs" className="space-y-3 mt-4">
          {selectedDocs.size > 0 && (
            <div className="flex items-center gap-3 rounded-md border bg-muted/40 px-4 py-2">
              <span className="text-sm font-medium">{selectedDocs.size} selected</span>
              <Button variant="destructive" size="sm" disabled={removing}
                onClick={() => confirm(`Remove ${selectedDocs.size} document(s) from this corpus?`) && doRemove()}>
                <Trash2 className="h-3 w-3 mr-1" /> Remove from corpus
              </Button>
            </div>
          )}
          {corpus.doc_ids.length === 0 ? (
            <p className="text-sm text-muted-foreground">No documents in this corpus yet.</p>
          ) : (
            <Card>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead className="border-b">
                    <tr className="text-left text-muted-foreground text-xs">
                      <th className="px-4 py-2 w-8"></th>
                      <th className="px-4 py-2 font-medium">Document ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {corpus.doc_ids.map(docId => (
                      <tr key={docId} className={cn("border-b last:border-0 hover:bg-muted/40 cursor-pointer", selectedDocs.has(docId) && "bg-muted/60")}>
                        <td className="px-4 py-2">
                          <input type="checkbox" checked={selectedDocs.has(docId)} onChange={() => toggleDoc(docId)} className="cursor-pointer" />
                        </td>
                        <td className="px-4 py-2 font-mono text-xs">{docId}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="changelog" className="mt-4">
          {changelog.length === 0 ? (
            <p className="text-sm text-muted-foreground">No changes recorded yet.</p>
          ) : (
            <Card>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead className="border-b">
                    <tr className="text-left text-muted-foreground text-xs">
                      <th className="px-4 py-2 font-medium">Action</th>
                      <th className="px-4 py-2 font-medium">Document</th>
                      <th className="px-4 py-2 font-medium">When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {changelog.map((entry, i) => (
                      <tr key={i} className="border-b last:border-0">
                        <td className="px-4 py-2">
                          <Badge variant={entry.action === "added" ? "success" : "destructive"} className="text-xs">
                            {entry.action}
                          </Badge>
                        </td>
                        <td className="px-4 py-2">
                          <span className="font-medium">{entry.title || "—"}</span>
                          <span className="block font-mono text-xs text-muted-foreground">{entry.doc_id}</span>
                        </td>
                        <td className="px-4 py-2 text-xs text-muted-foreground">{formatDate(entry.timestamp)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {editOpen && (
        <EditCorpusDialog
          corpus={corpus}
          onClose={() => setEditOpen(false)}
          onSaved={() => setDetailKey(k => k + 1)}
        />
      )}
    </div>
  )
}

export default function CorpusPage() {
  const [corpora, setCorpora] = useState<Corpus[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    listCorpora()
      .then(d => { if (!cancelled) { setCorpora(d as Corpus[]); setLoading(false) } })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [refreshKey])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Corpus</h1>
          <p className="text-muted-foreground text-sm mt-1">Named document collections scoped to a use case</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setRefreshKey(k => k + 1)}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4 mr-1.5" /> New Corpus
          </Button>
        </div>
      </div>

      <div className="flex gap-6 items-start">
        <div className="w-72 shrink-0 space-y-1">
          {loading ? (
            <p className="text-sm text-muted-foreground px-2">Loading…</p>
          ) : corpora.length === 0 ? (
            <p className="text-sm text-muted-foreground px-2">No corpora yet. Create one to get started.</p>
          ) : (
            corpora.map(c => (
              <button
                key={c.corpus_id}
                onClick={() => setSelected(c.corpus_id)}
                className={cn(
                  "w-full text-left rounded-lg border px-4 py-3 transition-colors hover:bg-muted/50",
                  selected === c.corpus_id ? "border-primary bg-muted/60" : "border-transparent"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm truncate">{c.name}</span>
                  <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                </div>
                <div className="flex gap-2 mt-1 flex-wrap">
                  {c.kb_names.map(kb => <span key={kb} className="text-xs font-mono text-muted-foreground">{kb}</span>)}
                </div>
                <p className="text-xs text-muted-foreground mt-1">{c.doc_count} docs · {c.chunk_count} chunks</p>
              </button>
            ))
          )}
        </div>

        <div className="flex-1 min-w-0">
          {selected ? (
            <CorpusDetail
              key={selected}
              corpusId={selected}
              onDeleted={() => { setSelected(null); setRefreshKey(k => k + 1) }}
            />
          ) : (
            <div className="flex items-center justify-center h-64 rounded-lg border border-dashed text-muted-foreground text-sm">
              Select a corpus to view details
            </div>
          )}
        </div>
      </div>

      <CreateCorpusDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => setRefreshKey(k => k + 1)}
      />
    </div>
  )
}
