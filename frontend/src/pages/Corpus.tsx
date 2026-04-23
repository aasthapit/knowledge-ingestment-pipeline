import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  listCorpora,
  getCorpus,
  createCorpus,
  updateCorpus,
  deleteCorpus,
  removeCorpusDocs,
  getCorpusChangelog,
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
  corpus_id: string
  name: string
  description: string
  kb_names: string[]
  usecase_id: string
  agent_filter: string
  sources: unknown[]
  doc_ids: string[]
  doc_count: number
  chunk_count: number
  last_updated: string
  created_at: string
}

type ChangelogEntry = {
  corpus_id: string
  action: "added" | "removed"
  doc_id: string
  title: string
  timestamp: string
}

function CreateCorpusDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [kbNames, setKbNames] = useState("default")
  const [usecaseId, setUsecaseId] = useState("")
  const [agentFilter, setAgentFilter] = useState("")

  const { mutate, isPending, error } = useMutation({
    mutationFn: () =>
      createCorpus({
        name: name.trim(),
        description,
        kb_names: kbNames.split(",").map(s => s.trim()).filter(Boolean),
        usecase_id: usecaseId.trim(),
        agent_filter: agentFilter.trim(),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["corpora"] })
      setName("")
      setDescription("")
      setKbNames("default")
      setUsecaseId("")
      setAgentFilter("")
      onClose()
    },
  })

  return (
    <Dialog open={open} onOpenChange={v => !v && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>New Corpus</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 pt-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Name *</label>
            <Input placeholder="e.g. support-kb-v2" value={name} onChange={e => setName(e.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Description</label>
            <Textarea
              rows={2}
              placeholder="What documents live here and why…"
              value={description}
              onChange={e => setDescription(e.target.value)}
            />
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
          {error && (
            <p className="text-xs text-red-600">{(error as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? String(error)}</p>
          )}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button onClick={() => mutate()} disabled={!name.trim() || isPending}>
              {isPending ? "Creating…" : "Create Corpus"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function EditCorpusDialog({
  corpus,
  onClose,
}: {
  corpus: Corpus
  onClose: () => void
}) {
  const qc = useQueryClient()
  const [description, setDescription] = useState(corpus.description)
  const [kbNames, setKbNames] = useState(corpus.kb_names.join(", "))
  const [usecaseId, setUsecaseId] = useState(corpus.usecase_id)
  const [agentFilter, setAgentFilter] = useState(corpus.agent_filter)

  const { mutate, isPending } = useMutation({
    mutationFn: () =>
      updateCorpus(corpus.corpus_id, {
        description,
        kb_names: kbNames.split(",").map(s => s.trim()).filter(Boolean),
        usecase_id: usecaseId.trim(),
        agent_filter: agentFilter.trim(),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["corpora"] })
      qc.invalidateQueries({ queryKey: ["corpus", corpus.corpus_id] })
      onClose()
    },
  })

  return (
    <Dialog open onOpenChange={v => !v && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit — {corpus.name}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 pt-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Description</label>
            <Textarea
              rows={2}
              value={description}
              onChange={e => setDescription(e.target.value)}
            />
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
            <Button onClick={() => mutate()} disabled={isPending}>
              {isPending ? "Saving…" : "Save"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function CorpusDetail({ corpusId }: { corpusId: string }) {
  const qc = useQueryClient()
  const [editOpen, setEditOpen] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set())

  const { data: corpus, isLoading } = useQuery({
    queryKey: ["corpus", corpusId],
    queryFn: () => getCorpus(corpusId),
  })

  const { data: changelog = [] } = useQuery({
    queryKey: ["corpus-changelog", corpusId],
    queryFn: () => getCorpusChangelog(corpusId),
  })

  const { mutate: doRemove, isPending: removing } = useMutation({
    mutationFn: () =>
      removeCorpusDocs(corpusId, {
        doc_ids: Array.from(selectedDocs),
        chunk_ids: [],
        titles: [],
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["corpus", corpusId] })
      qc.invalidateQueries({ queryKey: ["corpus-changelog", corpusId] })
      qc.invalidateQueries({ queryKey: ["corpora"] })
      setSelectedDocs(new Set())
    },
  })

  const { mutate: doDelete } = useMutation({
    mutationFn: () => deleteCorpus(corpusId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["corpora"] })
    },
  })

  if (isLoading) return <p className="text-sm text-muted-foreground p-4">Loading…</p>
  if (!corpus) return null

  const c = corpus as Corpus
  const log = changelog as ChangelogEntry[]

  const toggleDoc = (id: string) => {
    setSelectedDocs(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-6 min-w-0">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold">{c.name}</h2>
          {c.description && <p className="text-sm text-muted-foreground mt-0.5">{c.description}</p>}
        </div>
        <div className="flex gap-2 shrink-0">
          <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>Edit</Button>
          <Button
            variant="outline"
            size="sm"
            className="text-red-600 hover:text-red-700"
            onClick={() => {
              if (confirm(`Delete corpus "${c.name}"? This removes the corpus record but does not delete documents.`)) {
                doDelete()
              }
            }}
          >
            Delete
          </Button>
        </div>
      </div>

      {/* Meta chips */}
      <div className="flex flex-wrap gap-2 text-xs">
        {c.kb_names.map(kb => (
          <span key={kb} className="inline-flex items-center rounded-md border px-2 py-1 font-mono bg-muted/50">{kb}</span>
        ))}
        {c.usecase_id && (
          <span className="inline-flex items-center rounded-md border px-2 py-1 bg-blue-50 text-blue-700 border-blue-200">
            uc: {c.usecase_id}
          </span>
        )}
        {c.agent_filter && (
          <span className="inline-flex items-center rounded-md border px-2 py-1 bg-violet-50 text-violet-700 border-violet-200">
            agent: {c.agent_filter}
          </span>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <Card><CardContent className="p-4 text-center">
          <p className="text-2xl font-bold">{c.doc_count}</p>
          <p className="text-xs text-muted-foreground">Documents</p>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <p className="text-2xl font-bold">{c.chunk_count}</p>
          <p className="text-xs text-muted-foreground">Chunks</p>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <p className="text-sm font-medium">{formatDate(c.last_updated)}</p>
          <p className="text-xs text-muted-foreground">Last updated</p>
        </CardContent></Card>
      </div>

      <Tabs defaultValue="docs">
        <TabsList>
          <TabsTrigger value="docs">Documents ({c.doc_count})</TabsTrigger>
          <TabsTrigger value="changelog">Changelog ({log.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="docs" className="space-y-3 mt-4">
          {selectedDocs.size > 0 && (
            <div className="flex items-center gap-3 rounded-md border bg-muted/40 px-4 py-2">
              <span className="text-sm font-medium">{selectedDocs.size} selected</span>
              <Button
                variant="destructive"
                size="sm"
                disabled={removing}
                onClick={() => {
                  if (confirm(`Remove ${selectedDocs.size} document(s) from this corpus?`)) doRemove()
                }}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Remove from corpus
              </Button>
            </div>
          )}
          {c.doc_ids.length === 0 ? (
            <p className="text-sm text-muted-foreground">No documents in this corpus yet. Push documents with this corpus's use case ID and agent filter to populate it.</p>
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
                    {c.doc_ids.map(docId => (
                      <tr key={docId} className={cn("border-b last:border-0 hover:bg-muted/40 cursor-pointer", selectedDocs.has(docId) && "bg-muted/60")}>
                        <td className="px-4 py-2">
                          <input
                            type="checkbox"
                            checked={selectedDocs.has(docId)}
                            onChange={() => toggleDoc(docId)}
                            className="cursor-pointer"
                          />
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
          {log.length === 0 ? (
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
                    {log.map((entry, i) => (
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

      {editOpen && <EditCorpusDialog corpus={c} onClose={() => setEditOpen(false)} />}
    </div>
  )
}

export default function Corpus() {
  const qc = useQueryClient()
  const [selected, setSelected] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)

  const { data: corpora = [], isLoading, refetch } = useQuery({
    queryKey: ["corpora"],
    queryFn: listCorpora,
  })

  const list = corpora as Corpus[]

  const selectedCorpus = selected
    ? list.find(c => c.corpus_id === selected) ?? null
    : null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Corpus</h1>
          <p className="text-muted-foreground text-sm mt-1">Named document collections scoped to a use case</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4 mr-1.5" />
            New Corpus
          </Button>
        </div>
      </div>

      <div className="flex gap-6 items-start">
        {/* Corpus list */}
        <div className="w-72 shrink-0 space-y-1">
          {isLoading ? (
            <p className="text-sm text-muted-foreground px-2">Loading…</p>
          ) : list.length === 0 ? (
            <p className="text-sm text-muted-foreground px-2">No corpora yet. Create one to get started.</p>
          ) : (
            list.map(c => (
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
                  {c.kb_names.map(kb => (
                    <span key={kb} className="text-xs font-mono text-muted-foreground">{kb}</span>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {c.doc_count} docs · {c.chunk_count} chunks
                </p>
              </button>
            ))
          )}
        </div>

        {/* Detail panel */}
        <div className="flex-1 min-w-0">
          {selected ? (
            <CorpusDetail
              key={selected}
              corpusId={selected}
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
        onClose={() => {
          setCreateOpen(false)
          // Auto-select newly created corpus
          qc.invalidateQueries({ queryKey: ["corpora"] })
        }}
      />
    </div>
  )
}
