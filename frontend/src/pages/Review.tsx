import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { ChevronDown, ChevronRight, CheckCircle2, XCircle, Rocket, FileText, Globe, FileCode } from "lucide-react"
import { listDocs, getDoc, approveDoc, rejectDoc, pushDocs, updateChunk } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import QualityBadge from "@/components/QualityBadge"
import { cn, formatDate, scoreBg } from "@/lib/utils"

type DocSummary = Record<string, unknown>
type Chunk = Record<string, unknown>

const STATUS_TABS = ["all", "pending_review", "approved", "rejected"] as const
type StatusTab = (typeof STATUS_TABS)[number]

const STATUS_LABEL: Record<string, string> = {
  pending_review: "Needs Review",
  approved: "Approved",
  rejected: "Rejected",
  pushed: "Pushed",
}

const SOURCE_ICON: Record<string, React.FC<{ className?: string }>> = {
  pdf: FileText,
  docx: FileText,
  pptx: FileText,
  html: Globe,
  url: Globe,
  confluence: FileCode,
}

function sourceIcon(type: string) {
  const Icon = SOURCE_ICON[type] ?? FileText
  return <Icon className="h-4 w-4 text-muted-foreground" />
}

function statusBadge(status: string) {
  const map: Record<string, string> = {
    pending_review: "bg-amber-50 text-amber-700 border-amber-200",
    approved: "bg-blue-50 text-blue-700 border-blue-200",
    rejected: "bg-red-50 text-red-700 border-red-200",
    pushed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  }
  return (
    <span className={cn("inline-flex rounded-md border px-2 py-0.5 text-xs font-semibold", map[status] ?? "")}>
      {STATUS_LABEL[status] ?? status}
    </span>
  )
}

function ChunkRow({ chunk, docId }: { chunk: Chunk; docId: string }) {
  const [editing, setEditing] = useState(false)
  const [content, setContent] = useState(String(chunk.content ?? ""))
  const [tags, setTags] = useState((chunk.tags as string[] ?? []).join(", "))
  const qc = useQueryClient()

  const save = useMutation({
    mutationFn: () => updateChunk(docId, String(chunk._id ?? chunk.chunk_id), {
      content,
      tags: tags.split(",").map(t => t.trim()).filter(Boolean),
    }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["doc", docId] }); setEditing(false) },
  })

  const flags = chunk.quality_flags as string[] ?? []

  return (
    <div className="border rounded-md p-3 text-sm space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-xs text-muted-foreground truncate">{String(chunk.section ?? "—")}</p>
          {chunk.page_number != null && (
            <span className="text-xs text-muted-foreground">p.{String(chunk.page_number)}</span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {flags.map(f => (
            <span key={f} className="inline-flex rounded border border-amber-200 bg-amber-50 text-amber-700 px-1.5 py-0.5 text-xs">{f}</span>
          ))}
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setEditing(e => !e)}>
            {editing ? "Cancel" : "Edit"}
          </Button>
        </div>
      </div>

      {editing ? (
        <div className="space-y-2">
          <Textarea value={content} onChange={e => setContent(e.target.value)} rows={4} className="text-sm" />
          <div className="flex items-center gap-2">
            <Input value={tags} onChange={e => setTags(e.target.value)} placeholder="tags, comma-separated" className="text-xs h-7" />
            <Button size="sm" className="h-7 text-xs shrink-0" onClick={() => save.mutate()} disabled={save.isPending}>
              {save.isPending ? "Saving…" : "Save"}
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-xs text-foreground/80 line-clamp-3 whitespace-pre-wrap">{String(chunk.content ?? "")}</p>
      )}
    </div>
  )
}

function DocCard({ doc }: { doc: DocSummary }) {
  const [open, setOpen] = useState(false)
  const qc = useQueryClient()
  const docId = String(doc._id ?? doc.doc_id)

  const { data: detail } = useQuery({
    queryKey: ["doc", docId],
    queryFn: () => getDoc(docId),
    enabled: open,
  })

  const approve = useMutation({
    mutationFn: () => approveDoc(docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["docs"] }),
  })
  const reject = useMutation({
    mutationFn: () => rejectDoc(docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["docs"] }),
  })
  const push = useMutation({
    mutationFn: () => pushDocs(docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["docs"] }),
  })

  const chunks: Chunk[] = detail?.chunks ?? []
  const status = String(doc.status ?? "")
  const score = Number(doc.quality_score ?? 0)

  return (
    <div className="border rounded-lg bg-white shadow-sm">
      {/* Header */}
      <div
        className="flex items-center gap-3 p-4 cursor-pointer select-none"
        onClick={() => setOpen(o => !o)}
      >
        {open ? <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />}
        {sourceIcon(String(doc.source_type ?? ""))}
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{String(doc.title ?? "Untitled")}</p>
          <p className="text-xs text-muted-foreground">
            {String(doc.chunk_count ?? 0)} sections · {formatDate(String(doc.ingested_at ?? ""))}
            {doc.usecase_id ? ` · ${String(doc.usecase_id)}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0" onClick={e => e.stopPropagation()}>
          <QualityBadge score={score} />
          {statusBadge(status)}
          {status === "pending_review" && (
            <Button size="sm" className="h-7 text-xs" onClick={() => approve.mutate()} disabled={approve.isPending}>
              <CheckCircle2 className="h-3.5 w-3.5 mr-1" />Approve
            </Button>
          )}
          {status === "approved" && (
            <Button size="sm" className="h-7 text-xs" onClick={() => push.mutate()} disabled={push.isPending}>
              <Rocket className="h-3.5 w-3.5 mr-1" />{push.isPending ? "Pushing…" : "Push"}
            </Button>
          )}
          {status !== "rejected" && status !== "pushed" && (
            <Button size="sm" variant="outline" className="h-7 text-xs text-red-600 border-red-200 hover:bg-red-50" onClick={() => reject.mutate()} disabled={reject.isPending}>
              <XCircle className="h-3.5 w-3.5 mr-1" />Reject
            </Button>
          )}
        </div>
      </div>

      {/* Quality flags */}
      {open && (doc.quality_flags as string[] ?? []).length > 0 && (
        <div className="px-4 pb-2 flex flex-wrap gap-1">
          {(doc.quality_flags as string[]).map(f => (
            <span key={f} className={cn("inline-flex rounded border px-2 py-0.5 text-xs", scoreBg(score))}>{f}</span>
          ))}
        </div>
      )}

      {/* Chunks */}
      {open && (
        <div className="px-4 pb-4 space-y-2">
          {chunks.length === 0 ? (
            <p className="text-xs text-muted-foreground">Loading sections…</p>
          ) : (
            chunks.map((c, i) => <ChunkRow key={i} chunk={c} docId={docId} />)
          )}
        </div>
      )}
    </div>
  )
}

export default function Review() {
  const [tab, setTab] = useState<StatusTab>("all")
  const [pushing, setPushing] = useState(false)
  const qc = useQueryClient()

  const { data: docs = [], isLoading } = useQuery<DocSummary[]>({
    queryKey: ["docs"],
    queryFn: () => listDocs(),
    refetchInterval: 10000,
  })

  const filtered = tab === "all" ? docs : docs.filter(d => d.status === tab)
  const approvedCount = docs.filter(d => d.status === "approved").length

  async function pushAll() {
    setPushing(true)
    try {
      await pushDocs()
      qc.invalidateQueries({ queryKey: ["docs"] })
      qc.invalidateQueries({ queryKey: ["stats"] })
    } finally {
      setPushing(false)
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Review Queue</h1>
          <p className="text-muted-foreground text-sm mt-1">{docs.length} staged documents</p>
        </div>
        {approvedCount > 0 && (
          <Button onClick={pushAll} disabled={pushing} className="gap-2">
            <Rocket className="h-4 w-4" />
            {pushing ? "Pushing…" : `Push ${approvedCount} Approved`}
          </Button>
        )}
      </div>

      {/* Status tabs */}
      <div className="flex gap-1 border-b">
        {STATUS_TABS.map(t => {
          const count = t === "all" ? docs.length : docs.filter(d => d.status === t).length
          return (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors",
                tab === t ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"
              )}
            >
              {t === "all" ? "All" : STATUS_LABEL[t]}
              {count > 0 && (
                <span className="ml-1.5 rounded-full bg-muted text-muted-foreground text-xs px-1.5 py-0.5">{count}</span>
              )}
            </button>
          )
        })}
      </div>

      {/* Doc list */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          <p className="text-sm">No documents in this category.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((doc, i) => <DocCard key={i} doc={doc} />)}
        </div>
      )}
    </div>
  )
}
