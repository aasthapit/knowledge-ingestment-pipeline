import { useState, useEffect } from "react"
import { FolderGit2, ChevronDown, ChevronRight } from "lucide-react"
import { listManifests, getManifest, freezeManifest, snapshotManifest, diffManifests } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn, formatDate } from "@/lib/utils"

const STATUS_COLOR: Record<string, string> = {
  open: "bg-blue-50 text-blue-700 border-blue-200",
  frozen: "bg-emerald-50 text-emerald-700 border-emerald-200",
  archived: "bg-gray-50 text-gray-500 border-gray-200",
}

function ManifestRow({ m, onRefresh }: { m: Record<string, unknown>; onRefresh: () => void }) {
  const [open, setOpen] = useState(false)
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null)
  const [freezing, setFreezing] = useState(false)
  const mId = String(m._id ?? "")

  useEffect(() => {
    if (!open) return
    let cancelled = false
    getManifest(mId).then(d => { if (!cancelled) setDetail(d as Record<string, unknown>) }).catch(() => {})
    return () => { cancelled = true }
  }, [open, mId])

  async function freeze() {
    setFreezing(true)
    try { await freezeManifest(mId); onRefresh() } finally { setFreezing(false) }
  }

  return (
    <div className="border rounded-lg bg-white">
      <div className="flex items-center gap-3 p-3 cursor-pointer" onClick={() => setOpen(o => !o)}>
        {open ? <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />}
        <FolderGit2 className="h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{String(m.name ?? "Unnamed")}</p>
          <p className="text-xs text-muted-foreground">
            {String(m.entry_count ?? 0)} entries · {String(m.pushed_count ?? 0)} pushed
            {m.usecase_id ? ` · ${String(m.usecase_id)}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0" onClick={e => e.stopPropagation()}>
          <span className={cn("inline-flex rounded border px-2 py-0.5 text-xs font-semibold", STATUS_COLOR[String(m.status ?? "")] ?? "")}>
            {String(m.status ?? "")}
          </span>
          <span className="text-xs text-muted-foreground">{formatDate(String(m.created_at ?? ""))}</span>
          {m.status === "open" && (
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={freeze} disabled={freezing}>
              Freeze
            </Button>
          )}
        </div>
      </div>

      {open && detail && (
        <div className="px-4 pb-4">
          <table className="w-full text-xs mt-2">
            <thead className="border-b">
              <tr className="text-left text-muted-foreground">
                <th className="py-1.5 font-medium">Title</th>
                <th className="py-1.5 font-medium">Type</th>
                <th className="py-1.5 font-medium">Status</th>
                <th className="py-1.5 font-medium">Version</th>
                <th className="py-1.5 font-medium">Pushed</th>
              </tr>
            </thead>
            <tbody>
              {(detail.entries as Record<string, unknown>[] ?? []).map((e, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-1.5 max-w-xs truncate font-medium">{String(e.title ?? "—")}</td>
                  <td className="py-1.5 text-muted-foreground">{String(e.source_type ?? "—")}</td>
                  <td className="py-1.5"><span className="rounded bg-muted px-1.5 py-0.5">{String(e.status ?? "—")}</span></td>
                  <td className="py-1.5 font-mono text-muted-foreground">{String(e.version_id ?? "—").slice(0, 8)}</td>
                  <td className="py-1.5 text-muted-foreground">{formatDate(String(e.pushed_at ?? ""))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function Manifests() {
  const [manifests, setManifests] = useState<Record<string, unknown>[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)

  const [snapName, setSnapName] = useState("")
  const [snapUcId, setSnapUcId] = useState("")
  const [snapAgent, setSnapAgent] = useState("")
  const [snapping, setSnapping] = useState(false)
  const [snapDone, setSnapDone] = useState(false)

  const [diffA, setDiffA] = useState("")
  const [diffB, setDiffB] = useState("")
  const [diffResult, setDiffResult] = useState<Record<string, unknown> | null>(null)
  const [diffing, setDiffing] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    listManifests()
      .then(d => { if (!cancelled) { setManifests(d as Record<string, unknown>[]); setLoading(false) } })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [refreshKey])

  async function doSnapshot() {
    setSnapping(true); setSnapDone(false)
    try {
      await snapshotManifest({ usecase_id: snapUcId, agent_filter: snapAgent, manifest_name: snapName })
      setRefreshKey(k => k + 1)
      setSnapName(""); setSnapUcId(""); setSnapAgent("")
      setSnapDone(true)
    } finally {
      setSnapping(false)
    }
  }

  async function doDiff() {
    setDiffing(true); setDiffResult(null)
    try {
      const r = await diffManifests(diffA, diffB)
      setDiffResult(r as Record<string, unknown>)
    } finally {
      setDiffing(false)
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Manifests</h1>
        <p className="text-muted-foreground text-sm mt-1">Corpus versioning and snapshot management</p>
      </div>

      <Tabs defaultValue="browse">
        <TabsList>
          <TabsTrigger value="browse">Browse</TabsTrigger>
          <TabsTrigger value="snapshot">Snapshot</TabsTrigger>
          <TabsTrigger value="diff">Diff</TabsTrigger>
        </TabsList>

        <TabsContent value="browse" className="space-y-3">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : manifests.length === 0 ? (
            <p className="text-sm text-muted-foreground">No manifests yet. Take a snapshot to create one.</p>
          ) : (
            manifests.map((m, i) => (
              <ManifestRow key={i} m={m} onRefresh={() => setRefreshKey(k => k + 1)} />
            ))
          )}
        </TabsContent>

        <TabsContent value="snapshot">
          <Card>
            <CardHeader><CardTitle className="text-sm">Snapshot Current Corpus</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Creates a frozen, immutable record of all currently pushed documents for a use case.
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2">
                  <label className="block text-xs font-medium mb-1 text-muted-foreground">Manifest name</label>
                  <Input value={snapName} onChange={e => setSnapName(e.target.value)} placeholder="My Corpus v1.0" />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1 text-muted-foreground">Use Case ID</label>
                  <Input value={snapUcId} onChange={e => setSnapUcId(e.target.value)} placeholder="GENAI1597_SSOP" />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1 text-muted-foreground">Agent Filter</label>
                  <Input value={snapAgent} onChange={e => setSnapAgent(e.target.value)} placeholder="ssop_agent" />
                </div>
              </div>
              <Button
                onClick={doSnapshot}
                disabled={!snapName || !snapUcId || !snapAgent || snapping}
                className="w-full"
              >
                {snapping ? "Snapshotting…" : "Create Snapshot"}
              </Button>
              {snapDone && <p className="text-xs text-emerald-600">✓ Manifest created and frozen.</p>}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="diff">
          <Card>
            <CardHeader><CardTitle className="text-sm">Compare Two Manifests</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium mb-1 text-muted-foreground">Manifest A (before)</label>
                  <Input value={diffA} onChange={e => setDiffA(e.target.value)} placeholder="manifest-id-a" className="font-mono text-xs" />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1 text-muted-foreground">Manifest B (after)</label>
                  <Input value={diffB} onChange={e => setDiffB(e.target.value)} placeholder="manifest-id-b" className="font-mono text-xs" />
                </div>
              </div>
              <Button onClick={doDiff} disabled={!diffA || !diffB || diffing} className="w-full">
                {diffing ? "Comparing…" : "Compare"}
              </Button>
              {diffResult && (
                <div className="grid grid-cols-4 gap-3 text-center text-sm">
                  {[
                    { key: "added", label: "Added", color: "text-emerald-600" },
                    { key: "removed", label: "Removed", color: "text-red-600" },
                    { key: "changed", label: "Changed", color: "text-amber-600" },
                    { key: "unchanged", label: "Unchanged", color: "text-muted-foreground" },
                  ].map(({ key, label, color }) => (
                    <div key={key} className="border rounded p-3">
                      <p className={cn("text-xl font-bold", color)}>
                        {String((diffResult[key] as unknown[] ?? []).length)}
                      </p>
                      <p className="text-xs text-muted-foreground">{label}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
