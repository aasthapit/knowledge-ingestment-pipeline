import { useState, useEffect } from "react"
import { listLedger, listSnapshots } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import QualityBadge from "@/components/QualityBadge"
import { cn, formatDate, driftBadgeClass, driftIcon } from "@/lib/utils"

export default function Ledger() {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [docs, setDocs] = useState<Record<string, unknown>[]>([])
  const [snapshots, setSnapshots] = useState<Record<string, unknown>[]>([])
  const [docsLoading, setDocsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    listLedger()
      .then(d => { if (!cancelled) { setDocs(d as Record<string, unknown>[]); setDocsLoading(false) } })
      .catch(() => { if (!cancelled) setDocsLoading(false) })
    listSnapshots()
      .then(d => { if (!cancelled) setSnapshots(d as Record<string, unknown>[]) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const totalChunks = docs.reduce((s, d) => s + Number(d.chunk_count ?? 0), 0)

  return (
    <div className="max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Ledger</h1>
        <p className="text-muted-foreground text-sm mt-1">Permanent record of all pushed documents</p>
      </div>

      <div className="flex gap-6 text-sm">
        <div><span className="text-muted-foreground">Documents:</span> <strong>{docs.length}</strong></div>
        <div><span className="text-muted-foreground">Total chunks:</span> <strong>{totalChunks.toLocaleString()}</strong></div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Documents</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {docsLoading ? (
            <p className="text-sm text-muted-foreground p-4">Loading…</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b">
                  <tr className="text-left text-muted-foreground text-xs">
                    <th className="px-4 py-2 font-medium">Status</th>
                    <th className="px-4 py-2 font-medium">Title</th>
                    <th className="px-4 py-2 font-medium">Type</th>
                    <th className="px-4 py-2 font-medium">KB</th>
                    <th className="px-4 py-2 font-medium">Chunks</th>
                    <th className="px-4 py-2 font-medium">Quality</th>
                    <th className="px-4 py-2 font-medium">Pushed</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc, i) => {
                    const drift = String(doc.drift_status ?? "unknown")
                    return (
                      <tr key={i} className="border-b last:border-0 hover:bg-muted/40">
                        <td className="px-4 py-2">
                          <span className={cn("inline-flex items-center rounded border px-1.5 py-0.5 text-xs font-semibold", driftBadgeClass(drift))}>
                            {driftIcon(drift)}
                          </span>
                        </td>
                        <td className="px-4 py-2 font-medium max-w-xs truncate">{String(doc.title ?? "—")}</td>
                        <td className="px-4 py-2 text-muted-foreground text-xs">{String(doc.source_type ?? "—")}</td>
                        <td className="px-4 py-2 text-muted-foreground text-xs">{String(doc.kb_name ?? "—")}</td>
                        <td className="px-4 py-2">{String(doc.chunk_count ?? 0)}</td>
                        <td className="px-4 py-2"><QualityBadge score={Number(doc.quality_score ?? 0)} /></td>
                        <td className="px-4 py-2 text-muted-foreground text-xs">{formatDate(String(doc.pushed_at ?? ""))}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Push History</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {snapshots.length === 0 ? (
            <p className="text-sm text-muted-foreground p-4">No push history yet.</p>
          ) : (
            <table className="w-full text-sm">
              <thead className="border-b">
                <tr className="text-left text-muted-foreground text-xs">
                  <th className="px-4 py-2 font-medium">Snapshot</th>
                  <th className="px-4 py-2 font-medium">Docs pushed</th>
                  <th className="px-4 py-2 font-medium">Total in KB</th>
                  <th className="px-4 py-2 font-medium">Date</th>
                </tr>
              </thead>
              <tbody>
                {snapshots.map((s, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                      <button
                        onClick={() => setExpanded(e => e === String(s._id) ? null : String(s._id))}
                        className="hover:text-foreground"
                      >
                        {String(s._id ?? "").slice(0, 12)}…
                      </button>
                    </td>
                    <td className="px-4 py-2">{String(s.docs_pushed ?? 0)}</td>
                    <td className="px-4 py-2">{String(s.total_docs ?? 0)}</td>
                    <td className="px-4 py-2 text-muted-foreground text-xs">{formatDate(String(s.pushed_at ?? ""))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      {/* suppress unused warning */}
      {expanded && null}
    </div>
  )
}
