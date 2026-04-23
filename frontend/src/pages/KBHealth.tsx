import { useState, useEffect } from "react"
import { RefreshCw } from "lucide-react"
import { listLedger, runDriftCheck } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { cn, formatDate, driftBadgeClass, driftIcon } from "@/lib/utils"

const FILTERS = ["all", "current", "stale", "deleted", "unknown"] as const

export default function KBHealth() {
  const [filter, setFilter] = useState("all")
  const [kbFilter] = useState("")
  const [docs, setDocs] = useState<Record<string, unknown>[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)
  const [driftPending, setDriftPending] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    listLedger(kbFilter || undefined, filter !== "all" ? filter : undefined)
      .then(d => { if (!cancelled) { setDocs(d as Record<string, unknown>[]); setLoading(false) } })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [filter, kbFilter, refreshKey])

  async function checkDrift() {
    setDriftPending(true)
    try {
      await runDriftCheck(kbFilter || undefined)
      setRefreshKey(k => k + 1)
    } finally {
      setDriftPending(false)
    }
  }

  const counts: Record<string, number> = { current: 0, stale: 0, deleted: 0, unknown: 0 }
  docs.forEach(d => {
    const s = String(d.drift_status ?? "unknown")
    counts[s] = (counts[s] ?? 0) + 1
  })

  return (
    <div className="max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">KB Health</h1>
          <p className="text-muted-foreground text-sm mt-1">Drift detection for all pushed documents</p>
        </div>
        <Button variant="outline" onClick={checkDrift} disabled={driftPending} className="gap-2">
          <RefreshCw className={cn("h-4 w-4", driftPending && "animate-spin")} />
          {driftPending ? "Checking…" : "Check Drift"}
        </Button>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {[
          { key: "current", label: "Current", color: "text-emerald-600" },
          { key: "stale", label: "Stale", color: "text-amber-600" },
          { key: "deleted", label: "Deleted", color: "text-red-600" },
          { key: "unknown", label: "Unknown", color: "text-gray-500" },
        ].map(({ key, label, color }) => (
          <Card key={key} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => setFilter(f => f === key ? "all" : key)}>
            <CardContent className="p-4 text-center">
              <p className={cn("text-2xl font-bold", color)}>{counts[key] ?? 0}</p>
              <p className="text-xs text-muted-foreground">{label}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="flex gap-1 rounded-lg bg-muted p-1 w-fit">
        {FILTERS.map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={cn(
              "px-3 py-1 rounded-md text-sm transition-colors capitalize",
              filter === f ? "bg-white shadow text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            {f}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No documents found.</p>
      ) : (
        <div className="space-y-2">
          {docs.map((doc, i) => {
            const drift = String(doc.drift_status ?? "unknown")
            return (
              <div key={i} className="flex items-center gap-3 p-3 rounded-lg border bg-white text-sm">
                <span className={cn("inline-flex items-center rounded border px-2 py-0.5 text-xs font-semibold shrink-0", driftBadgeClass(drift))}>
                  {driftIcon(drift)} {drift}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{String(doc.title ?? "Untitled")}</p>
                  <p className="text-xs text-muted-foreground truncate">{String(doc.source_path ?? "")}</p>
                </div>
                <div className="text-xs text-muted-foreground text-right shrink-0">
                  <p>{String(doc.chunk_count ?? 0)} chunks</p>
                  <p>{formatDate(String(doc.pushed_at ?? ""))}</p>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
