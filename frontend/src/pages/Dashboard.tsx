import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { Upload, GitPullRequestArrow, Search, Clock } from "lucide-react"
import { getStats } from "@/lib/api"
import StatCard from "@/components/StatCard"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { formatDate } from "@/lib/utils"

export default function Dashboard() {
  const navigate = useNavigate()
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    function load() {
      getStats()
        .then(d => { if (!cancelled) { setData(d); setLoading(false) } })
        .catch(() => { if (!cancelled) setLoading(false) })
    }
    load()
    const id = setInterval(load, 15000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  return (
    <div className="max-w-5xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">Knowledge base health at a glance</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Indexed Sections"
          value={loading ? "—" : ((data?.indexed_chunks as number) ?? 0).toLocaleString()}
          sub="searchable in Redis"
        />
        <StatCard
          label="Needs Review"
          value={loading ? "—" : (data?.pending_review as number) ?? 0}
          sub="awaiting approval"
          className={data?.pending_review ? "border-amber-200" : ""}
          onClick={() => navigate("/review")}
        />
        <StatCard
          label="Ready to Push"
          value={loading ? "—" : (data?.approved as number) ?? 0}
          sub="approved, not yet live"
          className={data?.approved ? "border-blue-200" : ""}
          onClick={() => navigate("/review")}
        />
        <StatCard
          label="Pushed Today"
          value={loading ? "—" : (data?.pushed_today as number) ?? 0}
          sub="sections indexed today"
        />
      </div>

      <div>
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Quick Actions</h2>
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => navigate("/ingest")} className="gap-2">
            <Upload className="h-4 w-4" /> Add Document
          </Button>
          <Button variant="outline" onClick={() => navigate("/review")} className="gap-2">
            <GitPullRequestArrow className="h-4 w-4" />
            Review Queue
            {data?.pending_review ? (
              <span className="ml-1 rounded-full bg-amber-100 text-amber-800 text-xs px-1.5 py-0.5 font-semibold">
                {data.pending_review as number}
              </span>
            ) : null}
          </Button>
          <Button variant="outline" onClick={() => navigate("/search")} className="gap-2">
            <Search className="h-4 w-4" /> Search KB
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Clock className="h-4 w-4" /> Recent Push Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!(data?.recent_pushes as unknown[] | undefined)?.length ? (
            <p className="text-sm text-muted-foreground">No push history yet.</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted-foreground border-b">
                  <th className="pb-2 font-medium">Snapshot</th>
                  <th className="pb-2 font-medium">Docs pushed</th>
                  <th className="pb-2 font-medium">Total in KB</th>
                  <th className="pb-2 font-medium">Date</th>
                </tr>
              </thead>
              <tbody>
                {(data!.recent_pushes as Record<string, unknown>[]).map((s, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-2 font-mono text-xs text-muted-foreground">{String(s._id ?? "").slice(0, 12)}…</td>
                    <td className="py-2">{String(s.docs_pushed ?? 0)}</td>
                    <td className="py-2">{String(s.total_docs ?? 0)}</td>
                    <td className="py-2 text-muted-foreground">{formatDate(String(s.pushed_at ?? ""))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
