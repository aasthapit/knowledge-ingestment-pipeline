import { useState, useEffect } from "react"
import { getStatus } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import StatusDot from "@/components/StatusDot"

export default function Status() {
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    function load() {
      getStatus()
        .then(d => { if (!cancelled) { setData(d); setLoading(false) } })
        .catch(() => { if (!cancelled) setLoading(false) })
    }
    load()
    const id = setInterval(load, 30000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (loading) return <p className="text-sm text-muted-foreground">Checking connections…</p>

  const cfg = (data?.config ?? {}) as Record<string, unknown>
  const kb = (data?.kb_stats ?? {}) as Record<string, unknown>

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Status</h1>
        <p className="text-muted-foreground text-sm mt-1">Service connections and configuration</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { key: "redis", label: "Redis Stack" },
          { key: "mongodb", label: "MongoDB" },
          { key: "embeddings", label: "Embeddings" },
        ].map(({ key, label }) => {
          const svc = data?.[key] as Record<string, unknown> | undefined
          return (
            <Card key={key}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <StatusDot ok={svc?.ok as boolean ?? false} />
                  {label}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-xs text-muted-foreground space-y-1">
                {svc?.url ? <p className="font-mono truncate">{String(svc.url)}</p> : null}
                {svc?.detail ? <p className="text-red-500">{String(svc.detail)}</p> : null}
                {svc?.ok ? <p className="text-emerald-600">Connected</p> : null}
              </CardContent>
            </Card>
          )
        })}
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Knowledge Base</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {[
              { label: "Total staged", value: kb.total_staged },
              { label: "Pending review", value: kb.pending },
              { label: "Approved", value: kb.approved },
              { label: "Pushed", value: kb.pushed },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-muted-foreground text-xs">{label}</p>
                <p className="font-semibold text-lg">{(value as number) ?? 0}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
            {Object.entries(cfg).map(([k, v]) => (
              <div key={k} className="flex justify-between py-1 border-b last:border-0">
                <span className="text-muted-foreground text-xs">{k.replace(/_/g, " ")}</span>
                <span className="font-mono text-xs">{String(v)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
