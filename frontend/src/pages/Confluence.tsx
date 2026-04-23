import { useState, useRef } from "react"
import { BookOpen, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"

type SseEvent = { type: "progress" | "done" | "error"; message?: string; result?: Record<string, unknown> }

export default function Confluence() {
  const [baseUrl, setBaseUrl] = useState("")
  const [authType, setAuthType] = useState<"cloud" | "server">("cloud")
  const [email, setEmail] = useState("")
  const [token, setToken] = useState("")
  const [sslVerify, setSslVerify] = useState(false)
  const [pageUrl, setPageUrl] = useState("")
  const [maxDepth, setMaxDepth] = useState(-1)
  const [kbName, setKbName] = useState("default")
  const [tags, setTags] = useState("")
  const [ucId, setUcId] = useState("")
  const [agent, setAgent] = useState("")

  const [logs, setLogs] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState("")
  const abortRef = useRef<(() => void) | null>(null)

  function start() {
    setLogs([])
    setDone(null)
    setError("")
    setRunning(true)

    const payload = {
      base_url: baseUrl, auth_type: authType, email, api_token: token,
      ssl_verify: sslVerify, page_url: pageUrl, max_depth: maxDepth,
      kb_name: kbName,
      tags: tags.split(",").map(t => t.trim()).filter(Boolean),
      usecase_id: ucId || undefined,
      agent_filter: agent || undefined,
    }

    fetch("/api/confluence/crawl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(res => {
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ""

      abortRef.current = () => reader.cancel()

      function pump(): Promise<void> {
        return reader.read().then(({ done: d, value }) => {
          if (d) { setRunning(false); return }
          buf += decoder.decode(value, { stream: true })
          const parts = buf.split("\n\n")
          buf = parts.pop() ?? ""
          parts.forEach(part => {
            const line = part.replace(/^data: /, "")
            if (!line) return
            try {
              const ev: SseEvent = JSON.parse(line)
              if (ev.type === "progress") setLogs(l => [...l, ev.message ?? ""])
              if (ev.type === "done") { setDone(ev.result ?? {}); setRunning(false) }
              if (ev.type === "error") { setError(ev.message ?? "Unknown error"); setRunning(false) }
            } catch { /* ignore */ }
          })
          return pump()
        })
      }
      return pump()
    }).catch(e => { setError(String(e)); setRunning(false) })
  }

  const canStart = baseUrl && pageUrl && (authType === "cloud" ? (email && token) : token)

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Confluence</h1>
        <p className="text-muted-foreground text-sm mt-1">Crawl a Confluence page tree and stage it for review</p>
      </div>

      <div className="space-y-4">
        <h2 className="text-sm font-semibold">Connection</h2>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Base URL</label>
            <Input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://mycompany.atlassian.net" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Auth type</label>
            <select
              value={authType}
              onChange={e => setAuthType(e.target.value as "cloud" | "server")}
              className="h-9 w-full rounded-md border border-input px-3 text-sm bg-transparent"
            >
              <option value="cloud">Cloud (email + API token)</option>
              <option value="server">Server/DC (Personal Access Token)</option>
            </select>
          </div>
          <div className="flex items-end gap-2">
            <input type="checkbox" id="ssl" checked={sslVerify} onChange={e => setSslVerify(e.target.checked)} className="h-4 w-4 mb-2" />
            <label htmlFor="ssl" className="text-xs text-muted-foreground mb-2 cursor-pointer">Verify SSL certificate</label>
          </div>
          {authType === "cloud" && (
            <div>
              <label className="block text-xs font-medium mb-1 text-muted-foreground">Email</label>
              <Input value={email} onChange={e => setEmail(e.target.value)} placeholder="user@example.com" />
            </div>
          )}
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">
              {authType === "cloud" ? "API Token" : "Personal Access Token"}
            </label>
            <Input type="password" value={token} onChange={e => setToken(e.target.value)} placeholder="••••••••" />
          </div>
        </div>

        <h2 className="text-sm font-semibold pt-2">Page Selection</h2>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Parent Page URL</label>
            <Input value={pageUrl} onChange={e => setPageUrl(e.target.value)} placeholder="https://…/wiki/spaces/SPACE/pages/123456" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Max depth (−1 = unlimited)</label>
            <Input type="number" value={maxDepth} onChange={e => setMaxDepth(Number(e.target.value))} />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">KB Name</label>
            <Input value={kbName} onChange={e => setKbName(e.target.value)} placeholder="default" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Tags</label>
            <Input value={tags} onChange={e => setTags(e.target.value)} placeholder="confluence, docs" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Use Case ID</label>
            <Input value={ucId} onChange={e => setUcId(e.target.value)} placeholder="optional" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Agent Filter</label>
            <Input value={agent} onChange={e => setAgent(e.target.value)} placeholder="optional" />
          </div>
        </div>

        <Button onClick={start} disabled={!canStart || running} className="w-full gap-2">
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <BookOpen className="h-4 w-4" />}
          {running ? "Crawling…" : "Start Crawl"}
        </Button>
      </div>

      {/* Progress log */}
      {(logs.length > 0 || error) && (
        <Card className={error ? "border-red-200" : ""}>
          <CardContent className="p-4">
            <div className="space-y-1 max-h-48 overflow-y-auto font-mono text-xs">
              {logs.map((l, i) => <p key={i} className="text-muted-foreground">{l}</p>)}
              {error && <p className="text-red-600">{error}</p>}
            </div>
          </CardContent>
        </Card>
      )}

      {done && (
        <Card className="border-emerald-200">
          <CardContent className="p-4 text-sm space-y-1">
            <p className="font-medium text-emerald-700">✓ Crawl complete</p>
            <p>{String(done.pages ?? 0)} pages staged · doc ID: <span className="font-mono text-xs">{String(done.doc_id ?? "")}</span></p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
