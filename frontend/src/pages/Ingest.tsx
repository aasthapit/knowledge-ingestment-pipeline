import { useState, useRef } from "react"
import { useMutation } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Upload, Link, FileJson, CheckCircle, AlertTriangle } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import QualityBadge from "@/components/QualityBadge"

type IngestResult = {
  doc_id: string
  quality_score: number
  quality_passed: boolean
  quality_flags: string[]
  chunk_count: number
  tags: string[]
  schema?: string
}

function ResultCard({ result }: { result: IngestResult }) {
  const navigate = useNavigate()
  return (
    <Card className={result.quality_passed ? "border-emerald-200" : "border-amber-200"}>
      <CardContent className="p-5 space-y-3">
        <div className="flex items-center gap-3">
          {result.quality_passed
            ? <CheckCircle className="h-5 w-5 text-emerald-600" />
            : <AlertTriangle className="h-5 w-5 text-amber-500" />}
          <div>
            <p className="font-medium text-sm">
              {result.quality_passed ? "Staged and auto-approved" : "Staged — needs review"}
            </p>
            <p className="text-xs text-muted-foreground font-mono">{result.doc_id.slice(0, 16)}…</p>
          </div>
          <div className="ml-auto">
            <QualityBadge score={result.quality_score} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div><span className="text-muted-foreground">Sections:</span> {result.chunk_count}</div>
          {result.schema && <div><span className="text-muted-foreground">Schema:</span> {result.schema}</div>}
        </div>
        {result.quality_flags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {result.quality_flags.map(f => (
              <span key={f} className="rounded border border-amber-200 bg-amber-50 text-amber-700 px-2 py-0.5 text-xs">{f}</span>
            ))}
          </div>
        )}
        {result.tags.length > 0 && (
          <p className="text-xs text-muted-foreground">Tags: {result.tags.join(", ")}</p>
        )}
        <Button size="sm" variant="outline" onClick={() => navigate("/review")} className="w-full">
          Go to Review Queue →
        </Button>
      </CardContent>
    </Card>
  )
}

function MetaFields({
  tags, setTags, kbName, setKbName,
  ucId, setUcId, agent, setAgent,
  autoPush, setAutoPush,
}: {
  tags: string; setTags: (v: string) => void
  kbName: string; setKbName: (v: string) => void
  ucId: string; setUcId: (v: string) => void
  agent: string; setAgent: (v: string) => void
  autoPush: boolean; setAutoPush: (v: boolean) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-3 text-sm">
      <div className="col-span-2">
        <label className="block text-xs font-medium mb-1 text-muted-foreground">Tags (comma-separated)</label>
        <Input value={tags} onChange={e => setTags(e.target.value)} placeholder="openshift, installation" />
      </div>
      <div>
        <label className="block text-xs font-medium mb-1 text-muted-foreground">KB Name</label>
        <Input value={kbName} onChange={e => setKbName(e.target.value)} placeholder="default" />
      </div>
      <div>
        <label className="block text-xs font-medium mb-1 text-muted-foreground">Use Case ID</label>
        <Input value={ucId} onChange={e => setUcId(e.target.value)} placeholder="GENAI1597_SSOP" />
      </div>
      <div>
        <label className="block text-xs font-medium mb-1 text-muted-foreground">Agent Filter</label>
        <Input value={agent} onChange={e => setAgent(e.target.value)} placeholder="ssop_agent" />
      </div>
      <div className="flex items-center gap-2 pt-5">
        <input type="checkbox" id="auto_push" checked={autoPush} onChange={e => setAutoPush(e.target.checked)} className="h-4 w-4" />
        <label htmlFor="auto_push" className="text-xs text-muted-foreground cursor-pointer">Auto-push if quality passes</label>
      </div>
    </div>
  )
}

export default function Ingest() {
  // Shared metadata state
  const [tags, setTags] = useState("")
  const [kbName, setKbName] = useState("default")
  const [ucId, setUcId] = useState("")
  const [agent, setAgent] = useState("")
  const [autoPush, setAutoPush] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)

  // File tab
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // URL tab
  const [url, setUrl] = useState("")

  // JSONL tab
  const [jsonlFile, setJsonlFile] = useState<File | null>(null)
  const jsonlRef = useRef<HTMLInputElement>(null)

  const ingest = useMutation({
    mutationFn: async (formData: FormData) => {
      const res = await api.post<IngestResult>("/ingest/document", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      return res.data
    },
    onSuccess: data => setResult(data),
  })

  const ingestJsonl = useMutation({
    mutationFn: async (formData: FormData) => {
      const res = await api.post<IngestResult>("/ingest/jsonl", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      return res.data
    },
    onSuccess: data => setResult(data),
  })

  function buildMeta(fd: FormData) {
    if (tags) fd.append("tags", tags)
    fd.append("kb_name", kbName)
    if (ucId) fd.append("usecase_id", ucId)
    if (agent) fd.append("agent_filter", agent)
    fd.append("auto_push", String(autoPush))
  }

  function submitFile() {
    if (!file) return
    const fd = new FormData()
    fd.append("file", file)
    buildMeta(fd)
    ingest.mutate(fd)
  }

  function submitUrl() {
    if (!url.trim()) return
    const fd = new FormData()
    fd.append("url", url.trim())
    buildMeta(fd)
    ingest.mutate(fd)
  }

  function submitJsonl() {
    if (!jsonlFile) return
    const fd = new FormData()
    fd.append("file", jsonlFile)
    buildMeta(fd)
    ingestJsonl.mutate(fd)
  }

  const busy = ingest.isPending || ingestJsonl.isPending

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Add Document</h1>
        <p className="text-muted-foreground text-sm mt-1">Ingest content from a file, URL, or JSONL export</p>
      </div>

      <Tabs defaultValue="file" onValueChange={() => setResult(null)}>
        <TabsList>
          <TabsTrigger value="file"><Upload className="h-3.5 w-3.5 mr-1.5" />Upload File</TabsTrigger>
          <TabsTrigger value="url"><Link className="h-3.5 w-3.5 mr-1.5" />From URL</TabsTrigger>
          <TabsTrigger value="jsonl"><FileJson className="h-3.5 w-3.5 mr-1.5" />Bulk JSONL</TabsTrigger>
        </TabsList>

        {/* File tab */}
        <TabsContent value="file" className="space-y-4">
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); setFile(e.dataTransfer.files[0] ?? null) }}
          >
            <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            {file ? (
              <p className="text-sm font-medium">{file.name}</p>
            ) : (
              <>
                <p className="text-sm font-medium">Drop a file or click to browse</p>
                <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, PPTX, HTML, TXT, Markdown</p>
              </>
            )}
            <input ref={fileRef} type="file" className="hidden"
              accept=".pdf,.docx,.pptx,.html,.htm,.txt,.md"
              onChange={e => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <MetaFields {...{ tags, setTags, kbName, setKbName, ucId, setUcId, agent, setAgent, autoPush, setAutoPush }} />
          <Button onClick={submitFile} disabled={!file || busy} className="w-full">
            {ingest.isPending ? "Processing…" : "Ingest Document"}
          </Button>
        </TabsContent>

        {/* URL tab */}
        <TabsContent value="url" className="space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1 text-muted-foreground">Web Address</label>
            <Input value={url} onChange={e => setUrl(e.target.value)} placeholder="https://docs.example.com/guide" />
          </div>
          <MetaFields {...{ tags, setTags, kbName, setKbName, ucId, setUcId, agent, setAgent, autoPush, setAutoPush }} />
          <Button onClick={submitUrl} disabled={!url.trim() || busy} className="w-full">
            {ingest.isPending ? "Fetching…" : "Ingest URL"}
          </Button>
        </TabsContent>

        {/* JSONL tab */}
        <TabsContent value="jsonl" className="space-y-4">
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => jsonlRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); setJsonlFile(e.dataTransfer.files[0] ?? null) }}
          >
            <FileJson className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            {jsonlFile ? (
              <p className="text-sm font-medium">{jsonlFile.name}</p>
            ) : (
              <>
                <p className="text-sm font-medium">Drop a .jsonl file or click to browse</p>
                <p className="text-xs text-muted-foreground mt-1">Crawler schema, pipeline schema, or custom</p>
              </>
            )}
            <input ref={jsonlRef} type="file" className="hidden" accept=".jsonl"
              onChange={e => setJsonlFile(e.target.files?.[0] ?? null)} />
          </div>
          <MetaFields {...{ tags, setTags, kbName, setKbName, ucId, setUcId, agent, setAgent, autoPush, setAutoPush }} />
          <Button onClick={submitJsonl} disabled={!jsonlFile || busy} className="w-full">
            {ingestJsonl.isPending ? "Importing…" : "Import JSONL"}
          </Button>
        </TabsContent>
      </Tabs>

      {/* Result */}
      {(ingest.error || ingestJsonl.error) && (
        <p className="text-sm text-red-600">Error: {String((ingest.error || ingestJsonl.error) ?? "")}</p>
      )}
      {result && <ResultCard result={result} />}
    </div>
  )
}
