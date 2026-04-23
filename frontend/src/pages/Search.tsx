import { useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Search as SearchIcon, ExternalLink } from "lucide-react"
import { search, listUsecases, listAgents } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { cn, scoreBg } from "@/lib/utils"

type Result = {
  chunk_id: string
  content: string
  source: string
  title: string
  section: string
  tags: string[]
  score: number
  page_number?: number
}

function ScoreBadge({ score }: { score: number }) {
  const label = score >= 0.8 ? "Highly relevant" : score >= 0.6 ? "Relevant" : "Possibly relevant"
  return (
    <span className={cn("inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold", scoreBg(score))}>
      {Math.round(score * 100)}% · {label}
    </span>
  )
}

export default function SearchPage() {
  const [query, setQuery] = useState("")
  const [topK, setTopK] = useState(5)
  const [ucId, setUcId] = useState("all")
  const [agent, setAgent] = useState("all")
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const { data: usecases = [] } = useQuery({ queryKey: ["usecases"], queryFn: listUsecases })
  const { data: agents = [] } = useQuery({
    queryKey: ["agents", ucId],
    queryFn: () => listAgents(ucId),
    enabled: ucId !== "all",
  })

  const doSearch = useMutation({
    mutationFn: () => search({
      query,
      top_k: topK,
      usecase_id: ucId !== "all" ? ucId : undefined,
      agent_filter: agent !== "all" ? agent : undefined,
    }),
  })

  const results: Result[] = (doSearch.data as Result[]) ?? []

  function toggle(id: string) {
    setExpanded(s => {
      const n = new Set(s)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Search</h1>
        <p className="text-muted-foreground text-sm mt-1">Semantic search across the knowledge base</p>
      </div>

      {/* Search bar */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && query.trim() && doSearch.mutate()}
            placeholder="Ask a question or search by keyword…"
            className="pl-9"
          />
        </div>
        <Button onClick={() => doSearch.mutate()} disabled={!query.trim() || doSearch.isPending}>
          {doSearch.isPending ? "Searching…" : "Search"}
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground whitespace-nowrap">Use case</label>
          <Select value={ucId} onValueChange={v => { setUcId(v); setAgent("all") }}>
            <SelectTrigger className="h-8 w-44 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All use cases</SelectItem>
              {usecases.map((u: string) => <SelectItem key={u} value={u}>{u}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        {ucId !== "all" && (
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground whitespace-nowrap">Agent</label>
            <Select value={agent} onValueChange={setAgent}>
              <SelectTrigger className="h-8 w-44 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All agents</SelectItem>
                {agents.map((a: string) => <SelectItem key={a} value={a}>{a}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        )}
        <div className="flex items-center gap-2 ml-auto">
          <label className="text-xs text-muted-foreground">Results</label>
          <select
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="h-8 rounded-md border border-input px-2 text-xs bg-transparent"
          >
            {[3, 5, 10, 20].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>

      {/* Results */}
      {doSearch.isPending && <p className="text-sm text-muted-foreground">Searching…</p>}
      {!doSearch.isPending && doSearch.isSuccess && results.length === 0 && (
        <p className="text-sm text-muted-foreground">No results found.</p>
      )}

      <div className="space-y-3">
        {results.map(r => {
          const isExpanded = expanded.has(r.chunk_id)
          const preview = r.content.slice(0, 600)
          const hasMore = r.content.length > 600

          return (
            <Card key={r.chunk_id} className="overflow-hidden">
              <CardContent className="p-4 space-y-2">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium leading-tight">
                      {r.section.split(" > ").map((seg, i, arr) =>
                        i === arr.length - 1
                          ? <span key={i}>{seg}</span>
                          : <span key={i} className="text-muted-foreground">{seg} › </span>
                      )}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {r.title}{r.page_number != null ? ` · p.${r.page_number}` : ""}
                    </p>
                  </div>
                  <ScoreBadge score={r.score} />
                </div>

                <p className="text-sm text-foreground/80 whitespace-pre-wrap">
                  {isExpanded ? r.content : preview}
                  {!isExpanded && hasMore && "…"}
                </p>

                <div className="flex items-center justify-between gap-2 pt-1">
                  <div className="flex items-center gap-2">
                    {r.source && (
                      <a
                        href={r.source.startsWith("http") ? r.source : undefined}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
                      >
                        {r.source.startsWith("http") ? <ExternalLink className="h-3 w-3" /> : null}
                        {r.source.length > 50 ? `…${r.source.slice(-40)}` : r.source}
                      </a>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {r.tags.slice(0, 3).map(t => (
                      <span key={t} className="rounded bg-muted text-muted-foreground text-xs px-1.5 py-0.5">{t}</span>
                    ))}
                    {hasMore && (
                      <button
                        onClick={() => toggle(r.chunk_id)}
                        className="text-xs text-primary hover:underline"
                      >
                        {isExpanded ? "Show less" : "Show more"}
                      </button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
