import { NavLink, Outlet } from "react-router-dom"
import {
  LayoutDashboard, Upload, GitPullRequestArrow, Search,
  Activity, BookOpen, List, Library, FolderGit2, Server,
} from "lucide-react"
import { cn } from "@/lib/utils"

const nav = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { label: "Ingestion", divider: true },
  { to: "/ingest", icon: Upload, label: "Add Document" },
  { to: "/confluence", icon: BookOpen, label: "Confluence" },
  { label: "Content", divider: true },
  { to: "/review", icon: GitPullRequestArrow, label: "Review Queue" },
  { to: "/search", icon: Search, label: "Search" },
  { label: "Management", divider: true },
  { to: "/health", icon: Activity, label: "KB Health" },
  { to: "/ledger", icon: List, label: "Ledger" },
  { to: "/corpus", icon: Library, label: "Corpus" },
  { to: "/manifests", icon: FolderGit2, label: "Manifests" },
  { label: "System", divider: true },
  { to: "/status", icon: Server, label: "Status" },
]

export default function Layout() {
  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-56 border-r bg-white flex flex-col shrink-0">
        <div className="h-14 flex items-center px-5 border-b">
          <span className="font-semibold text-sm tracking-tight">Knowledge Pipeline</span>
        </div>
        <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
          {nav.map((item, i) => {
            if (item.divider) {
              return (
                <p key={i} className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground px-2 pt-4 pb-1">
                  {item.label}
                </p>
              )
            }
            const Icon = item.icon!
            return (
              <NavLink
                key={item.to}
                to={item.to!}
                end={item.to === "/"}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-sm transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                {item.label}
              </NavLink>
            )
          })}
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
