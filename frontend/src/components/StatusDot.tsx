import { cn } from "@/lib/utils"

export default function StatusDot({ ok, label }: { ok: boolean; label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("h-2 w-2 rounded-full", ok ? "bg-emerald-500" : "bg-red-500")} />
      {label && <span className="text-sm">{label}</span>}
    </span>
  )
}
