import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(d: string | null | undefined): string {
  if (!d) return "—"
  return new Date(d).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  })
}

export function scoreColor(score: number): string {
  if (score >= 0.8) return "text-emerald-600"
  if (score >= 0.6) return "text-amber-600"
  return "text-red-500"
}

export function scoreBg(score: number): string {
  if (score >= 0.8) return "bg-emerald-50 text-emerald-700 border-emerald-200"
  if (score >= 0.6) return "bg-amber-50 text-amber-700 border-amber-200"
  return "bg-red-50 text-red-700 border-red-200"
}

export function driftIcon(status: string): string {
  switch (status) {
    case "current": return "✓"
    case "stale": return "⚠"
    case "deleted": return "✕"
    default: return "?"
  }
}

export function driftBadgeClass(status: string): string {
  switch (status) {
    case "current": return "bg-emerald-50 text-emerald-700 border-emerald-200"
    case "stale": return "bg-amber-50 text-amber-700 border-amber-200"
    case "deleted": return "bg-red-50 text-red-700 border-red-200"
    default: return "bg-gray-50 text-gray-500 border-gray-200"
  }
}
