import { scoreBg } from "@/lib/utils"

export default function QualityBadge({ score }: { score: number }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold ${scoreBg(score)}`}>
      {Math.round(score * 100)}%
    </span>
  )
}
