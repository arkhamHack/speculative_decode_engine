export default function SpecTimeline({ data, k }) {
  if (!data?.length) {
    return (
      <div className="card flex items-center justify-center h-36">
        <div className="text-zinc-600 text-sm">Run benchmark to see per-iteration acceptance</div>
      </div>
    )
  }

  // Show up to 20 iterations
  const display = data.slice(0, 20)
  const maxK = k || Math.max(...display.map(d => d.proposed), 1)

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <div className="label">Per-iteration acceptance</div>
        <div className="flex items-center gap-3 text-xs text-zinc-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm bg-emerald-500" /> Accepted
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm bg-zinc-700" /> Rejected
          </span>
        </div>
      </div>

      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
        {display.map((iter) => {
          const proposed = iter.proposed || maxK
          const accepted = Math.min(iter.accepted, proposed)
          const rejected = proposed - accepted
          const accPct = (accepted / proposed) * 100

          return (
            <div key={iter.iter} className="flex items-center gap-2">
              <div className="text-xs text-zinc-600 font-mono w-8 shrink-0 text-right">
                {iter.iter}
              </div>
              <div className="flex-1 flex rounded-sm overflow-hidden h-5 gap-0.5">
                {Array.from({ length: accepted }).map((_, i) => (
                  <div
                    key={`a${i}`}
                    className="flex-1 bg-emerald-500 hover:bg-emerald-400 transition-colors"
                    title={`Token ${i + 1}: accepted`}
                  />
                ))}
                {Array.from({ length: rejected }).map((_, i) => (
                  <div
                    key={`r${i}`}
                    className="flex-1 bg-zinc-700 hover:bg-zinc-600 transition-colors"
                    title={`Token ${accepted + i + 1}: rejected`}
                  />
                ))}
              </div>
              <div className="text-xs text-zinc-500 font-mono w-8 shrink-0">
                {accepted}/{proposed}
              </div>
            </div>
          )
        })}
        {data.length > 20 && (
          <div className="text-center text-xs text-zinc-600">
            … {data.length - 20} more iterations
          </div>
        )}
      </div>
    </div>
  )
}
