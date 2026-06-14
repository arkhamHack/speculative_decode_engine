import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-xs space-y-1 shadow-xl">
      <div className="font-semibold text-zinc-200">k = {label}</div>
      {row?.draft_temp != null && row.draft_temp !== 1 && (
        <div className="text-zinc-400">draft temp: {row.draft_temp.toFixed(2)}</div>
      )}
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <span className="font-mono">{p.value?.toFixed(2)}{p.dataKey === 'speedup' ? '×' : p.dataKey === 'acceptance' ? '%' : ''}</span>
        </div>
      ))}
    </div>
  )
}

function MetricPill({ label, value, accent = 'text-zinc-100' }) {
  return (
    <div className="card-sm text-center space-y-1">
      <div className="label">{label}</div>
      <div className={`text-lg font-bold font-mono ${accent}`}>{value}</div>
    </div>
  )
}

export default function AutotunePanel({ data, loading, onApply }) {
  if (loading) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-500 text-sm animate-pulse">
          Running autotuner — sweeping k values…
        </div>
      </div>
    )
  }

  if (!data?.best) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-600 text-sm text-center px-4">
          Click <span className="text-zinc-400">Autotune</span> to search k ∈ [1, k_max]
          {data?.sweep_temp ? ' and draft temperature' : ''} and pick the best settings.
        </div>
      </div>
    )
  }

  const { best, grid, min_acceptance, objective, n_eligible, n_total, used_fallback } = data

  // Aggregate by k (best point per k for chart)
  const byK = {}
  for (const row of grid) {
    const metric = objective === 'speedup' ? row.speedup : row.spec_tok_per_s
    if (!byK[row.k] || metric > (objective === 'speedup' ? byK[row.k].speedup : byK[row.k].spec_tok_per_s)) {
      byK[row.k] = row
    }
  }
  const chartData = Object.values(byK)
    .sort((a, b) => a.k - b.k)
    .map(d => ({
      k: d.k,
      speedup: d.speedup,
      acceptance: d.acceptance_rate * 100,
      draft_temp: d.draft_temp,
    }))

  const metricLabel = objective === 'speedup' ? 'Speedup' : 'Tok/s'

  return (
    <div className="space-y-4">
      {/* Best config banner */}
      <div className="card border-emerald-500/30 bg-emerald-500/5 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="label text-emerald-400">Recommended configuration</div>
          {used_fallback && (
            <span className="text-[11px] text-amber-400 border border-amber-500/30 rounded px-2 py-0.5">
              No config met α ≥ {(min_acceptance * 100).toFixed(0)}% — showing best overall
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricPill label="k (draft depth)" value={best.k} accent="text-emerald-400" />
          <MetricPill
            label="Draft temp"
            value={best.draft_temp?.toFixed(2) ?? '1.00'}
          />
          <MetricPill
            label="Speedup"
            value={`${best.speedup?.toFixed(2)}×`}
            accent="text-blue-400"
          />
          <MetricPill
            label="Acceptance"
            value={`${Math.round((best.acceptance_rate ?? 0) * 100)}%`}
          />
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="text-xs text-zinc-500">
            Optimized for <span className="text-zinc-300">{metricLabel}</span>
            {' '}with α ≥ {(min_acceptance * 100).toFixed(0)}%
            {' '}({n_eligible}/{n_total} configs eligible)
            {' '}· {best.spec_tok_per_s?.toFixed(0)} tok/s
          </div>
          {onApply && (
            <button
              type="button"
              onClick={() => onApply(best)}
              className="btn-primary text-xs px-3 py-1.5"
            >
              Apply to config
            </button>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="card space-y-3">
        <div className="label">Search results by k</div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="k"
              tick={{ fill: '#71717a', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              yAxisId="left"
              tick={{ fill: '#71717a', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              domain={[0.8, 'auto']}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: '#71717a', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              domain={[0, 100]}
              unit="%"
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#a1a1aa' }}
              formatter={v => v === 'speedup' ? 'Speedup (×)' : 'Accept rate (%)'}
            />
            <ReferenceLine
              yAxisId="left"
              x={best.k}
              stroke="#22c55e"
              strokeDasharray="4 3"
              label={{ value: `k*=${best.k}`, fill: '#22c55e', fontSize: 10, position: 'top' }}
            />
            <ReferenceLine
              yAxisId="right"
              y={min_acceptance * 100}
              stroke="#71717a"
              strokeDasharray="2 4"
              label={{ value: `α≥${(min_acceptance * 100).toFixed(0)}%`, fill: '#71717a', fontSize: 9, position: 'insideTopRight' }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="speedup"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', r: 4 }}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="acceptance"
              stroke="#a855f7"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={{ fill: '#a855f7', r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Grid table */}
      <div className="card overflow-x-auto">
        <div className="label mb-2">Full search grid</div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-800">
              <th className="text-left py-2 pr-3">k</th>
              <th className="text-left py-2 pr-3">Temp</th>
              <th className="text-right py-2 pr-3">Speedup</th>
              <th className="text-right py-2 pr-3">Accept</th>
              <th className="text-right py-2">Tok/s</th>
            </tr>
          </thead>
          <tbody>
            {grid.map((row, i) => {
              const isBest = row.k === best.k && row.draft_temp === best.draft_temp
              const meets = row.acceptance_rate >= min_acceptance
              return (
                <tr
                  key={i}
                  className={`border-b border-zinc-800/50 ${
                    isBest ? 'bg-emerald-500/10' : meets ? '' : 'opacity-50'
                  }`}
                >
                  <td className="py-1.5 pr-3 font-mono text-zinc-200">{row.k}</td>
                  <td className="py-1.5 pr-3 font-mono text-zinc-400">{row.draft_temp?.toFixed(2)}</td>
                  <td className="py-1.5 pr-3 text-right font-mono text-blue-400">{row.speedup?.toFixed(2)}×</td>
                  <td className="py-1.5 pr-3 text-right font-mono">{Math.round(row.acceptance_rate * 100)}%</td>
                  <td className="py-1.5 text-right font-mono text-zinc-300">{row.spec_tok_per_s?.toFixed(0)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
