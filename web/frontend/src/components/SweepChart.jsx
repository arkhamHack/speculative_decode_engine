import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-xs space-y-1 shadow-xl">
      <div className="font-semibold text-zinc-200">k = {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <span className="font-mono">{p.value?.toFixed(2)}{p.dataKey === 'speedup' ? '×' : '%'}</span>
        </div>
      ))}
    </div>
  )
}

export default function SweepChart({ data, loading }) {
  if (loading) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-500 text-sm animate-pulse">Sweeping k values…</div>
      </div>
    )
  }
  if (!data?.length) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-600 text-sm">Click <span className="text-zinc-400">Sweep k</span> to find the optimal draft length</div>
      </div>
    )
  }

  const chartData = data.map(d => ({
    k: d.k,
    speedup: d.speedup,
    acceptance: (d.acceptance_rate * 100),
  }))

  // Find optimal k
  const best = chartData.reduce((a, b) => b.speedup > a.speedup ? b : a, chartData[0])

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <div className="label">Speedup vs. draft length k</div>
        <div className="text-xs text-zinc-400">
          Optimal: <span className="font-mono text-emerald-400">k = {best.k}</span>
          <span className="text-zinc-500 ml-1">({best.speedup.toFixed(2)}×)</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="k"
            label={{ value: 'k (draft tokens)', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 10 }}
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
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="speedup"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: '#3b82f6', r: 4 }}
            activeDot={{ r: 6 }}
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
  )
}
