import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList,
} from 'recharts'

const COLORS = ['#52525b', '#3b82f6', '#6366f1', '#22c55e']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-xs space-y-1 shadow-xl">
      <div className="font-semibold text-zinc-200">{label}</div>
      <div className="text-zinc-400">
        <span className="text-blue-400 font-mono">{d.spec_tok_per_s?.toFixed(0)}</span> tok/s
      </div>
      <div className="text-zinc-400">
        Speedup: <span className="text-emerald-400 font-mono">{d.speedup?.toFixed(2)}×</span>
      </div>
      {d.acceptance_rate != null && d.acceptance_rate > 0 && (
        <div className="text-zinc-400">
          Accept: <span className="text-purple-400 font-mono">{(d.acceptance_rate * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  )
}

export default function CompareChart({ data, loading }) {
  if (loading) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-500 text-sm animate-pulse">Running all modes…</div>
      </div>
    )
  }
  if (!data?.length) {
    return (
      <div className="card flex items-center justify-center h-52">
        <div className="text-zinc-600 text-sm">Click <span className="text-zinc-400">Compare All Modes</span></div>
      </div>
    )
  }

  const chartData = data.map((d, i) => ({
    ...d,
    label: d.label,
    spec_tok_per_s: d.spec_tok_per_s,
    speedup: d.speedup,
    acceptance_rate: d.acceptance_rate,
    fill: COLORS[i % COLORS.length],
  }))

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <div className="label">Mode comparison — throughput (tok/s)</div>
        <div className="text-xs text-zinc-500">All 4 combinations</div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: '#a1a1aa', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#71717a', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: '#ffffff08' }} />
          <Bar dataKey="spec_tok_per_s" radius={[4, 4, 0, 0]} maxBarSize={56}>
            {chartData.map((d, i) => (
              <Cell key={i} fill={d.fill} />
            ))}
            <LabelList
              dataKey="speedup"
              position="top"
              formatter={v => v > 1 ? `${v.toFixed(2)}×` : ''}
              style={{ fill: '#a1a1aa', fontSize: 10 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
