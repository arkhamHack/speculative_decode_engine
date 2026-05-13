import { Cpu, Zap, Settings2 } from 'lucide-react'

function Toggle({ value, onChange, label, description }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <div className="text-sm font-medium text-zinc-200">{label}</div>
        {description && <div className="text-xs text-zinc-500 mt-0.5">{description}</div>}
      </div>
      <button
        onClick={() => onChange(!value)}
        className={`toggle ${value ? 'bg-blue-600' : 'bg-zinc-700'}`}
        role="switch"
        aria-checked={value}
      >
        <span className={`toggle-thumb ${value ? 'translate-x-6' : 'translate-x-1'}`} />
      </button>
    </div>
  )
}

function Slider({ label, value, min, max, step = 1, onChange, format }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <div className="label">{label}</div>
        <div className="font-mono text-sm font-semibold text-blue-400">
          {format ? format(value) : value}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="slider"
      />
      <div className="flex justify-between text-xs text-zinc-600">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

function ModeSelect({ value, onChange }) {
  const modes = [
    { id: 'multi', label: 'Multi-kernel', icon: '⚙️', desc: 'One launch per op' },
    { id: 'mega',  label: 'Megakernel',   icon: '⚡', desc: 'Single persistent launch' },
  ]
  return (
    <div className="space-y-2">
      <div className="label">Kernel Mode</div>
      <div className="grid grid-cols-2 gap-2">
        {modes.map(m => (
          <button
            key={m.id}
            onClick={() => onChange(m.id)}
            className={`p-3 rounded-lg border text-left transition-all duration-150 ${
              value === m.id
                ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                : 'border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-600'
            }`}
          >
            <div className="text-base mb-0.5">{m.icon}</div>
            <div className="text-xs font-semibold">{m.label}</div>
            <div className="text-xs text-zinc-500">{m.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}

export default function ConfigPanel({ config, onChange, onRun, onCompare, onSweep, loading }) {
  const set = (key) => (val) => onChange({ ...config, [key]: val })

  return (
    <aside className="flex flex-col gap-4 w-72 shrink-0">
      {/* Header */}
      <div className="flex items-center gap-2 pb-1">
        <Settings2 size={18} className="text-blue-400" />
        <span className="font-semibold text-zinc-200">Configuration</span>
      </div>

      {/* Kernel mode */}
      <div className="card space-y-4">
        <ModeSelect value={config.mode} onChange={set('mode')} />
        <Toggle
          value={config.spec}
          onChange={set('spec')}
          label="Speculative Decoding"
          description="Draft + target model"
        />
      </div>

      {/* Parameters */}
      <div className="card space-y-5">
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-zinc-400" />
          <span className="label">Parameters</span>
        </div>

        <Slider label="Draft tokens k" value={config.k} min={1} max={8} onChange={set('k')}
          format={v => `k = ${v}`} />
        <Slider label="Max tokens" value={config.maxTokens} min={8} max={128} step={8} onChange={set('maxTokens')} />
        <Slider label="Prompt length" value={config.promptLen} min={1} max={16} onChange={set('promptLen')} />

        <div className="space-y-2">
          <div className="label">Seed</div>
          <input
            type="number"
            value={config.seed}
            min={0}
            onChange={e => set('seed')(Number(e.target.value))}
            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm font-mono text-zinc-200 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      {/* Actions */}
      <div className="space-y-2">
        <button
          onClick={onRun}
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          <Zap size={15} />
          {loading ? 'Running…' : 'Run Benchmark'}
        </button>
        <button onClick={onCompare} disabled={loading} className="btn-secondary w-full">
          Compare All Modes
        </button>
        <button onClick={onSweep} disabled={loading} className="btn-secondary w-full">
          Sweep k (1 → {config.k})
        </button>
      </div>
    </aside>
  )
}
