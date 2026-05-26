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

function WeightSourcePicker({ production, onChange }) {
  return (
    <div className="space-y-2">
      <div className="label">Weight source</div>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onChange(false)}
          className={`rounded-lg border px-3 py-2.5 text-left transition-all duration-150 ${
            !production
              ? 'border-blue-500 bg-blue-500/15 text-blue-300 ring-1 ring-blue-500/40'
              : 'border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-600'
          }`}
        >
          <div className="text-xs font-semibold text-zinc-200">Dummy bench</div>
          <div className="text-[11px] text-zinc-500 mt-0.5 leading-snug">
            Random weights + synthetic prompt (seed / prompt length)
          </div>
        </button>
        <button
          type="button"
          onClick={() => onChange(true)}
          className={`rounded-lg border px-3 py-2.5 text-left transition-all duration-150 ${
            production
              ? 'border-emerald-500 bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/40'
              : 'border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-600'
          }`}
        >
          <div className="text-xs font-semibold text-zinc-200">Production</div>
          <div className="text-[11px] text-zinc-500 mt-0.5 leading-snug">
            SDEC <code className="text-zinc-400">.bin</code> + HF tokenizer + prompt
          </div>
        </button>
      </div>
    </div>
  )
}

export default function ConfigPanel({ config, onChange, onRun, onCompare, onSweep, loading }) {
  const set = (key) => (val) => onChange({ ...config, [key]: val })

  return (
    <aside className="flex flex-col gap-4 w-72 shrink-0 min-h-0 overflow-y-auto pr-1 pb-2">
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
        <Slider label="Max tokens" value={config.maxTokens} min={8} max={512} step={8} onChange={set('maxTokens')} />
        {!config.production && (
          <>
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
          </>
        )}
      </div>

      {/* Stochastic spec decode */}
      {config.spec && (
        <div className="card space-y-4">
          <div className="flex items-center gap-2">
            <Zap size={14} className="text-purple-400" />
            <span className="label">Sampling Mode</span>
          </div>

          <Toggle
            value={config.stochastic}
            onChange={set('stochastic')}
            label="Stochastic spec decode"
            description="Exact paper algorithm (Leviathan et al.) — preserves target distribution"
          />

          {config.stochastic && (
            <div className="space-y-4 pl-1 border-l-2 border-purple-800/60">
              <Slider
                label="Draft temperature"
                value={config.draftTemp}
                min={0.1}
                max={2.0}
                step={0.05}
                onChange={set('draftTemp')}
                format={v => v.toFixed(2)}
              />

              <Toggle
                value={config.adaptiveDraftTemp}
                onChange={set('adaptiveDraftTemp')}
                label="Adaptive temperature"
                description="EWMA-nudge draft temp toward target acceptance rate"
              />

              {config.adaptiveDraftTemp && (
                <div className="space-y-3 pl-1 border-l border-zinc-700">
                  <Slider
                    label="Accept target α"
                    value={config.adaptAccept}
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    onChange={set('adaptAccept')}
                    format={v => v.toFixed(2)}
                  />
                  <Slider
                    label="Temp gain"
                    value={config.adaptGain}
                    min={0.001}
                    max={0.2}
                    step={0.001}
                    onChange={set('adaptGain')}
                    format={v => v.toFixed(3)}
                  />
                  <Slider
                    label="EWMA mix λ"
                    value={config.adaptEwma}
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    onChange={set('adaptEwma')}
                    format={v => v.toFixed(2)}
                  />
                </div>
              )}

              <div className="space-y-1">
                <div className="label">RNG seed</div>
                <input
                  type="number"
                  value={config.specSeed}
                  min={0}
                  onChange={e => set('specSeed')(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm font-mono text-zinc-200 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="text-[11px] text-zinc-500 leading-relaxed border-l-2 border-zinc-700 pl-2">
                Greedy (off): token match. Stochastic (on): accept with prob min(1, p/q).
                Use <span className="text-zinc-300">megakernel</span> for stochastic — it avoids all CPU syncs.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Production: real SDEC weights + prompt */}
      <div className="card space-y-4">
        <WeightSourcePicker
          production={config.production}
          onChange={(v) => set('production')(v)}
        />
        {config.production && (
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="label">Draft weights (.bin)</div>
              <input
                type="text"
                value={config.draftPath}
                onChange={e => set('draftPath')(e.target.value)}
                placeholder="weights/draft.bin or absolute path"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="space-y-1">
              <div className="label">Target weights (.bin)</div>
              <input
                type="text"
                value={config.targetPath}
                onChange={e => set('targetPath')(e.target.value)}
                placeholder="weights/target.bin"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="space-y-1">
              <div className="label">Tokenizer (HF model id)</div>
              <input
                type="text"
                value={config.tokenizerModel}
                onChange={e => set('tokenizerModel')(e.target.value)}
                placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-blue-500"
              />
            </div>
            <Toggle
              value={config.useChatTemplate}
              onChange={set('useChatTemplate')}
              label="Apply chat template"
              description="Wrap prompt with HF chat template (disable for raw text / base models)"
            />
            <div className="space-y-1">
              <div className="label">Prompt text</div>
              <textarea
                value={config.promptText}
                onChange={e => set('promptText')(e.target.value)}
                rows={4}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-blue-500 resize-y min-h-[88px]"
              />
            </div>
          </div>
        )}
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
