import { CheckCircle2, XCircle, AlertCircle, Monitor } from 'lucide-react'

function MetricCard({ label, value, unit, color = 'text-zinc-100', sub }) {
  return (
    <div className="card-sm space-y-1">
      <div className="label">{label}</div>
      <div className={`metric-value ${color}`}>
        {value}
        {unit && <span className="text-base font-normal text-zinc-500 ml-1">{unit}</span>}
      </div>
      {sub && <div className="text-xs text-zinc-500">{sub}</div>}
    </div>
  )
}

function SpeedupBadge({ value }) {
  const color = value >= 1.5 ? 'text-emerald-400' : value >= 1.1 ? 'text-blue-400' : 'text-zinc-400'
  const arrow = value >= 1.1 ? '▲' : value < 0.99 ? '▼' : '─'
  return (
    <div className="card-sm flex flex-col items-center justify-center text-center">
      <div className="label">Speedup</div>
      <div className={`text-4xl font-bold font-mono tabular-nums ${color}`}>
        {value.toFixed(2)}×
      </div>
      <div className={`text-sm mt-1 ${color}`}>{arrow} vs baseline</div>
    </div>
  )
}

function TokenBar({ label, tokens, highlight = false }) {
  const display = tokens.slice(0, 32)
  return (
    <div className="space-y-1">
      <div className="label">{label}</div>
      <div className={`font-mono text-xs p-2 rounded-lg flex flex-wrap gap-1 max-h-16 overflow-hidden ${
        highlight ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-zinc-800/60'
      }`}>
        {display.map((t, i) => (
          <span key={i} className="text-zinc-300">{t}</span>
        ))}
        {tokens.length > 32 && <span className="text-zinc-600">…</span>}
      </div>
    </div>
  )
}

function AcceptanceBar({ rate, proposed, accepted }) {
  const pct = Math.round(rate * 100)
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 50 ? 'bg-blue-500' : 'bg-amber-500'
  return (
    <div className="card-sm space-y-2">
      <div className="flex justify-between items-center">
        <div className="label">Acceptance Rate</div>
        <div className="font-mono text-sm font-bold text-zinc-200">{pct}%</div>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-zinc-500">
        <span>{accepted} accepted</span>
        <span>{proposed} proposed</span>
      </div>
    </div>
  )
}

export default function ResultsPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="inline-block w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <div className="text-zinc-400 text-sm">Running on GPU…</div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3 text-zinc-600">
          <AlertCircle size={40} className="mx-auto opacity-40" />
          <div className="text-sm">Configure and click <span className="text-zinc-400">Run Benchmark</span></div>
        </div>
      </div>
    )
  }

  const isMock = result.device?.includes('mock')
  const isProd = result.production

  return (
    <div className="flex-1 space-y-4 overflow-y-auto">
      {/* Device + mode header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <Monitor size={14} />
          <span className="font-mono">{result.device || 'Unknown device'}</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {isProd && (
            <span className="badge-mock border-emerald-500/30 text-emerald-400 bg-emerald-500/10">production</span>
          )}
          {isMock && <span className="badge-mock">mock data</span>}
          {result.match != null && (
            result.match
              ? <span className="badge-pass"><CheckCircle2 size={11} /> Output match PASS</span>
              : <span className="badge-fail"><XCircle size={11} /> Output mismatch!</span>
          )}
        </div>
      </div>

      {/* Top metrics row */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Baseline"
          value={result.baseline_tok_per_s?.toFixed(0)}
          unit="tok/s"
          sub={`${result.baseline_ms?.toFixed(1)} ms`}
        />
        <MetricCard
          label="Speculative"
          value={result.spec_tok_per_s?.toFixed(0)}
          unit="tok/s"
          color="text-emerald-400"
          sub={`${result.spec_ms?.toFixed(1)} ms`}
        />
      </div>

      {/* Speedup + iterations */}
      <div className="grid grid-cols-2 gap-3">
        <SpeedupBadge value={result.speedup ?? 1} />
        <MetricCard
          label="Spec iterations"
          value={result.spec_iterations ?? '–'}
          color="text-purple-400"
          sub={`${result.spec_n ?? 0} tokens total`}
        />
      </div>

      {/* Acceptance rate */}
      {result.draft_proposed > 0 && (
        <AcceptanceBar
          rate={result.acceptance_rate ?? 0}
          proposed={result.draft_proposed}
          accepted={result.draft_accepted}
        />
      )}

      {/* Timing bars */}
      <div className="card space-y-3">
        <div className="label">Relative latency</div>
        {[
          { label: 'Baseline', ms: result.baseline_ms, color: 'bg-zinc-500' },
          { label: 'Speculative', ms: result.spec_ms, color: 'bg-emerald-500' },
        ].map(({ label, ms, color }) => {
          const maxMs = Math.max(result.baseline_ms, result.spec_ms, 1)
          const pct = (ms / maxMs) * 100
          return (
            <div key={label} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-400">{label}</span>
                <span className="font-mono text-zinc-300">{ms?.toFixed(1)} ms</span>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${color}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Decoded text (production) */}
      {(result.production || result.decode_error) ? (
        <div className="card space-y-3">
          <div className="label">Decoded text (HF tokenizer)</div>
          {result.prompt_preview ? (
            <div className="text-xs text-zinc-500 border-l-2 border-zinc-600 pl-2">
              <span className="text-zinc-500">Prompt · </span>
              <span className="text-zinc-300 whitespace-pre-wrap">{result.prompt_preview}</span>
            </div>
          ) : null}
          {result.decode_error && (
            <div className="text-xs text-amber-400">Decode error: {result.decode_error}</div>
          )}
          <div className="space-y-1">
            <div className="text-xs text-zinc-500">Baseline continuation</div>
            <p className="text-sm text-zinc-200 whitespace-pre-wrap leading-relaxed bg-zinc-800/60 rounded-lg p-3 border border-zinc-700/80">
              {result.baseline_text || '—'}
            </p>
          </div>
          <div className="space-y-1">
            <div className="text-xs text-zinc-500">Speculative continuation</div>
            <p className="text-sm text-zinc-200 whitespace-pre-wrap leading-relaxed bg-blue-500/5 rounded-lg p-3 border border-blue-500/25">
              {result.spec_text || '—'}
            </p>
          </div>
          <p className="text-xs text-zinc-600">
            Greedy decoding with Llama-style RoPE on Q/K (SDEC v2{' '}
            <code className="text-zinc-500">rope_theta</code>). HF may still differ slightly on very long contexts if{' '}
            <code className="text-zinc-500">rope_scaling</code> is enabled there.
          </p>
        </div>
      ) : null}

      {/* Token output */}
      <div className="card space-y-3">
        <TokenBar label="Baseline output" tokens={result.baseline_tokens ?? []} />
        <TokenBar label="Speculative output" tokens={result.spec_tokens ?? []} highlight />
      </div>
    </div>
  )
}
