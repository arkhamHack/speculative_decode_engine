import { useState, useEffect } from 'react'
import { Zap, Github, AlertTriangle } from 'lucide-react'
import ConfigPanel   from './components/ConfigPanel.jsx'
import ResultsPanel  from './components/ResultsPanel.jsx'
import CompareChart  from './components/CompareChart.jsx'
import SpecTimeline  from './components/SpecTimeline.jsx'
import SweepChart    from './components/SweepChart.jsx'

const API = ''  // proxied via Vite → localhost:8000

const DEFAULT_CONFIG = {
  mode:      'multi',
  spec:      true,
  k:         4,
  maxTokens: 32,
  promptLen: 4,
  seed:      42,
}

async function apiFetch(path, body) {
  const res = await fetch(`${API}/api/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

function TabBar({ active, onChange }) {
  const tabs = [
    { id: 'single',  label: '⚡ Single run' },
    { id: 'compare', label: '📊 Compare modes' },
    { id: 'sweep',   label: '🔍 Sweep k' },
  ]
  return (
    <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1 w-fit">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all duration-150 ${
            active === t.id
              ? 'bg-zinc-700 text-zinc-100'
              : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

function StatusBanner({ status }) {
  if (!status) return null
  const isMock = status.mode === 'mock'
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs border ${
      isMock
        ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
        : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
    }`}>
      {isMock && <AlertTriangle size={13} />}
      {isMock
        ? 'Binary not found – running in mock mode. Build cuda/spec_decode.exe to use real GPU data.'
        : `GPU: ${status.exe_path?.split(/[\\/]/).pop()}`}
    </div>
  )
}

export default function App() {
  const [config,      setConfig]      = useState(DEFAULT_CONFIG)
  const [activeTab,   setActiveTab]   = useState('single')
  const [backendStatus, setBackendStatus] = useState(null)

  // Single run
  const [runResult,   setRunResult]   = useState(null)
  const [runLoading,  setRunLoading]  = useState(false)
  const [runError,    setRunError]    = useState(null)

  // Compare
  const [cmpData,     setCmpData]     = useState(null)
  const [cmpLoading,  setCmpLoading]  = useState(false)

  // Sweep
  const [sweepData,   setSweepData]   = useState(null)
  const [sweepLoading,setSweepLoading]= useState(false)

  // Check backend on load
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(setBackendStatus)
      .catch(() => setBackendStatus(null))
  }, [])

  const buildBody = () => ({
    mode:        config.mode,
    spec:        config.spec,
    max_tokens:  config.maxTokens,
    k:           config.k,
    seed:        config.seed,
    prompt_len:  config.promptLen,
  })

  const handleRun = async () => {
    setRunLoading(true)
    setRunError(null)
    setActiveTab('single')
    try {
      const r = await apiFetch('run', buildBody())
      setRunResult(r)
    } catch (e) {
      setRunError(e.message)
    } finally {
      setRunLoading(false)
    }
  }

  const handleCompare = async () => {
    setCmpLoading(true)
    setActiveTab('compare')
    try {
      const r = await apiFetch('compare', buildBody())
      setCmpData(r)
    } catch (e) {
      setCmpData(null)
    } finally {
      setCmpLoading(false)
    }
  }

  const handleSweep = async () => {
    setSweepLoading(true)
    setActiveTab('sweep')
    try {
      const r = await apiFetch('sweep', {
        mode:       config.mode,
        max_tokens: config.maxTokens,
        k_max:      config.k,
        seed:       config.seed,
        prompt_len: config.promptLen,
      })
      setSweepData(r)
    } catch (e) {
      setSweepData(null)
    } finally {
      setSweepLoading(false)
    }
  }

  const anyLoading = runLoading || cmpLoading || sweepLoading

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-zinc-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap size={20} className="text-blue-400" />
          <span className="font-bold text-lg text-zinc-100">Speculative Decode Explorer</span>
        </div>
        <div className="flex items-center gap-3">
          <StatusBanner status={backendStatus} />
          <a
            href="https://arxiv.org/abs/2211.17192"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Leviathan et al. 2022 ↗
          </a>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 gap-6 p-6 overflow-hidden">
        {/* Left: config */}
        <ConfigPanel
          config={config}
          onChange={setConfig}
          onRun={handleRun}
          onCompare={handleCompare}
          onSweep={handleSweep}
          loading={anyLoading}
        />

        {/* Right: results */}
        <main className="flex-1 flex flex-col gap-4 min-w-0 overflow-y-auto">
          <TabBar active={activeTab} onChange={setActiveTab} />

          {runError && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-2 text-sm">
              {runError}
            </div>
          )}

          {/* Single run tab */}
          {activeTab === 'single' && (
            <div className="flex gap-4 flex-1">
              {/* Main result */}
              <div className="flex-1 flex flex-col gap-4">
                <ResultsPanel result={runResult} loading={runLoading} />
              </div>
              {/* Side: iteration timeline */}
              <div className="w-72 shrink-0 space-y-4">
                <SpecTimeline
                  data={runResult?.iterations_detail}
                  k={config.k}
                />
                {/* Info box */}
                <div className="card space-y-2 text-xs text-zinc-500">
                  <div className="label">About</div>
                  <p>
                    The <span className="text-zinc-300">draft model</span> (2L, d=128) proposes k tokens.
                    The <span className="text-zinc-300">target model</span> (4L, d=256) verifies them in one pass.
                    Matching tokens are kept; the first mismatch triggers a rollback.
                  </p>
                  <p>
                    Expected speedup: <span className="text-zinc-300 font-mono">(1-α^(k+1))/((1-α)(ck+1))</span>
                    &nbsp;from Leviathan et al.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Compare tab */}
          {activeTab === 'compare' && (
            <div className="space-y-4">
              <CompareChart data={cmpData} loading={cmpLoading} />
              {cmpData && (
                <div className="grid grid-cols-4 gap-3">
                  {cmpData.map((d, i) => (
                    <div key={i} className="card-sm space-y-1 text-center">
                      <div className="label text-center">{d.label}</div>
                      <div className="text-lg font-bold font-mono text-zinc-100">
                        {d.spec_tok_per_s?.toFixed(0)}
                      </div>
                      <div className="text-xs text-zinc-500">tok/s</div>
                      <div className={`text-sm font-mono font-semibold ${
                        d.speedup >= 1.1 ? 'text-emerald-400' : 'text-zinc-400'
                      }`}>
                        {d.speedup?.toFixed(2)}×
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Sweep tab */}
          {activeTab === 'sweep' && (
            <div className="space-y-4">
              <SweepChart data={sweepData} loading={sweepLoading} />
              {sweepData && (
                <div className="card text-xs text-zinc-500 space-y-1">
                  <div className="label">Reading the chart</div>
                  <p>
                    The blue line is speedup (×). It rises as k increases (more tokens per iteration)
                    then falls as acceptance probability drops and wasted compute increases.
                    The optimal k* balances these forces. The purple dashed line shows how
                    acceptance rate α declines with larger k.
                  </p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
