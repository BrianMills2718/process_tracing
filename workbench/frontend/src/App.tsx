import { useState } from 'react'
import './App.css'
import { InputPanel } from './components/InputPanel'
import { PassPanel } from './components/PassPanel'
import type { Session, PassId } from './types'

const PASSES: { id: PassId; label: string; desc: string }[] = [
  { id: 'extract',    label: 'Pass 1 — Extract',         desc: 'Evidence, actors, events, mechanisms' },
  { id: 'hypothesize',label: 'Pass 2 — Hypothesize',     desc: 'Rival causal hypotheses' },
  { id: 'partition',  label: 'Pass 2.5 — Partition Audit', desc: 'Check hypothesis rivalry quality' },
  { id: 'test',       label: 'Pass 3 — Test',             desc: 'Likelihood vector per evidence item' },
  { id: 'absence',    label: 'Pass 3b — Absence',         desc: 'Missing predicted evidence' },
  { id: 'bayesian',   label: 'Pass 3.5 — Bayesian',       desc: 'Comparative support update' },
  { id: 'synthesize', label: 'Pass 4 — Synthesize',       desc: 'Written analytical narrative' },
]

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [activePass, setActivePass] = useState<PassId | null>(null)
  const [results, setResults] = useState<Record<string, unknown>>({})
  const [running, setRunning] = useState<PassId | null>(null)
  const [error, setError] = useState<string | null>(null)

  const passComplete = (id: PassId) => Boolean(results[id])

  const runPass = async (id: PassId) => {
    if (!session) return
    setRunning(id)
    setError(null)
    try {
      const res = await fetch(`/api/sessions/${session.session_id}/${id}`, { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(body.detail ?? res.statusText)
      }
      const data = await res.json()
      setResults(prev => ({ ...prev, [id]: data }))
      setActivePass(id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(null)
    }
  }

  const onSessionCreated = (s: Session) => {
    setSession(s)
    setResults({})
    setActivePass(null)
    setError(null)
  }

  const onHypothesesEdited = async (hypotheses: unknown[]) => {
    if (!session) return
    const res = await fetch(`/api/sessions/${session.session_id}/hypotheses`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hypotheses }),
    })
    if (!res.ok) throw new Error((await res.json()).detail)
    const data = await res.json()
    // Reset downstream results
    setResults(prev => ({
      ...prev,
      hypothesize: data,
      partition: undefined,
      test: undefined,
      absence: undefined,
      bayesian: undefined,
      synthesize: undefined,
    }))
  }

  const nextPass = () => {
    const idx = PASSES.findIndex(p => p.id === activePass)
    if (idx < PASSES.length - 1) setActivePass(PASSES[idx + 1].id)
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-logo">⚙ Process Tracing Workbench</span>
        {session && (
          <span className="session-badge">
            session <code>{session.session_id}</code> · <code>{session.model}</code>
          </span>
        )}
      </header>

      <div className="app-body">
        {/* Left sidebar: pass list */}
        <nav className="pass-nav">
          <div
            className={`pass-nav-item ${!session ? 'active' : ''}`}
            onClick={() => setActivePass(null)}
          >
            <span className="pass-status-dot" style={{ background: session ? 'var(--done)' : 'var(--accent)' }} />
            Input Text
          </div>
          {PASSES.map(p => {
            const done = passComplete(p.id)
            const isActive = activePass === p.id
            const isRunning = running === p.id
            return (
              <div
                key={p.id}
                className={`pass-nav-item ${isActive ? 'active' : ''} ${!session ? 'disabled' : ''}`}
                onClick={() => session && setActivePass(p.id)}
              >
                <span
                  className="pass-status-dot"
                  style={{
                    background: isRunning ? 'var(--warn)' : done ? 'var(--done)' : 'var(--border)',
                    animation: isRunning ? 'pulse 1s infinite' : 'none',
                  }}
                />
                <span className="pass-nav-label">{p.label}</span>
                {done && <span className="pass-done-tick">✓</span>}
              </div>
            )
          })}
        </nav>

        {/* Main panel */}
        <main className="main-panel">
          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
              <button className="btn-ghost" style={{ marginLeft: 12, padding: '2px 8px' }} onClick={() => setError(null)}>✕</button>
            </div>
          )}

          {activePass === null && (
            <InputPanel onSessionCreated={onSessionCreated} />
          )}

          {session && activePass !== null && (
            <PassPanel
              pass={PASSES.find(p => p.id === activePass)!}
              session={session}
              result={results[activePass]}
              allResults={results}
              running={running === activePass}
              done={passComplete(activePass)}
              onRun={() => runPass(activePass)}
              onNext={nextPass}
              onHypothesesEdited={onHypothesesEdited}
            />
          )}
        </main>
      </div>
    </div>
  )
}
