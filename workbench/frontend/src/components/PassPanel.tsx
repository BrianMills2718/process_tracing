import { useState } from 'react'
import type { PassId, Session } from '../types'
import { ExtractionView } from './views/ExtractionView'
import { HypothesisView } from './views/HypothesisView'
import { PartitionView } from './views/PartitionView'
import { TestingView } from './views/TestingView'
import { AbsenceView } from './views/AbsenceView'
import { BayesianView } from './views/BayesianView'
import { SynthesisView } from './views/SynthesisView'

interface PassDef {
  id: PassId
  label: string
  desc: string
}

interface Props {
  pass: PassDef
  session: Session
  result: unknown
  allResults: Record<string, unknown>
  running: boolean
  done: boolean
  onRun: () => void
  onNext: () => void
  onHypothesesEdited: (hypotheses: unknown[]) => Promise<void>
}

const DESCRIPTIONS: Record<PassId, string> = {
  extract:     'Extracts evidence items, actors, events, and causal mechanisms from the source text.',
  hypothesize: 'Generates competing causal hypotheses with observable predictions. You can edit these before testing.',
  partition:   'Audits each hypothesis pair for rivalry quality: overlap, complementary, or absorptive concerns.',
  test:        'Elicits a likelihood vector for each evidence item across all hypotheses (one LLM call).',
  absence:     'Evaluates missing predicted evidence (failed hoop tests) — qualitative only, not Bayesian.',
  bayesian:    'Deterministic log-space softmax update: posterior ∝ prior × ∏ likelihood ratios.',
  synthesize:  'Writes the analytical narrative with verdicts, steelmans, and robustness assessment.',
}

export function PassPanel({
  pass, session, result, allResults, running, done, onRun, onNext, onHypothesesEdited,
}: Props) {
  const [showRaw, setShowRaw] = useState(false)

  const renderView = () => {
    if (!result) return null
    if (showRaw) return <pre>{JSON.stringify(result, null, 2)}</pre>
    switch (pass.id) {
      case 'extract':     return <ExtractionView data={result as Record<string, unknown>} />
      case 'hypothesize': return <HypothesisView data={result as Record<string, unknown>} onEdit={onHypothesesEdited} />
      case 'partition':   return <PartitionView data={result as Record<string, unknown>} />
      case 'test':        return <TestingView data={result as Record<string, unknown>} hypotheses={(allResults['hypothesize'] as Record<string, unknown>)} />
      case 'absence':     return <AbsenceView data={result as Record<string, unknown>} />
      case 'bayesian':    return <BayesianView data={result as Record<string, unknown>} hypotheses={(allResults['hypothesize'] as Record<string, unknown>)} />
      case 'synthesize':  return <SynthesisView data={result as Record<string, unknown>} sessionId={session.session_id} />
      default:            return <pre>{JSON.stringify(result, null, 2)}</pre>
    }
  }

  return (
    <div className="pass-panel">
      <div className="pass-panel-header">
        <div>
          <h2 className="pass-title">{pass.label}</h2>
          <p className="pass-desc">{DESCRIPTIONS[pass.id]}</p>
        </div>
        <div className="pass-header-controls">
          {done && (
            <button
              className="btn-ghost"
              onClick={() => setShowRaw(r => !r)}
              style={{ fontSize: 12 }}
            >
              {showRaw ? 'Formatted' : 'Raw JSON'}
            </button>
          )}
          <button
            className="btn-primary"
            onClick={onRun}
            disabled={running}
          >
            {running ? (
              <span className="spinner-text">⟳ Running…</span>
            ) : done ? (
              '↺ Re-run'
            ) : (
              `▶ Run ${pass.label.split('—')[0].trim()}`
            )}
          </button>
          {done && (
            <button className="btn-ghost" onClick={onNext}>
              Next pass →
            </button>
          )}
        </div>
      </div>

      <div className="pass-panel-body">
        {running && (
          <div className="running-indicator">
            <div className="spinner" />
            <span>Running {pass.label}… this may take 30–120 seconds</span>
          </div>
        )}
        {!running && !result && (
          <div className="empty-state">
            <p>Click <strong>Run {pass.label.split('—')[0].trim()}</strong> to execute this pass.</p>
          </div>
        )}
        {!running && result && renderView()}
      </div>
    </div>
  )
}
