import { useState } from 'react'
import type { Session } from '../types'

const MODELS = [
  'gemini/gemini-2.5-flash',
  'deepseek/deepseek-chat',
  'openrouter/openai/gpt-5.4-mini',
  'openrouter/anthropic/claude-haiku-4-5',
]

const SAMPLE = `The French Revolution began in 1789 with a severe fiscal crisis. France was
effectively bankrupt after years of costly wars, including support for American
independence. King Louis XVI called the Estates-General in May 1789 — the first
time in 175 years — to address the financial emergency. The Third Estate,
representing commoners, declared itself a National Assembly in June and was soon
joined by liberal nobles and clergy. When the king appeared to mobilise troops
around Paris in July, Parisians stormed the Bastille on 14 July, a moment that
became the symbolic start of the Revolution. The peasant uprising known as the
Great Fear swept rural France simultaneously. The Assembly abolished feudalism in
August and issued the Declaration of the Rights of Man. Louis XVI's failed flight
to Varennes in 1791 destroyed public trust in the monarchy. War with Austria and
Prussia from 1792 radicalised politics; the monarchy was abolished in September
1792 and Louis executed in January 1793. The radical Jacobins dominated the
Committee of Public Safety during the Terror of 1793-94, executing thousands of
perceived enemies. Thermidorian reaction ended the Terror in 1794, leading
eventually to the Directory and then Napoleon's rise to power.`

interface Props {
  onSessionCreated: (s: Session) => void
}

export function InputPanel({ onSessionCreated }: Props) {
  const [text, setText] = useState('')
  const [model, setModel] = useState(MODELS[0])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), model }),
      })
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
      const session: Session = await res.json()
      onSessionCreated(session)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="input-panel">
      <div className="input-panel-header">
        <h2>Process Tracing Workbench</h2>
        <p className="subtitle">
          Paste any historical or political text. Run each analytical pass individually,
          inspect the output, edit hypotheses, then continue.
        </p>
      </div>

      <div className="input-panel-body">
        <div className="field-group">
          <label>Source Text</label>
          <textarea
            className="text-input"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste historical text here (500–50,000 characters)…"
            rows={14}
          />
          <div className="char-count">{text.length.toLocaleString()} chars</div>
        </div>

        <div className="row-controls">
          <div className="field-group" style={{ flex: 1 }}>
            <label>Model</label>
            <select className="model-select" value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="button-group">
            <button
              className="btn-ghost"
              onClick={() => setText(SAMPLE)}
              disabled={loading}
            >
              Load sample (French Revolution)
            </button>
            <button
              className="btn-primary"
              onClick={start}
              disabled={loading || !text.trim()}
            >
              {loading ? 'Creating session…' : '▶ Start analysis'}
            </button>
          </div>
        </div>

        {error && <div className="inline-error">{error}</div>}
      </div>
    </div>
  )
}
