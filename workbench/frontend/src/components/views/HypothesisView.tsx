import { useState } from 'react'

interface Prediction {
  id: string
  description: string
}

interface Hypothesis {
  id: string
  description: string
  causal_mechanism: string
  theoretical_basis: string
  observable_predictions: Prediction[]
}

interface Props {
  data: Record<string, unknown>
  onEdit: (hypotheses: unknown[]) => Promise<void>
}

export function HypothesisView({ data, onEdit }: Props) {
  const rq = data.research_question as string
  const hypotheses = (data.hypotheses ?? []) as Hypothesis[]
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)

  const startEdit = () => {
    setDraft(JSON.stringify(hypotheses, null, 2))
    setEditing(true)
    setEditError(null)
  }

  const save = async () => {
    setSaving(true)
    setEditError(null)
    try {
      const parsed = JSON.parse(draft)
      await onEdit(parsed)
      setEditing(false)
    } catch (e) {
      setEditError(e instanceof Error ? e.message : 'Invalid JSON')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="view">
      {rq && (
        <div className="rq-box">
          <strong>Research question:</strong> {rq}
        </div>
      )}

      <div className="view-stats">
        <span className="stat">{hypotheses.length} hypotheses</span>
        <button className="btn-ghost" onClick={startEdit} style={{ fontSize: 12, padding: '3px 10px' }}>
          ✎ Edit hypotheses
        </button>
      </div>

      {editing && (
        <div className="edit-overlay">
          <div className="edit-box">
            <h4>Edit Hypotheses (JSON)</h4>
            <p className="edit-hint">
              Edit descriptions, mechanisms, or predictions. Changes invalidate downstream passes.
            </p>
            <textarea
              className="json-editor"
              value={draft}
              onChange={e => setDraft(e.target.value)}
              rows={20}
            />
            {editError && <div className="inline-error">{editError}</div>}
            <div className="edit-actions">
              <button className="btn-ghost" onClick={() => setEditing(false)} disabled={saving}>Cancel</button>
              <button className="btn-primary" onClick={save} disabled={saving}>
                {saving ? 'Saving…' : 'Save & invalidate downstream'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="card-list">
        {hypotheses.map((h, i) => (
          <div key={h.id} className="card">
            <div className="card-header-row">
              <code className="id-badge">{h.id}</code>
              <span className="hyp-label">H{i + 1}</span>
            </div>
            <p className="hyp-desc">{h.description}</p>
            <div className="hyp-detail">
              <div><strong>Mechanism:</strong> {h.causal_mechanism}</div>
              <div><strong>Theory:</strong> {h.theoretical_basis}</div>
            </div>
            {h.observable_predictions?.length > 0 && (
              <div className="predictions">
                <strong>Observable predictions ({h.observable_predictions.length}):</strong>
                <ul>
                  {h.observable_predictions.map(p => (
                    <li key={p.id}>{p.description}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
