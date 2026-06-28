interface Evidence {
  id: string
  source_text: string
  causal_type: string
  relevance?: number
  actor?: string
  event?: string
  mechanism?: string
}

interface Props {
  data: Record<string, unknown>
}

export function ExtractionView({ data }: Props) {
  const evidence = (data.evidence ?? []) as Evidence[]
  const actors = (data.actors ?? []) as Array<Record<string, string>>
  const mechanisms = (data.mechanisms ?? []) as Array<Record<string, string>>

  return (
    <div className="view">
      <div className="view-stats">
        <span className="stat">{evidence.length} evidence items</span>
        <span className="stat">{actors.length} actors</span>
        <span className="stat">{mechanisms.length} mechanisms</span>
      </div>

      <section className="view-section">
        <h3>Evidence Items</h3>
        <div className="card-list">
          {evidence.map(e => (
            <div key={e.id} className="card">
              <div className="card-header-row">
                <code className="id-badge">{e.id}</code>
                <span className={`type-badge type-${e.causal_type?.replace(/\s/g, '_')}`}>
                  {e.causal_type}
                </span>
                {e.relevance !== undefined && (
                  <span className={`relevance-badge ${e.relevance < 0.4 ? 'low' : e.relevance < 0.7 ? 'mid' : 'high'}`}>
                    rel {e.relevance.toFixed(2)}
                  </span>
                )}
              </div>
              <p className="source-text">"{e.source_text}"</p>
              <div className="card-meta">
                {e.actor && <span><strong>Actor:</strong> {e.actor}</span>}
                {e.mechanism && <span><strong>Mech:</strong> {e.mechanism}</span>}
              </div>
            </div>
          ))}
        </div>
      </section>

      {actors.length > 0 && (
        <section className="view-section">
          <h3>Actors ({actors.length})</h3>
          <div className="pill-list">
            {actors.map((a, i) => (
              <span key={i} className="pill">{a.id}: {a.name}</span>
            ))}
          </div>
        </section>
      )}

      {mechanisms.length > 0 && (
        <section className="view-section">
          <h3>Mechanisms ({mechanisms.length})</h3>
          <div className="pill-list">
            {mechanisms.map((m, i) => (
              <span key={i} className="pill">{m.id}: {m.description}</span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
