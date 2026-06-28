interface RivalPair {
  h1_id: string
  h2_id: string
  overlap_concern: boolean
  complementary_concern: boolean
  absorptive_concern: boolean
  discriminator_count: number
  concern_detail: string
}

interface Props {
  data: Record<string, unknown>
}

export function PartitionView({ data }: Props) {
  const quality = data.overall_quality as string
  const pairs = (data.rival_pairs ?? []) as RivalPair[]
  const flagged = (data.hypotheses_flagged ?? []) as string[]
  const summary = data.summary as string

  const isAdequate = quality === 'adequate'

  return (
    <div className="view">
      <div className={`quality-banner ${isAdequate ? 'adequate' : 'needs-review'}`}>
        <span className="quality-label">
          {isAdequate ? '✓ Adequate' : '⚠ Needs Review'}
        </span>
        <span className="quality-summary">{summary}</span>
      </div>

      {flagged.length > 0 && (
        <div className="flagged-box">
          <strong>Flagged hypotheses:</strong> {flagged.join(', ')}
        </div>
      )}

      <section className="view-section">
        <h3>Rival Pair Analysis ({pairs.length} pairs)</h3>
        <div className="partition-table-wrap">
          <table className="partition-table">
            <thead>
              <tr>
                <th>Pair</th>
                <th>Overlap</th>
                <th>Complementary</th>
                <th>Absorptive</th>
                <th>Discriminators</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, i) => (
                <tr key={i} className={p.overlap_concern || p.complementary_concern || p.absorptive_concern ? 'row-concern' : ''}>
                  <td><code>{p.h1_id} ↔ {p.h2_id}</code></td>
                  <td><ConcernDot on={p.overlap_concern} label="Overlap" /></td>
                  <td><ConcernDot on={p.complementary_concern} label="Complementary" /></td>
                  <td><ConcernDot on={p.absorptive_concern} label="Absorptive" /></td>
                  <td>
                    <span className={`disc-count ${p.discriminator_count === 0 ? 'bad' : p.discriminator_count < 3 ? 'warn' : 'good'}`}>
                      {p.discriminator_count}
                    </span>
                  </td>
                  <td className="concern-detail">{p.concern_detail || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

function ConcernDot({ on, label }: { on: boolean; label: string }) {
  return on ? (
    <span className="concern-badge on">{label}</span>
  ) : (
    <span className="concern-badge off">—</span>
  )
}
