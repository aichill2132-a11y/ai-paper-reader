import './StatusBadge.css'

const LABELS = {
  checking: 'Checking backend...',
  online: 'Backend online',
  offline: 'Backend offline',
}

function StatusBadge({ status }) {
  const state = LABELS[status] ? status : 'checking'

  return (
    <span className={`status-badge status-badge--${state}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      {LABELS[state]}
    </span>
  )
}

export default StatusBadge
