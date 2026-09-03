import './SummaryCard.css'

const NOT_STATED = 'Not stated in the paper'

function isMissing(value) {
  if (Array.isArray(value)) {
    return value.length === 0 || value.every((item) => item === NOT_STATED)
  }
  return !value || value === NOT_STATED
}

function SourceBadges({ pages }) {
  if (!pages || pages.length === 0) return null

  // Each page stays a separate integer in the data and a separate badge in the
  // markup, but the separators are real text. Without them the numbers run
  // together as "Pages146" wherever the badge spacing does not apply - copied
  // text, screen readers, narrow layouts.
  const label = pages.length === 1 ? 'Page' : 'Pages'
  return (
    <span className="source-pages">
      <span className="source-pages__label">{label}</span>{' '}
      {pages.map((page, index) => (
        <span key={page}>
          <span className="source-pages__badge">{page}</span>
          {index < pages.length - 1 ? <span aria-hidden="true">, </span> : null}
        </span>
      ))}
    </span>
  )
}

function Section({ title, children, pages }) {
  return (
    <section className="summary-section">
      <div className="summary-section__header">
        <h3 className="summary-section__title">{title}</h3>
        <SourceBadges pages={pages} />
      </div>
      {children}
    </section>
  )
}

function TextSection({ title, value, pages }) {
  return (
    <Section title={title} pages={pages}>
      <p className={`summary-section__text${isMissing(value) ? ' is-missing' : ''}`}>
        {value}
      </p>
    </Section>
  )
}

function ListSection({ title, items, pages, emptyText }) {
  if (isMissing(items)) {
    // An optional list that came back empty is an answer, not a gap, so it
    // gets its own wording and no page badges - citing pages for something
    // the section says is absent would be misleading.
    if (emptyText) {
      return <TextSection title={title} value={emptyText} pages={null} />
    }
    return <TextSection title={title} value={NOT_STATED} pages={pages} />
  }

  return (
    <Section title={title} pages={pages}>
      <ul className="summary-section__list">
        {items.map((item, index) => (
          <li key={`${index}-${item.slice(0, 24)}`}>{item}</li>
        ))}
      </ul>
    </Section>
  )
}

function SummaryCard({ data }) {
  const { summary, model, chunk_count: chunkCount, page_count: pageCount } = data
  const sourcePages = summary.source_pages || {}

  return (
    <article className="summary">
      <header className="summary__header">
        <div>
          <h2 className="summary__title">{summary.title}</h2>
          <p className="summary__authors">
            {isMissing(summary.authors)
              ? NOT_STATED
              : summary.authors.join(', ')}
          </p>
        </div>
        <p className="summary__meta">
          {model} &middot; {pageCount} {pageCount === 1 ? 'page' : 'pages'} &middot;{' '}
          {chunkCount} {chunkCount === 1 ? 'chunk' : 'chunks'}
        </p>
      </header>

      <div className="summary__highlight">
        <h3 className="summary-section__title">In plain English</h3>
        <p className="summary-section__text">{summary.plain_english_summary}</p>
      </div>

      <div className="summary__body">
        <TextSection
          title="Research question"
          value={summary.research_question}
          pages={sourcePages.research_question}
        />
        <TextSection title="Background" value={summary.background} />
        <TextSection
          title="Methods"
          value={summary.methods}
          pages={sourcePages.methods}
        />
        <TextSection
          title="Participants or data"
          value={summary.participants_or_data}
        />
        <ListSection
          title="Key findings"
          items={summary.key_findings}
          pages={sourcePages.key_findings}
        />
        <ListSection
          title="Author-stated limitations"
          items={summary.author_stated_limitations}
          pages={sourcePages.author_stated_limitations}
          emptyText="No explicit author-stated limitations were identified."
        />
        {!isMissing(summary.model_identified_considerations) && (
          <Section title="Model-identified considerations">
            <p className="summary-section__caveat">
              Raised by the model, not stated by the authors.
            </p>
            <ul className="summary-section__list">
              {summary.model_identified_considerations.map((item, index) => (
                <li key={`${index}-${item.slice(0, 24)}`}>{item}</li>
              ))}
            </ul>
          </Section>
        )}
      </div>

      <footer className="summary__footer">
        <h3 className="summary-section__title">Confidence notes</h3>
        <p className="summary-section__text">{summary.confidence_notes}</p>
        <p className="summary__disclaimer">
          Generated locally by {model}. Check anything important against the paper
          itself.
        </p>
      </footer>
    </article>
  )
}

export default SummaryCard
