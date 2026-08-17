import { useCallback, useMemo, useRef, useState } from 'react'
import { askPaper } from '../services/api'
import './AskPaper.css'

const EXAMPLE_QUESTIONS = [
  'What were the main findings?',
  'How were participants recruited?',
  'What limitations did the authors identify?',
]

const STATUS_LABEL = {
  supported: 'Grounded answer',
  partially_supported: 'Partial evidence',
  not_supported: 'Not answered',
}

function sectionLabel(section) {
  if (!section) return ''
  return section
    .split('_')
    .filter(Boolean)
    .join(' ')
    .replace(/^./, (character) => character.toUpperCase())
}

/** Group sources by page so a page label is never repeated. */
function groupByPage(sources) {
  const order = []
  const byPage = new Map()

  for (const source of sources) {
    if (!byPage.has(source.page)) {
      byPage.set(source.page, { page: source.page, sections: [], excerpts: [] })
      order.push(source.page)
    }
    const group = byPage.get(source.page)
    const label = sectionLabel(source.section)
    if (label && !group.sections.includes(label)) group.sections.push(label)
    group.excerpts.push({ id: source.chunk_id, text: source.evidence })
  }

  return order.map((page) => byPage.get(page))
}

function AskPaper({ paper }) {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const inputRef = useRef(null)

  const grouped = useMemo(
    () => (result ? groupByPage(result.sources || []) : []),
    [result],
  )

  const submit = useCallback(async () => {
    const trimmed = question.trim()
    if (!trimmed || isAsking) return

    setIsAsking(true)
    setError('')
    setResult(null)

    try {
      const data = await askPaper({
        question: trimmed,
        filename: paper.filename,
        pages: paper.pages,
      })
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while answering.')
    } finally {
      setIsAsking(false)
    }
  }, [question, isAsking, paper])

  const handleKeyDown = (event) => {
    // Enter submits; Shift+Enter keeps the newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const applyExample = (example) => {
    setQuestion(example)
    inputRef.current?.focus()
  }

  const status = result?.status
  const isUnsupported = status === 'not_supported'

  return (
    <section className="ask" aria-labelledby="ask-title">
      <div className="ask__header">
        <h2 className="ask__title" id="ask-title">
          Ask the Paper
        </h2>
        <p className="ask__hint">
          Ask one question at a time. Answers come only from this paper, with the
          pages they were taken from.
        </p>
      </div>

      <div className="ask__form">
        <label className="ask__label" htmlFor="ask-question">
          Your question
        </label>
        <textarea
          id="ask-question"
          ref={inputRef}
          className="ask__input"
          rows={2}
          value={question}
          placeholder="e.g. What limitations did the authors identify?"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isAsking}
        />

        <div className="ask__actions">
          <ul className="ask__examples">
            {EXAMPLE_QUESTIONS.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  className="ask__example"
                  onClick={() => applyExample(example)}
                  disabled={isAsking}
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="button"
            onClick={submit}
            disabled={isAsking || !question.trim()}
          >
            {isAsking ? 'Asking...' : 'Ask'}
          </button>
        </div>
      </div>

      <div aria-live="polite" aria-busy={isAsking}>
        {isAsking && (
          <div className="panel panel--info" role="status">
            <span className="spinner" aria-hidden="true" />
            Searching the paper and checking the evidence...
          </div>
        )}

        {error && (
          <div className="panel panel--error" role="alert">
            <span className="panel__message">
              <strong>Question failed.</strong> {error}
            </span>
            <button
              type="button"
              className="button button--ghost"
              onClick={submit}
              disabled={isAsking}
            >
              Retry
            </button>
          </div>
        )}

        {result && isUnsupported && (
          <div className="ask__answer ask__answer--unsupported">
            <span className="ask__status ask__status--unsupported">
              {STATUS_LABEL.not_supported}
            </span>
            <p className="ask__text">{result.answer}</p>
            {result.confidence_note && (
              <p className="ask__note">{result.confidence_note}</p>
            )}
          </div>
        )}

        {result && !isUnsupported && (
          <article
            className={
              status === 'partially_supported'
                ? 'ask__answer ask__answer--partial'
                : 'ask__answer'
            }
          >
            <span
              className={
                status === 'partially_supported'
                  ? 'ask__status ask__status--partial'
                  : 'ask__status ask__status--supported'
              }
            >
              {STATUS_LABEL[status] || STATUS_LABEL.supported}
            </span>

            <p className="ask__text">{result.answer}</p>

            {grouped.length > 0 && (
              <div className="ask__sources">
                <h3 className="ask__sources-title">Sources</h3>
                {grouped.map((group) => (
                  <details className="ask__source" key={group.page}>
                    <summary className="ask__source-summary">
                      <span className="ask__page">Page {group.page}</span>
                      {group.sections.length > 0 && (
                        <span className="ask__section">
                          {group.sections.join(' · ')}
                        </span>
                      )}
                      <span className="ask__toggle">Supporting evidence</span>
                    </summary>
                    {group.excerpts.map((excerpt) => (
                      <blockquote className="ask__evidence" key={excerpt.id}>
                        {excerpt.text}
                      </blockquote>
                    ))}
                  </details>
                ))}
              </div>
            )}

            {result.confidence_note && (
              <p className="ask__note">{result.confidence_note}</p>
            )}
          </article>
        )}
      </div>
    </section>
  )
}

export default AskPaper
