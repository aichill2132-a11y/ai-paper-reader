import { useCallback, useEffect, useState } from 'react'
import StatusBadge from '../components/StatusBadge'
import SummaryCard from '../components/SummaryCard'
import UploadCard from '../components/UploadCard'
import { checkHealth, generateSummary, uploadPaper } from '../services/api'
import './Home.css'

function Home() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const [summary, setSummary] = useState(null)
  const [summaryError, setSummaryError] = useState('')
  const [isSummarizing, setIsSummarizing] = useState(false)

  useEffect(() => {
    let cancelled = false

    checkHealth()
      .then(() => {
        if (!cancelled) setBackendStatus('online')
      })
      .catch(() => {
        if (!cancelled) setBackendStatus('offline')
      })

    return () => {
      cancelled = true
    }
  }, [])

  const resetSummary = useCallback(() => {
    setSummary(null)
    setSummaryError('')
  }, [])

  const handleSelectFile = useCallback(
    (selected) => {
      setFile(selected)
      setResult(null)
      setError('')
      resetSummary()
    },
    [resetSummary],
  )

  const handleUpload = useCallback(async () => {
    if (!file) return

    setIsLoading(true)
    setError('')
    setResult(null)
    resetSummary()

    try {
      const data = await uploadPaper(file)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while uploading.')
    } finally {
      setIsLoading(false)
    }
  }, [file, resetSummary])

  const handleSummarize = useCallback(async () => {
    if (!result) return

    setIsSummarizing(true)
    setSummaryError('')
    setSummary(null)

    try {
      const data = await generateSummary({
        filename: result.filename,
        pages: result.pages,
      })
      setSummary(data)
    } catch (err) {
      setSummaryError(err.message || 'Something went wrong while summarising.')
    } finally {
      setIsSummarizing(false)
    }
  }, [result])

  return (
    <div className="page">
      <header className="page__header">
        <div className="page__heading">
          <h1 className="page__title">AI Paper Reader</h1>
          <p className="page__subtitle">
            Upload a research paper, extract its text, and summarise it with a
            local model.
          </p>
        </div>
        <StatusBadge status={backendStatus} />
      </header>

      <main className="page__main">
        <UploadCard
          file={file}
          onSelectFile={handleSelectFile}
          onUpload={handleUpload}
          isLoading={isLoading}
        />

        {isLoading && (
          <div className="panel panel--info" role="status">
            <span className="spinner" aria-hidden="true" />
            Extracting text from your paper...
          </div>
        )}

        {error && (
          <div className="panel panel--error" role="alert">
            <strong>Upload failed.</strong> {error}
          </div>
        )}

        {result && (
          <section className="result">
            <div className="metadata">
              <div className="metadata__item">
                <span className="metadata__label">File</span>
                <span className="metadata__value metadata__value--wrap">
                  {result.filename}
                </span>
              </div>
              <div className="metadata__item">
                <span className="metadata__label">Pages</span>
                <span className="metadata__value">{result.page_count}</span>
              </div>
              <div className="metadata__item">
                <span className="metadata__label">Characters</span>
                <span className="metadata__value">
                  {result.character_count.toLocaleString()}
                </span>
              </div>
            </div>

            <div className="summary-launcher">
              <div className="summary-launcher__copy">
                <h2 className="summary-launcher__title">AI summary</h2>
                <p className="summary-launcher__hint">
                  Runs locally. A long paper can take a few minutes.
                </p>
              </div>
              <button
                type="button"
                className="button"
                onClick={handleSummarize}
                disabled={isSummarizing}
              >
                {isSummarizing
                  ? 'Generating...'
                  : summary
                    ? 'Regenerate Summary'
                    : 'Generate Summary'}
              </button>
            </div>

            {isSummarizing && (
              <div className="panel panel--info" role="status">
                <span className="spinner" aria-hidden="true" />
                Reading the paper section by section and writing the summary...
              </div>
            )}

            {summaryError && (
              <div className="panel panel--error" role="alert">
                <span className="panel__message">
                  <strong>Summary failed.</strong> {summaryError}
                </span>
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={handleSummarize}
                  disabled={isSummarizing}
                >
                  Retry
                </button>
              </div>
            )}

            {summary && <SummaryCard data={summary} />}

            <div className="preview">
              <div className="preview__header">
                <h2 className="preview__title">Extracted text preview</h2>
                <span className="preview__meta">
                  First {result.text_preview.length.toLocaleString()} characters
                </span>
              </div>
              <pre className="preview__body">{result.text_preview}</pre>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default Home
