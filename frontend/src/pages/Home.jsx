import { useCallback, useEffect, useState } from 'react'
import StatusBadge from '../components/StatusBadge'
import UploadCard from '../components/UploadCard'
import { checkHealth, uploadPaper } from '../services/api'
import './Home.css'

function Home() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

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

  const handleSelectFile = useCallback((selected) => {
    setFile(selected)
    setResult(null)
    setError('')
  }, [])

  const handleUpload = useCallback(async () => {
    if (!file) return

    setIsLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await uploadPaper(file)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while uploading.')
    } finally {
      setIsLoading(false)
    }
  }, [file])

  return (
    <div className="page">
      <header className="page__header">
        <div className="page__heading">
          <h1 className="page__title">AI Paper Reader</h1>
          <p className="page__subtitle">
            Upload a research paper and extract its text, page by page.
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
