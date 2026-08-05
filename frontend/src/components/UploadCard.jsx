import { useRef, useState } from 'react'
import './UploadCard.css'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function UploadCard({ file, onSelectFile, onUpload, isLoading }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const openFilePicker = () => {
    if (!isLoading) inputRef.current?.click()
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openFilePicker()
    }
  }

  const handleInputChange = (event) => {
    const selected = event.target.files?.[0]
    if (selected) onSelectFile(selected)
    event.target.value = ''
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    if (!isLoading) setIsDragging(true)
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    if (isLoading) return
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) onSelectFile(dropped)
  }

  return (
    <div className="upload-card">
      <div
        className={`dropzone${isDragging ? ' dropzone--active' : ''}${
          isLoading ? ' dropzone--disabled' : ''
        }`}
        role="button"
        tabIndex={0}
        onClick={openFilePicker}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <svg
          className="dropzone__icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M12 16V4" />
          <path d="m7 9 5-5 5 5" />
          <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
        <p className="dropzone__title">Drop a PDF here, or click to browse</p>
        <p className="dropzone__hint">Text-based PDFs only, up to 20 MB</p>

        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="dropzone__input"
          onChange={handleInputChange}
        />
      </div>

      <div className="upload-card__footer">
        <p className="upload-card__file">
          {file ? (
            <>
              <span className="upload-card__filename">{file.name}</span>
              <span className="upload-card__filesize">{formatSize(file.size)}</span>
            </>
          ) : (
            <span className="upload-card__empty">No file selected</span>
          )}
        </p>

        <button
          type="button"
          className="button"
          onClick={onUpload}
          disabled={!file || isLoading}
        >
          {isLoading ? 'Uploading...' : 'Upload Paper'}
        </button>
      </div>
    </div>
  )
}

export default UploadCard
