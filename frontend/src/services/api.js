const API_BASE_URL = 'http://127.0.0.1:8000'

async function readError(response, fallback) {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
    // FastAPI request-validation errors arrive as an array of issues.
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      return data.detail.map((issue) => issue.msg).join('; ')
    }
  } catch {
    // Response had no JSON body; fall through to the default message.
  }
  return fallback
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error(await readError(response, 'Backend health check failed.'))
  }
  return response.json()
}

export async function uploadPaper(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await readError(response, 'Upload failed. Please try again.'))
  }

  return response.json()
}

export async function generateSummary({ filename, pages }) {
  const response = await fetch(`${API_BASE_URL}/summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, pages }),
  })

  if (!response.ok) {
    throw new Error(
      await readError(response, 'Summary generation failed. Please try again.'),
    )
  }

  return response.json()
}

export async function askPaper({ question, filename, pages, topK = 5 }) {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, filename, pages, top_k: topK }),
  })

  if (!response.ok) {
    throw new Error(
      await readError(response, 'Could not answer that question. Please try again.'),
    )
  }

  return response.json()
}
