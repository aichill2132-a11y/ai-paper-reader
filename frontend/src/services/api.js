const API_BASE_URL = 'http://127.0.0.1:8000'

async function readError(response, fallback) {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
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

export { API_BASE_URL }
