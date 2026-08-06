import logging
from typing import Any, Dict

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import OLLAMA_MODEL
from ollama_client import OllamaError
from schemas import SummaryRequest, SummaryResponse
from summarizer import summarize_paper

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
PREVIEW_LENGTH = 3000

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Paper Reader API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "AI Paper Reader API is running"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or ""

    # Accept PDFs only.
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The uploaded file is larger than the 20 MB limit.",
        )

    # Read the PDF straight from memory. Nothing is written to disk.
    try:
        document = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The file could not be opened as a valid PDF.",
        )

    try:
        page_count = document.page_count
        pages = [
            {"page_number": number, "text": document[number - 1].get_text()}
            for number in range(1, page_count + 1)
        ]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The PDF appears to be corrupted and could not be read.",
        )
    finally:
        document.close()

    full_text = "".join(page["text"] for page in pages)

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text was found in this PDF. "
                "It may be a scanned document that needs OCR."
            ),
        )

    return {
        "filename": filename,
        "page_count": page_count,
        "character_count": len(full_text),
        "text_preview": full_text[:PREVIEW_LENGTH],
        "pages": pages,
    }


@app.post("/summary", response_model=SummaryResponse)
async def summarize(request: SummaryRequest) -> SummaryResponse:
    if not request.pages:
        raise HTTPException(
            status_code=400,
            detail="No pages were supplied. Upload a paper first.",
        )

    if not any(page.text.strip() for page in request.pages):
        raise HTTPException(
            status_code=422,
            detail="The supplied pages contain no extractable text to summarise.",
        )

    try:
        summary, chunk_count = await summarize_paper(request.filename, request.pages)
    except OllamaError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    return SummaryResponse(
        filename=request.filename,
        model=OLLAMA_MODEL,
        page_count=max(page.page_number for page in request.pages),
        chunk_count=chunk_count,
        summary=summary,
    )
