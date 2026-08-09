import logging
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import OLLAMA_MODEL
from grounded_answer import answer_question
from ollama_client import OllamaError
from pdf import PdfError, extract_document
from schemas import AskRequest, AskResponse, SummaryRequest, SummaryResponse
from summarizer import summarize_paper

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Paper Reader API",
    version="0.3.0",
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
        return extract_document(contents, filename)
    except PdfError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


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


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer one question about an already-uploaded paper.

    Stateless by design: the client posts back the pages POST /upload returned,
    exactly as POST /summary already works. The evidence pipeline decides
    whether the paper can answer the question; the model is only asked to
    phrase an answer, and only when it can.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    if not request.pages:
        raise HTTPException(
            status_code=400,
            detail="No pages were supplied. Upload a paper first.",
        )

    if not any(page.text.strip() for page in request.pages):
        raise HTTPException(
            status_code=422,
            detail="The supplied pages contain no extractable text to search.",
        )

    try:
        return await answer_question(
            question, request.filename, request.pages, request.top_k
        )
    except OllamaError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)
