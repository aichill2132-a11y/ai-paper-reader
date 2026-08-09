"""PDF text extraction.

Shared by POST /upload and the developer retrieval demo so there is exactly one
implementation. The PDF is read from memory with PyMuPDF and never written to
disk. Transport concerns (file size, content type) stay with the caller.
"""

from typing import Any, Dict, List

import fitz  # PyMuPDF

# How much of the extracted text the upload response echoes back.
PREVIEW_LENGTH = 3000


class PdfError(Exception):
    """A PDF that could not be turned into usable text.

    Carries the HTTP status the API should report, so the endpoint does not
    have to re-derive it.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def extract_pages(data: bytes) -> List[Dict[str, Any]]:
    """Return ``[{"page_number": n, "text": ...}]`` for every page."""
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise PdfError("The file could not be opened as a valid PDF.", 400)

    try:
        page_count = document.page_count
        return [
            {"page_number": number, "text": document[number - 1].get_text()}
            for number in range(1, page_count + 1)
        ]
    except Exception:
        raise PdfError("The PDF appears to be corrupted and could not be read.", 400)
    finally:
        document.close()


def extract_document(data: bytes, filename: str) -> Dict[str, Any]:
    """Extract a whole PDF into the payload shape POST /upload returns."""
    pages = extract_pages(data)
    full_text = "".join(page["text"] for page in pages)

    if not full_text.strip():
        raise PdfError(
            "No extractable text was found in this PDF. "
            "It may be a scanned document that needs OCR.",
            422,
        )

    return {
        "filename": filename,
        "page_count": len(pages),
        "character_count": len(full_text),
        "text_preview": full_text[:PREVIEW_LENGTH],
        "pages": pages,
    }
