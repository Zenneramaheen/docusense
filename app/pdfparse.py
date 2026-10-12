"""PDF text extraction with per-page tracking.

We extract text page-by-page (instead of one big blob) so every chunk can
be cited with its approximate page number -- that is what makes the
"Sources" in answers trustworthy.
"""

from io import BytesIO

from pypdf import PdfReader


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] for pages with text.

    Page numbers are 1-indexed (as a human reading the PDF would expect).
    Raises ValueError if the bytes are not a readable PDF.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Collapse all whitespace (newlines, double spaces) to single spaces.
        text = " ".join(text.split())
        if text:
            pages.append((i, text))
    return pages
