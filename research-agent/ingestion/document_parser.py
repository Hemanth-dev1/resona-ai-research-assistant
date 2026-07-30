"""Extract plain text from uploaded chat documents.

Supports: .txt, .md, .pdf, .docx
Deliberately dependency-light — pypdf and python-docx are the only new
requirements this adds (see requirements.txt).
"""

import io

MAX_FILE_SIZE_MB = 15
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedFileType(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


def _ext(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def extract_text(filename: str, content: bytes) -> str:
    """Extract raw text from an uploaded file's bytes.

    Args:
        filename: Original filename (used to detect type).
        content: Raw file bytes.

    Returns:
        Extracted plain text.

    Raises:
        UnsupportedFileType: If the extension isn't supported.
        FileTooLarge: If the file exceeds MAX_FILE_SIZE_MB.
    """
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileTooLarge(f"File is {size_mb:.1f}MB — max is {MAX_FILE_SIZE_MB}MB")

    extension = _ext(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"'{extension}' not supported. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension in (".txt", ".md"):
        return content.decode("utf-8", errors="ignore")

    if extension == ".pdf":
        return _extract_pdf(content)

    if extension == ".docx":
        return _extract_docx(content)

    raise UnsupportedFileType(extension)


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def _extract_docx(content: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    document = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
