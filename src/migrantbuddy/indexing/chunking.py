"""Structure-aware chunking: split documents on headings, merge undersized
sections, sub-split oversized ones (never splitting a table), and prefix a
context breadcrumb. Extracted from notebooks/02_chunking.ipynb.

Pure text transformation -- no I/O, no model loading -- kept separate from
indexing/service.py (which handles embedding + Chroma persistence) so this
logic is fast and cheap to unit test in isolation.
"""

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from migrantbuddy.config import CHUNK_OVERLAP, CHUNK_SIZE
from migrantbuddy.ingestion import IngestedDocument

# Allow leading whitespace before the '#' -- some source pages (e.g.
# medical-insurance) have headings indented by stray spaces from trafilatura
# extraction, which a strictly-anchored "^#" regex would silently miss and
# fold into the prior section.
HEADING_RE = re.compile(r"^\s*(#{2,3})\s+(.*)$")

TABLE_BLOCK_RE = re.compile(r"(?:[ \t]*\|.*\|[ \t]*\n(?:[ \t]*\n)*)+")


def count_tokens(text: str) -> int:
    return len(text) // 4


@dataclass(frozen=True)
class Section:
    heading_path: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    url: str
    heading_path: str
    text: str
    token_count: int


def split_by_headers(text: str) -> list[Section]:
    """Split markdown text at heading boundaries (##, ###), keeping the
    heading path (e.g. "Overtime pay > How overtime pay is calculated")
    attached to each resulting section.
    """
    lines = text.split("\n")
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            heading_path = " > ".join(title for _, title in stack)
            sections.append(Section(heading_path=heading_path, text=body))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            current_lines.append(line)
    flush()
    return sections


def merge_sections(sections: Sequence[Section], chunk_size: int) -> list[Section]:
    """Header splitting alone leaves some sections far too small. Merge
    adjacent small sections forward until they approach the target chunk
    size, instead of emitting tiny fragments.
    """
    merged: list[Section] = []
    buffer_texts: list[str] = []
    buffer_heading_path = ""

    def flush() -> None:
        if buffer_texts:
            merged.append(Section(heading_path=buffer_heading_path, text="\n\n".join(buffer_texts)))

    for section in sections:
        if not buffer_texts:
            buffer_texts = [section.text]
            buffer_heading_path = section.heading_path
            continue

        combined_tokens = count_tokens("\n\n".join(buffer_texts)) + count_tokens(section.text)
        if combined_tokens <= chunk_size:
            buffer_texts.append(section.text)
        else:
            flush()
            buffer_texts = [section.text]
            buffer_heading_path = section.heading_path

    flush()
    return merged


def protect_tables(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        key = f"\x00TABLE{len(placeholders)}\x00"
        # Always pad with blank lines on both sides, regardless of how much
        # (if any) surrounding whitespace the match itself consumed -- a
        # table immediately followed by a paragraph with only a single
        # blank line between them would otherwise leave the placeholder
        # glued directly onto that paragraph's text with no separator,
        # breaking the paragraph split below.
        placeholders[key] = match.group(0).strip()
        return f"\n\n{key}\n\n"

    return TABLE_BLOCK_RE.sub(replace, text), placeholders


def split_into_blocks(text: str) -> list[str]:
    protected, placeholders = protect_tables(text)
    raw_blocks = re.split(r"\n\s*\n", protected)
    return [placeholders.get(block.strip(), block.strip()) for block in raw_blocks if block.strip()]


def pack_blocks(blocks: Sequence[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)
        if current and current_tokens + block_tokens > chunk_size:
            chunks.append("\n\n".join(current))

            overlap_blocks: list[str] = []
            overlap_tokens = 0
            for b in reversed(current):
                b_tokens = count_tokens(b)
                if overlap_tokens + b_tokens > overlap:
                    break
                overlap_blocks.insert(0, b)
                overlap_tokens += b_tokens
            current, current_tokens = overlap_blocks, overlap_tokens

        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def split_oversized(section: Section, chunk_size: int, overlap: int) -> list[Section]:
    """For a section still over the max token size, recursively split it
    (paragraph boundaries) with overlap -- but never at a point that falls
    inside a markdown table block. Tables stay whole even if that means a
    chunk exceeds the token target.
    """
    if count_tokens(section.text) <= chunk_size:
        return [section]

    blocks = split_into_blocks(section.text)
    sub_texts = pack_blocks(blocks, chunk_size, overlap)
    return [Section(heading_path=section.heading_path, text=t) for t in sub_texts]


def add_breadcrumb(section: Section, title: str | None) -> Section:
    """Prefix a chunk's text with "{document title} > {heading path}" so a
    standalone chunk doesn't lose the page/section context it needs to be
    understood on its own.
    """
    parts = [part for part in (title, section.heading_path) if part]
    breadcrumb = " > ".join(parts)
    text = f"{breadcrumb}\n\n{section.text}" if breadcrumb else section.text
    return Section(heading_path=section.heading_path, text=text)


def chunk_document(
    document: IngestedDocument, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    sections = split_by_headers(document.text or "")
    sections = merge_sections(sections, chunk_size)
    sections = [sub for section in sections for sub in split_oversized(section, chunk_size, overlap)]
    sections = [add_breadcrumb(section, document.title) for section in sections]

    return [
        Chunk(
            chunk_id=f"{document.document_id}::chunk-{i}",
            document_id=document.document_id,
            url=document.url,
            heading_path=section.heading_path,
            text=section.text,
            token_count=count_tokens(section.text),
        )
        for i, section in enumerate(sections)
    ]


def chunk_documents(
    documents: Sequence[IngestedDocument], *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    ]


def validate_chunks(documents: Sequence[IngestedDocument], chunks: Sequence[Chunk]) -> None:
    """Confirm no table got split across chunks -- a table's row count in the
    source document must equal the sum across that document's chunks.
    """

    def count_tables(text: str) -> int:
        return len(TABLE_BLOCK_RE.findall(text))

    for document in documents:
        original_table_count = count_tables(document.text or "")
        document_chunks = [chunk for chunk in chunks if chunk.document_id == document.document_id]
        chunked_table_count = sum(count_tables(chunk.text) for chunk in document_chunks)
        if chunked_table_count != original_table_count:
            raise ValueError(
                f"Table count mismatch for {document.document_id}: "
                f"{original_table_count} in source vs {chunked_table_count} across chunks "
                "-- a table may have been split."
            )


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
