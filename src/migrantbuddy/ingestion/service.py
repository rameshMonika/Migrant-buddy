"""Document ingestion: fetch MOM.gov.sg pages, extract main content, assemble
structured records. Extracted from notebooks/01_ingestion.ipynb -- see that
notebook for the experiment history behind these defaults.
"""

import json
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import requests
import trafilatura

DEFAULT_USER_AGENT = "migrantBuddy-ingestion/0.1 (research prototype)"

# Constant for now -- all sources are MOM guidance pages -- kept as a separate
# field for when other source types (FAQs, forms, third-party guidance) are
# added later.
CONTENT_TYPE = "official_guidance"


@dataclass(frozen=True)
class SourceSpec:
    url: str
    category: str


# category is assigned per-URL (not a blanket constant) since the corpus spans
# multiple real categories. work-injury (WICA) and housing were dropped --
# both are landing/hub pages with ~90% navigation and no substantive content
# of their own; the real content lives on un-sourced sub-pages. See CLAUDE.md
# Domain notes for deferred categories.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        url="https://www.mom.gov.sg/employment-practices/salary/paying-salary",
        category="salary",
    ),
    SourceSpec(
        url="https://www.mom.gov.sg/employment-practices/hours-of-work-overtime-and-rest-days",
        category="working-hours",
    ),
    SourceSpec(
        url=(
            "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/"
            "sector-specific-rules/work-permit-conditions"
        ),
        category="work-permit",
    ),
    SourceSpec(
        url=(
            "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/"
            "sector-specific-rules/medical-insurance"
        ),
        category="medical",
    ),
    SourceSpec(url="https://www.mom.gov.sg/contact-us", category="help"),
)


@dataclass
class IngestedDocument:
    document_id: str
    url: str
    title: str | None
    authority: str
    category: str
    content_type: str
    language: str
    source_type: str
    fetched_at: str
    text: str | None


def slugify(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-")


def fetch_html(url: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 15.0) -> str:
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    return response.text


def cache_raw_html(raw_dir: Path, url: str, html: str) -> Path:
    """Save untouched HTML to raw_dir before any parsing -- keeps ingestion
    reproducible/offline-repeatable and gives a debugging reference if
    extraction looks wrong later.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{slugify(url)}.html"
    path.write_text(html, encoding="utf-8")
    return path


def extract_text(html: str) -> str | None:
    """Strip nav/footer/boilerplate, keep substantive page text as markdown
    (tables inline, correctly positioned relative to surrounding headings).
    """
    return trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        favor_precision=False,
    )


def build_document(source: SourceSpec, html: str) -> IngestedDocument:
    metadata = trafilatura.extract_metadata(html)
    return IngestedDocument(
        document_id=slugify(source.url),
        url=source.url,
        title=metadata.title if metadata else None,
        authority="MOM",
        category=source.category,
        content_type=CONTENT_TYPE,
        language="en",
        source_type="html",
        fetched_at=datetime.now(timezone.utc).isoformat(),
        text=extract_text(html),
    )


def ingest_sources(
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES, *, raw_dir: Path
) -> list[IngestedDocument]:
    """Fetch, cache, and parse each source into an IngestedDocument. Only a
    handful of pages, so no concurrency/rate-limiting machinery here.
    """
    documents = []
    for source in sources:
        html = fetch_html(source.url)
        cache_raw_html(raw_dir, source.url, html)
        documents.append(build_document(source, html))
    return documents


def save_ingested(documents: Sequence[IngestedDocument], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(document) for document in documents], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def validate_documents(documents: Sequence[IngestedDocument], *, min_text_length: int = 500) -> None:
    """Confirm every document produced non-trivial text and nothing silently
    failed (e.g. a page reduced to near-empty text after boilerplate
    stripping). Raises on a genuine failure; warns on a merely-suspicious but
    possibly-legitimate case (e.g. a short contact page with no headings).
    """
    for document in documents:
        if not document.title:
            raise ValueError(f"Missing title: {document.url}")
        if not document.text:
            raise ValueError(f"Empty text: {document.url}")
        if len(document.text) < min_text_length:
            raise ValueError(
                f"Suspiciously short text ({len(document.text)} chars): {document.url}"
            )
        if "#" not in document.text:
            warnings.warn(f"No heading structure found (may be legitimate): {document.url}")
