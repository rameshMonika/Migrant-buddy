from pathlib import Path

import pytest

from migrantbuddy.ingestion import SourceSpec, build_document, save_ingested, slugify, validate_documents
from migrantbuddy.ingestion.service import IngestedDocument, ingest_sources

SAMPLE_HTML = """
<html>
<head><title>Paying salary</title></head>
<body>
<article>
<h1>Paying salary</h1>
<p>Your employer must pay your salary at least once a month and within 7 days
after the end of the salary period. There are exceptions for overtime,
resignation without notice and other situations.</p>
<h2>How often salary must be paid</h2>
<p>If you are covered by the Employment Act, your employer must pay your
salary at least once a month. They can also pay it at shorter intervals if
they choose. Salary must be paid within 7 days after the end of the salary
period, and for overtime work, within 14 days after the end of the salary
period. This section intentionally repeats itself a little to push the
extracted text comfortably past the minimum length threshold used in
ingestion validation, since trafilatura strips most boilerplate and short
fixture pages can otherwise end up too short to pass that check.</p>
</article>
</body>
</html>
"""


def test_slugify_strips_scheme_and_punctuation():
    assert slugify("https://www.mom.gov.sg/contact-us") == "https-www-mom-gov-sg-contact-us"


def test_slugify_collapses_runs_of_punctuation():
    assert slugify("https://a.b//c--d") == "https-a-b-c-d"


def test_build_document_populates_expected_fields():
    source = SourceSpec(url="https://www.mom.gov.sg/employment-practices/salary/paying-salary", category="salary")

    document = build_document(source, SAMPLE_HTML)

    assert document.document_id == slugify(source.url)
    assert document.url == source.url
    assert document.category == "salary"
    assert document.content_type == "official_guidance"
    assert document.language == "en"
    assert document.source_type == "html"
    assert document.title
    assert document.text and "salary" in document.text.lower()


def test_validate_documents_passes_for_healthy_document():
    document = IngestedDocument(
        document_id="doc-1",
        url="https://example.com/doc-1",
        title="Doc 1",
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text="# Heading\n\n" + ("word " * 200),
    )

    validate_documents([document])  # should not raise


def test_validate_documents_rejects_missing_title():
    document = IngestedDocument(
        document_id="doc-1",
        url="https://example.com/doc-1",
        title=None,
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text="# Heading\n\n" + ("word " * 200),
    )

    with pytest.raises(ValueError, match="Missing title"):
        validate_documents([document])


def test_validate_documents_rejects_short_text():
    document = IngestedDocument(
        document_id="doc-1",
        url="https://example.com/doc-1",
        title="Doc 1",
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text="too short",
    )

    with pytest.raises(ValueError, match="Suspiciously short text"):
        validate_documents([document])


def test_validate_documents_warns_on_missing_heading_structure():
    document = IngestedDocument(
        document_id="doc-1",
        url="https://example.com/doc-1",
        title="Doc 1",
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text="word " * 200,
    )

    with pytest.warns(UserWarning, match="No heading structure"):
        validate_documents([document])


def test_ingest_sources_fetches_caches_and_builds_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = SourceSpec(url="https://www.mom.gov.sg/employment-practices/salary/paying-salary", category="salary")

    monkeypatch.setattr("migrantbuddy.ingestion.service.fetch_html", lambda url, **kwargs: SAMPLE_HTML)

    documents = ingest_sources([source], raw_dir=tmp_path)

    assert len(documents) == 1
    assert documents[0].category == "salary"
    cached_files = list(tmp_path.glob("*.html"))
    assert len(cached_files) == 1
    assert cached_files[0].read_text(encoding="utf-8") == SAMPLE_HTML


def test_save_ingested_round_trips_through_json(tmp_path: Path):
    document = IngestedDocument(
        document_id="doc-1",
        url="https://example.com/doc-1",
        title="Doc 1",
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text="# Heading\n\nSome text.",
    )
    output_path = tmp_path / "nested" / "ingested.json"

    result_path = save_ingested([document], output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert '"document_id": "doc-1"' in output_path.read_text(encoding="utf-8")
