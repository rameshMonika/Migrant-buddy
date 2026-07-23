import pytest

from migrantbuddy.indexing.chunking import (
    Section,
    add_breadcrumb,
    chunk_document,
    count_tokens,
    merge_sections,
    pack_blocks,
    protect_tables,
    split_by_headers,
    split_into_blocks,
    split_oversized,
    validate_chunks,
)
from migrantbuddy.ingestion import IngestedDocument


def make_document(text: str, *, title: str | None = "Doc Title", document_id: str = "doc-1") -> IngestedDocument:
    return IngestedDocument(
        document_id=document_id,
        url=f"https://example.com/{document_id}",
        title=title,
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text=text,
    )


def test_count_tokens_is_chars_over_four():
    assert count_tokens("a" * 40) == 10


def test_split_by_headers_attaches_nested_heading_path():
    text = "Intro text.\n\n## Section A\n\nBody A.\n\n### Sub A1\n\nBody A1."

    sections = split_by_headers(text)

    assert [s.heading_path for s in sections] == ["", "Section A", "Section A > Sub A1"]
    assert sections[1].text == "Body A."
    assert sections[2].text == "Body A1."


def test_split_by_headers_tolerates_leading_whitespace_before_hash():
    text = "  ## Indented Heading\n\nBody."

    sections = split_by_headers(text)

    assert sections[0].heading_path == "Indented Heading"


def test_split_by_headers_pops_stack_on_sibling_heading():
    text = "## A\n\nBody A.\n\n### A1\n\nBody A1.\n\n## B\n\nBody B."

    sections = split_by_headers(text)

    assert [s.heading_path for s in sections] == ["A", "A > A1", "B"]


def test_merge_sections_combines_small_sections_under_target():
    sections = [Section("A", "short"), Section("B", "also short")]

    merged = merge_sections(sections, chunk_size=500)

    assert len(merged) == 1
    assert merged[0].heading_path == "A"
    assert "short" in merged[0].text and "also short" in merged[0].text


def test_merge_sections_keeps_large_sections_separate():
    big_text = "word " * 200  # ~1000 chars -> 250 tokens
    sections = [Section("A", big_text), Section("B", big_text)]

    merged = merge_sections(sections, chunk_size=300)

    assert len(merged) == 2


def test_protect_tables_replaces_table_block_with_placeholder():
    text = "before\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nafter"

    protected, placeholders = protect_tables(text)

    assert len(placeholders) == 1
    (placeholder_key, placeholder_value) = next(iter(placeholders.items()))
    assert placeholder_key in protected
    assert "| a | b |" in placeholder_value


def test_split_into_blocks_keeps_table_as_single_block():
    text = "para one\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\npara two"

    blocks = split_into_blocks(text)

    assert blocks == ["para one", "| a | b |\n| - | - |\n| 1 | 2 |", "para two"]


def test_pack_blocks_carries_overlap_into_the_next_chunk():
    block_a, block_b, block_c = "A" * 40, "B" * 40, "C" * 40  # 10 tokens each

    packed = pack_blocks([block_a, block_b, block_c], chunk_size=15, overlap=10)

    assert packed == [block_a, f"{block_a}\n\n{block_b}", f"{block_b}\n\n{block_c}"]


def test_split_oversized_returns_section_unchanged_when_under_limit():
    section = Section("A", "short text")

    result = split_oversized(section, chunk_size=500, overlap=100)

    assert result == [section]


def test_split_oversized_never_splits_a_table():
    table = "| a | b |\n| - | - |\n| 1 | 2 |"
    text = "word " * 200 + "\n\n" + table + "\n\n" + "word " * 200
    section = Section("A", text)

    result = split_oversized(section, chunk_size=100, overlap=20)

    # Overlap may legitimately repeat the table whole across two adjacent
    # chunks -- that's not a split. What must never happen is a *partial*
    # table (e.g. only the header row) showing up in any chunk.
    assert any(table in r.text for r in result)
    for r in result:
        assert ("| a | b |" in r.text) == (table in r.text)


def test_add_breadcrumb_combines_title_and_heading_path():
    section = Section("Section A", "body text")

    result = add_breadcrumb(section, "Doc Title")

    assert result.text == "Doc Title > Section A\n\nbody text"


def test_add_breadcrumb_handles_missing_heading_path():
    section = Section("", "body text")

    result = add_breadcrumb(section, "Doc Title")

    assert result.text == "Doc Title\n\nbody text"


def test_add_breadcrumb_handles_missing_title():
    section = Section("Section A", "body text")

    result = add_breadcrumb(section, None)

    assert result.text == "Section A\n\nbody text"


def test_chunk_document_produces_ids_and_preserves_table():
    table = "| a | b |\n| - | - |\n| 1 | 2 |"
    text = f"## Overview\n\nSome intro text.\n\n## Details\n\n{table}\n\nMore text."
    document = make_document(text, document_id="my-doc")

    chunks = chunk_document(document)

    assert all(chunk.chunk_id.startswith("my-doc::chunk-") for chunk in chunks)
    assert all(chunk.document_id == "my-doc" for chunk in chunks)
    assert sum(table in chunk.text for chunk in chunks) == 1


def test_validate_chunks_passes_when_table_counts_match():
    table = "| a | b |\n| - | - |\n| 1 | 2 |"
    text = f"## Details\n\n{table}\n\nMore text."
    document = make_document(text)
    chunks = chunk_document(document)

    validate_chunks([document], chunks)  # should not raise


def test_validate_chunks_raises_on_table_count_mismatch():
    document = make_document("## Details\n\n| a | b |\n| - | - |\n| 1 | 2 |")
    chunks = chunk_document(document)
    # Corrupt a chunk so its entire table block got dropped, simulating a
    # splitting bug -- removing only one row would still count as "1 table"
    # (validate_chunks counts table blocks, not rows) and wouldn't trigger
    # the mismatch this test is checking for.
    chunks[0].text = chunks[0].text.replace("| a | b |\n| - | - |\n| 1 | 2 |", "")

    with pytest.raises(ValueError, match="Table count mismatch"):
        validate_chunks([document], chunks)
