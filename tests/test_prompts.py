from migrantbuddy.generation.prompts import (
    SYSTEM_PROMPT,
    build_prompt,
    build_query_rewrite_prompt,
    build_summary_prompt,
    detect_query_language,
)
from migrantbuddy.indexing import Chunk


def make_chunk(url: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{url}::chunk-0",
        document_id=url,
        url=url,
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


def test_build_prompt_includes_question():
    prompt = build_prompt("How much overtime pay am I entitled to?", [])

    assert "Question: How much overtime pay am I entitled to?" in prompt


def test_build_prompt_includes_each_chunk_source_and_text():
    chunks = [
        make_chunk("https://example.com/a", "text about salary"),
        make_chunk("https://example.com/b", "text about overtime"),
    ]

    prompt = build_prompt("query", chunks)

    assert "Source: https://example.com/a" in prompt
    assert "text about salary" in prompt
    assert "Source: https://example.com/b" in prompt
    assert "text about overtime" in prompt


def test_build_prompt_separates_multiple_chunks():
    chunks = [
        make_chunk("https://example.com/a", "first"),
        make_chunk("https://example.com/b", "second"),
    ]

    prompt = build_prompt("query", chunks)

    assert "---" in prompt


def test_build_prompt_handles_no_context_chunks():
    prompt = build_prompt("query", [])

    assert "Question: query" in prompt
    assert "Context:" in prompt


def test_system_prompt_instructs_answering_in_the_question_language():
    # Regression check: an earlier version omitted this instruction,
    # relying on the model inferring it unprompted -- which was unreliable
    # (same query, same model, would sometimes answer in English instead of
    # the query's language). See prompts.py's module docstring.
    assert "same language the question was asked in" in SYSTEM_PROMPT


def test_build_prompt_repeats_language_instruction_next_to_the_question():
    # Regression check: the system-prompt instruction alone wasn't reliable
    # enough in practice (long English context + an 8B quantized model would
    # still sometimes answer in English). Repeating the instruction right
    # before "Answer:" is the actual fix -- see prompts.py's module docstring.
    prompt = build_prompt("query", [])

    assert "answer ONLY in the same language as the question" in prompt
    assert prompt.index("answer ONLY in the same language as the question") > prompt.index(
        "Question: query"
    )
    assert prompt.rstrip().endswith("Answer:")


def test_system_prompt_forbids_defaulting_to_malay():
    # Regression check: the model answered an English question in Malay
    # despite the pre-existing language instruction -- this makes the rule
    # explicit rather than relying on the model to infer it never applies
    # to English-in/English-out.
    assert "Never switch to Malay" in SYSTEM_PROMPT
    assert "1. LANGUAGE" in SYSTEM_PROMPT


def test_detect_query_language_identifies_english():
    assert detect_query_language("How much overtime pay am I entitled to?") == "English"


def test_detect_query_language_returns_none_for_undetectable_text():
    assert detect_query_language("query") is None


def test_build_prompt_names_the_detected_language_in_the_reminder():
    # Regression check: the reminder previously only said "the same language
    # as the question" without naming it -- naming the actual detected
    # language is a harder constraint than asking the model to infer it.
    prompt = build_prompt("How much overtime pay am I entitled to?", [])

    assert "Rules reminder" in prompt
    assert "answer ONLY in English, matching the question above" in prompt
    assert prompt.index("Rules reminder") > prompt.index("Question: How much overtime")


def test_build_prompt_falls_back_to_generic_rule_when_language_undetectable():
    prompt = build_prompt("query", [])

    assert "Rules reminder" in prompt
    assert "answer ONLY in the same language as the question" in prompt


def test_system_prompt_caps_length_at_2_to_3_sentences():
    # Regression check: user-reported bug -- answers ran long. "2 to 4" was
    # loosened to "2-3" as part of the prose -> numbered-rules rewrite (see
    # module docstring) since giving the model more headroom was part of
    # what let answers run long in the first place.
    assert "2-3 sentences" in SYSTEM_PROMPT


def test_system_prompt_forbids_unrequested_extras():
    # Regression check: user-reported bug -- the model added examples/notes
    # nobody asked for.
    assert "NO EXTRAS" in SYSTEM_PROMPT
    assert "did not ask for" in SYSTEM_PROMPT


def test_system_prompt_requires_a_finished_sentence():
    # Regression check: user-reported bug -- answers trailed off unfinished.
    assert "COMPLETE" in SYSTEM_PROMPT
    assert "never trail off" in SYSTEM_PROMPT


def test_build_prompt_reminder_covers_all_four_rules():
    prompt = build_prompt("query", [])

    assert "1. LANGUAGE" in prompt
    assert "2. LENGTH" in prompt
    assert "3. NO EXTRAS" in prompt
    assert "4. COMPLETE" in prompt


def test_build_query_rewrite_prompt_includes_history_summary_and_followup():
    history = [
        {"role": "user", "content": "How much overtime pay if my salary is $3000?"},
        {"role": "assistant", "content": "Depends on whether you're a workman..."},
    ]

    prompt = build_query_rewrite_prompt(
        "Earlier: discussed salary basics.", history, "What about daily-rated workers?"
    )

    assert "Earlier: discussed salary basics." in prompt
    assert "How much overtime pay if my salary is $3000?" in prompt
    assert "Follow-up question: What about daily-rated workers?" in prompt


def test_build_query_rewrite_prompt_omits_empty_summary_block():
    prompt = build_query_rewrite_prompt("", [], "How much overtime pay am I entitled to?")

    assert "Summary of earlier conversation" not in prompt
    assert "Recent conversation" not in prompt
    assert "Follow-up question: How much overtime pay am I entitled to?" in prompt


def test_build_summary_prompt_includes_existing_summary_and_new_messages():
    messages = [
        {"role": "user", "content": "How much overtime pay if my salary is $3000?"},
        {"role": "assistant", "content": "You'd get 1.5x your hourly rate."},
    ]

    prompt = build_summary_prompt("User previously asked about salary.", messages)

    assert "Existing summary:\nUser previously asked about salary." in prompt
    assert "How much overtime pay if my salary is $3000?" in prompt
    assert "You'd get 1.5x your hourly rate." in prompt


def test_build_summary_prompt_omits_empty_existing_summary_block():
    prompt = build_summary_prompt("", [{"role": "user", "content": "hello"}])

    assert "Existing summary" not in prompt
