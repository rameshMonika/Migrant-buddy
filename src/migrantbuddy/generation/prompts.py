"""Grounded RAG prompt construction. Extracted from notebooks/05_generation.ipynb.

SYSTEM_PROMPT is a short, numbered, priority-ordered rule list rather than
prose -- an earlier prose version (multiple paragraphs, each rule buried in
justification text) was less reliable in practice: the model would
regularly drift on one rule or another (answering in Malay for an English
question, adding an unrequested example, running long, or trailing off
without finishing) even though every one of those rules was stated
somewhere in the prompt. A short numbered list an 8B quantized model can
actually hold onto turned out to matter more than how thoroughly each rule
was explained.

The language rule is still repeated a second time, right next to the
question itself in the user turn (see build_prompt) -- with a long,
100%-English context block ahead of the question, the model can lose track
of a system-level instruction by the time it starts generating. Repeating
it immediately before "Answer:" (right where generation begins) is a
well-known, more reliable way to steer output language than relying on the
system prompt alone. build_prompt also names the *actual detected*
language rather than just saying "the same language as the question" --
naming it explicitly is a harder constraint than asking the model to infer
it (see detect_query_language below).
"""

from typing import Sequence

from langdetect import DetectorFactory, LangDetectException, detect

from migrantbuddy.indexing import Chunk

# Deterministic detection -- langdetect's default is seeded from wall-clock
# time, which would make the same query detect differently across calls.
DetectorFactory.seed = 0

# Limited to languages SEA-LION targets (see CLAUDE.md Domain notes) plus
# English. langdetect conflates Malay/Indonesian (both tag as "id" on short
# text) -- labeled to say so explicitly rather than asserting the wrong one.
_LANGUAGE_NAMES = {
    "en": "English",
    "ms": "Malay",
    "id": "Malay or Indonesian",
    "ta": "Tamil",
    "my": "Burmese",
    "th": "Thai",
    "vi": "Vietnamese",
    "tl": "Filipino (Tagalog)",
}


def detect_query_language(query: str) -> str | None:
    """Best-effort language name for `query`, or None if detection fails or
    lands outside `_LANGUAGE_NAMES` -- callers should fall back to a generic
    instruction rather than asserting an unrecognized language.
    """
    try:
        code = detect(query)
    except LangDetectException:
        return None
    return _LANGUAGE_NAMES.get(code)


SYSTEM_PROMPT = """You are migrantBuddy, an assistant that answers questions about \
Singapore employment rules (work passes, salary, working hours) for migrant workers.

Follow these rules exactly, in priority order -- if any two would conflict, the \
lower-numbered rule wins:

1. LANGUAGE: answer in the exact same language the question was asked in, nothing \
else. If the question is in English, answer ONLY in English. Never switch to Malay \
or any other language just because the domain is Singapore/Southeast Asia -- that is \
not a reason to change language. Never add a translation or parenthetical in a \
second language, and never mix in words from another language. The provided context \
is always in English regardless of the question's language -- translate the \
relevant facts into the question's language yourself; do not answer in English \
because the context is in English.

2. LENGTH: 2-3 sentences, answering only what was actually asked. Nothing more.

3. GROUNDING: use ONLY facts stated in the provided context. If the context doesn't \
answer the question, say so plainly instead of guessing -- never invent or infer a \
figure, rule, or detail that isn't explicitly there.

4. NO EXTRAS: do not add an example, sample calculation, exception, caveat, or extra \
note the person did not ask for -- even a short "note that..." aside. Only include \
one if the question explicitly requests it. Do not use headers, bold text, or \
bullet lists unless the question requires comparing multiple items.

5. COMPLETE: always end on a fully finished sentence -- never trail off, never end \
mid-word, and never end with a dangling connector like "and", "however", or "in \
addition" as if more text follows. If you are running low on room, stop after your \
last complete sentence rather than starting a new thought you can't finish.

Write like a knowledgeable person giving a direct, spoken answer -- not a policy \
document."""


def build_prompt(query: str, context_chunks: Sequence[Chunk]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Source: {chunk.url}\n{chunk.text}" for chunk in context_chunks
    )
    language_name = detect_query_language(query)
    language_clause = (
        f"answer ONLY in {language_name}, matching the question above"
        if language_name
        else "answer ONLY in the same language as the question above"
    )
    reminder = (
        f"(Rules reminder -- 1. LANGUAGE: {language_clause}, never Malay or any "
        "other language by default, no translation alongside it. 2. LENGTH: 2-3 "
        "sentences, only what was asked. 3. NO EXTRAS: no example, calculation, or "
        "note unless explicitly requested. 4. COMPLETE: end on a finished "
        "sentence, never trail off.)"
    )
    return f"""Context:
{context_text}

Question: {query}

{reminder}

Answer:"""


def _format_turns(turns: Sequence[dict]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite a follow-up question into a standalone \
question that can be understood on its own, without needing the conversation history. \
Use the conversation history and summary (if given) only to fill in what the follow-up \
question is implicitly referring to -- do not answer the question, only rewrite it. \
Keep the rewritten question in the same language as the original follow-up question. \
Output ONLY the rewritten question, nothing else -- no preamble, no explanation."""


def build_query_rewrite_prompt(summary: str, history: Sequence[dict], message: str) -> str:
    """history is the recent turns kept verbatim ({"role": "user"/"assistant",
    "content": str}); summary (may be empty) covers anything older than that.
    If there's no history and no summary yet (first turn), rewriting is a
    no-op -- callers should skip calling this and use `message` directly.
    """
    summary_block = f"Summary of earlier conversation:\n{summary}\n\n" if summary else ""
    history_block = f"Recent conversation:\n{_format_turns(history)}\n\n" if history else ""
    return f"""{summary_block}{history_block}Follow-up question: {message}

Rewrite the follow-up question above as a standalone question:"""


SUMMARY_SYSTEM_PROMPT = """You summarize a conversation between a migrant worker and \
an assistant about Singapore employment rules (salary, work permits, medical \
insurance, working hours). Keep specific facts that later questions might depend on \
-- figures (salary amounts, hours), categories (workman/non-workman), dates, and what \
was already answered. Write the summary in at most 3-4 short sentences -- this is a \
running memory aid, not a restatement of every detail. Output ONLY the summary, \
nothing else."""


def build_summary_prompt(existing_summary: str, messages_to_summarize: Sequence[dict]) -> str:
    """Folds any existing summary into the new one along with the newly-old
    messages, so re-summarizing as the conversation grows doesn't silently
    drop what was captured in an earlier summarization pass.
    """
    summary_block = f"Existing summary:\n{existing_summary}\n\n" if existing_summary else ""
    return f"""{summary_block}Conversation to fold into the summary:
{_format_turns(messages_to_summarize)}

Updated summary:"""
