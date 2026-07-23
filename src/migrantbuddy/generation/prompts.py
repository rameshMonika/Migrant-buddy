"""Grounded RAG prompt construction. Extracted from notebooks/05_generation.ipynb.

Explicitly instructs the model to answer in the query's language, in two
places: the system prompt, and again right next to the question itself in
the user turn. The system-prompt instruction alone was not enough in
practice -- with a long, 100%-English context block ahead of the question,
an 8B quantized model can lose track of a system-level instruction by the
time it starts generating. Repeating it immediately before "Answer:" (right
where generation begins) is a well-known, more reliable way to steer output
language than relying on the system prompt alone.
"""

from typing import Sequence

from migrantbuddy.indexing import Chunk

SYSTEM_PROMPT = """You are migrantBuddy, an assistant that answers questions about \
Singapore employment rules (work passes, salary, working hours) for migrant workers.

Answer ONLY using the information in the provided context. If the context does not \
contain enough information to answer the question, say so clearly instead of \
guessing. Do not use any outside knowledge. Keep answers clear and concise, \
suitable for someone who may not be a native English speaker.

Always answer in the same language the question was asked in. The provided \
context will always be in English regardless of the question's language -- \
translate the relevant information into the question's language rather than \
answering in English."""


def build_prompt(query: str, context_chunks: Sequence[Chunk]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Source: {chunk.url}\n{chunk.text}" for chunk in context_chunks
    )
    return f"""Context:
{context_text}

Question: {query}

(Remember: answer in the same language as the question above, even though \
the context is in English.)

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
