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
guessing. Do not use any outside knowledge, and do not add your own details, \
figures, or explanations that are not explicitly stated in the context -- even when \
being brief, never fill a gap with something that sounds plausible. Every fact you \
state must come directly from the context.

Answer directly and briefly -- 2 to 4 sentences for the common case, addressing what \
was actually asked. Do not enumerate every related rule, exception, or work \
arrangement unless the question specifically asks you to compare or list them. Do \
not use headers, bold text, or bullet lists unless the question truly requires \
comparing multiple items. Do not give worked examples, sample calculations, or \
hypothetical scenarios unless the person explicitly asks for one. Answers should read \
like a direct reply from a knowledgeable person, not a policy document.

Always finish your answer as a complete thought within that length -- never cut off \
mid-sentence or mid-word, and never end with a dangling connector (e.g. "and", \
"however", "in addition") as if the answer continues onto another line. The answer \
must read as fully self-contained and complete on its own, from the first word to \
the last, in whatever language you are answering in -- this applies equally to every \
language, not just English. If there's an important exception that likely applies to \
the person asking (e.g. shift work, overtime), only mention it, in one short \
sentence, once your main answer is already complete -- if you are running low on \
room, leave it out entirely rather than risk cutting off the main answer to fit it in.

Always answer in the same language the question was asked in. The provided \
context will always be in English regardless of the question's language -- \
translate the relevant information into the question's language rather than \
answering in English. Reply ONLY in that language -- never add an English \
translation or parenthetical alongside it."""


def build_prompt(query: str, context_chunks: Sequence[Chunk]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Source: {chunk.url}\n{chunk.text}" for chunk in context_chunks
    )
    return f"""Context:
{context_text}

Question: {query}

(Remember: answer ONLY in the same language as the question above, even though \
the context is in English -- no English translation or parenthetical alongside it. \
Keep it brief and finish as a complete thought -- do not cut off mid-sentence. Do \
NOT add a worked example, sample calculation, or hypothetical scenario -- only give \
one if the question above explicitly asks for it.)

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
