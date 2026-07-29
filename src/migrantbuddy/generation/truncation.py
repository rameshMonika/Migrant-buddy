"""Trims a truncated generation back to its last complete sentence, instead
of returning a fragment that ends mid-word/mid-clause. SYSTEM_PROMPT
instructs the model to always finish as a complete thought, but that's
guidance, not a guarantee -- confirmed in practice (a Tamil answer cut off
mid-word despite the instruction) -- so this is the backend-side safety net
for when the model still runs past GENERATION_MAX_TOKENS before wrapping up.

Only meant to be called when the backend actually reports the output was
cut off by the token limit (not a natural stop) -- see
ollama_client.py/vllm_client.py, which check done_reason/finish_reason
before calling this. Ruled out simpler explanations first: no stop
sequences are configured anywhere in generation/, and no code path
byte-slices the response text (which could otherwise corrupt multi-byte
UTF-8 text mid-character) -- this really is the model running out of
tokens mid-sentence.

Punctuation-based, not NLP-based -- covers the sentence-ending marks used
across SEA-LION's target languages (Latin ".", "!", "?"; CJK full-width
equivalents; Burmese "၊"/"။"; Arabic "؟"). Deliberately conservative: if no
sentence boundary is found at all, returns the text unchanged rather than
guessing -- a still-dangling answer is better than silently returning
nothing.
"""

_SENTENCE_END_CHARS = ".!?。!?؟၊။"


def trim_to_last_complete_sentence(text: str) -> str:
    text = text.rstrip()
    if not text or text[-1] in _SENTENCE_END_CHARS:
        return text

    last_end = max((text.rfind(char) for char in _SENTENCE_END_CHARS), default=-1)
    if last_end == -1:
        return text

    return text[: last_end + 1]
