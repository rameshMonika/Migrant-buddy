from migrantbuddy.generation.truncation import trim_to_last_complete_sentence


def test_trims_back_to_last_complete_sentence():
    text = "First complete sentence. Second sentence cut off mid-wo"

    assert trim_to_last_complete_sentence(text) == "First complete sentence."


def test_leaves_text_unchanged_when_it_already_ends_cleanly():
    text = "A single complete sentence."

    assert trim_to_last_complete_sentence(text) == text


def test_leaves_text_unchanged_when_no_sentence_boundary_exists():
    # Nothing safe to trim to -- a dangling answer beats an empty one.
    text = "no punctuation anywhere in this fragment"

    assert trim_to_last_complete_sentence(text) == text


def test_handles_multiple_complete_sentences_before_the_cutoff():
    text = "First. Second! Third cut off mid-w"

    assert trim_to_last_complete_sentence(text) == "First. Second!"


def test_recognizes_non_latin_sentence_terminators():
    # Burmese full stop.
    text = "စာက်ခံးပြီး။ ကတ်စက်"

    trimmed = trim_to_last_complete_sentence(text)

    assert trimmed == "စာက်ခံးပြီး။"


def test_empty_string_returns_empty_string():
    assert trim_to_last_complete_sentence("") == ""


def test_whitespace_only_returns_empty_string():
    assert trim_to_last_complete_sentence("   ") == ""
