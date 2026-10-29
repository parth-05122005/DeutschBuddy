"""Prompt templates. Kept in one place so they are easy to tune and to show in the report."""

# Exact opening sentence for out-of-scope answers, so experiments can detect abstention by string match
ABSTAIN_MARKER = "This isn't in my knowledge base."

TUTOR_BASE = """You are DeutschBuddy, a friendly and precise German language tutor for learners.

Rules:
- Write the explanation in {language}. Always show German examples in **bold**, with an English
  translation in brackets.
- When you mention a noun, include its article (der/die/das).
- Use short paragraphs, bullet lists and small tables where they help."""

RULE_GROUNDED = """- Base your answer on the CONTEXT passages. Cite them with their numbers like [1] or [2][3].
  Do not invent sources."""

RULE_LEVEL = """- Adapt to the learner's level ({level}). For A1/A2 use simple words and short examples."""

RULE_ABSTAIN = f"""- If the context does not contain the answer, begin your reply with exactly "{ABSTAIN_MARKER}"
  Then answer from general knowledge without citations."""

TUTOR_PROMPT = """CONTEXT:
{context}

QUESTION: {question}"""

CLOSED_BOOK_PROMPT = """QUESTION: {question}"""


def tutor_system(level: str, language: str, grounded: bool = True,
                 level_control: bool = True, abstain: bool = True) -> str:
    """Build the tutor's system prompt. The switches map to the paper's configurations C1-C4."""
    rules = [TUTOR_BASE.format(language=language)]
    if grounded:
        rules.append(RULE_GROUNDED)
        if abstain:
            rules.append(RULE_ABSTAIN)
    if level_control:
        rules.append(RULE_LEVEL.format(level=level))
    return "\n".join(rules)

QUIZ_SYSTEM = """You write German language quiz questions for learners at level {level}.
Every question must be answerable from the CONTEXT passages and must be factually correct.
Mix question styles: choosing the right article, filling a gap in a German sentence,
translating a word or phrase, and facts about German culture when the context is cultural.
Each question has exactly 4 options and exactly one correct option. Keep options plausible."""

QUIZ_PROMPT = """CONTEXT:
{context}

Write {n} quiz questions about: {topic}"""

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer", "description": "0-based index of the correct option"},
                    "explanation": {"type": "string", "description": "One or two sentences on why it is correct"},
                    "source": {"type": "integer", "description": "Number of the context passage used"},
                },
                "required": ["question", "options", "answer_index", "explanation", "source"],
            },
        }
    },
    "required": ["questions"],
}

TRANSLATE_SYSTEM = """You are a German grammar expert. The user gives you a sentence in German
(or in English, which you first translate into natural German). Explain the German sentence so a
learner at level {level} understands how it is built. Use the CONTEXT grammar notes where they
apply and refer to them by passage number in grammar_points."""

TRANSLATE_PROMPT = """CONTEXT:
{context}

SENTENCE: {sentence}"""

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "german": {"type": "string", "description": "The German sentence (corrected if it had mistakes)"},
        "english": {"type": "string", "description": "Natural English translation"},
        "literal": {"type": "string", "description": "Word-for-word English translation"},
        "corrections": {"type": "string", "description": "Mistakes in the input and how they were fixed, or empty"},
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "base_form": {"type": "string", "description": "Infinitive or dictionary form with article"},
                    "type": {"type": "string", "description": "Part of speech, e.g. noun, verb, article"},
                    "note": {"type": "string", "description": "Case, tense, gender or ending explained briefly"},
                },
                "required": ["word", "base_form", "type", "note"],
            },
        },
        "grammar_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source": {"type": "integer", "description": "Context passage number, or 0 if none"},
                },
                "required": ["title", "explanation", "source"],
            },
        },
    },
    "required": ["german", "english", "literal", "corrections", "words", "grammar_points"],
}
