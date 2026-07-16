import re

from ai_research_assistant.models import SourceType

OVERVIEW_SOURCE_HINT = "official documentation overview features architecture"
RELEASE_SOURCE_HINT = "official release notes changelog current version"
LIMITATION_SOURCE_HINT = "official documentation limitations tradeoffs when to use an alternative"
MAX_AGENT_EVIDENCE_CONTENT_LENGTH = 1_200
MAX_SYNTHESIS_EVIDENCE_CONTENT_LENGTH = 1_000
MAX_SYNTHESIS_EVIDENCE_ITEMS = 3
TECHNICAL_TERM_PATTERN = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9_.+#-]{1,}(?!\w)")
RELEASE_INTENT_PATTERN = re.compile(
    r"\b(?:version|versions|release|releases|changelog)\b|версі\w*|реліз\w*",
    re.IGNORECASE,
)
LIMITATION_INTENT_PATTERN = re.compile(
    r"\b(?:limitation|limitations|drawback|drawbacks|disadvantage|disadvantages|"
    r"tradeoff|tradeoffs|weakness|weaknesses|cons)\b|"
    r"обмежен\w*|мінус\w*|недолік\w*|слабк\w*",
    re.IGNORECASE,
)
EXPLICIT_LIMITATION_EVIDENCE_PATTERN = re.compile(
    r"\b(?:limitation|limitations|drawback|drawbacks|disadvantage|disadvantages|"
    r"trade-?off|tradeoffs|does not|do not|doesn't|cannot|can't|not designed|"
    r"not intended|not supported|instead|alternative|recommend(?:ed|s)?|requires?)\b|"
    r"обмежен\w*|недолік\w*|не\s+підтрим\w*|не\s+призначен\w*|натомість|"
    r"альтернатив\w*|рекоменду\w*|потребу\w*",
    re.IGNORECASE,
)
LOW_LEVEL_ALTERNATIVE_PATTERN = re.compile(
    r"\blow-level\b.*\b(?:higher-level|prebuilt|instead|recommend(?:ed|s)?)\b|"
    r"\b(?:higher-level|prebuilt|instead|recommend(?:ed|s)?)\b.*\blow-level\b",
    re.IGNORECASE,
)
PERSONAL_EXPERIENCE_PATTERN = re.compile(
    r"\b(?:I|I've|I'd|I'm|me|my|mine|we|we've|we'd|we're|our|ours)\b",
    re.IGNORECASE,
)
EXPERIENCE_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:author|user|developer|reviewer|community|post|reported|described|"
    r"argued|according)\b|автор\w*|користувач\w*|розробник\w*|рецензент\w*|"
    r"спільнот\w*|допис\w*|повідом\w*|опис\w*|за\s+(?:словами|оцінкою)",
    re.IGNORECASE,
)
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MARKDOWN_TOKEN_PATTERN = re.compile(r"(?:#{1,6}|[*_`>|])+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_PATTERN = re.compile(r"\s+")

SUBJECT_STOPWORDS = {
    "and",
    "about",
    "capabilities",
    "capability",
    "architecture",
    "current",
    "describe",
    "documented",
    "documentation",
    "does",
    "explain",
    "feature",
    "features",
    "for",
    "framework",
    "how",
    "latest",
    "main",
    "of",
    "official",
    "the",
    "to",
    "overview",
    "purpose",
    "technology",
    "using",
    "what",
    "with",
    "work",
    "works",
}
INFORMATION_MARKERS = (
    " is ",
    " are ",
    " provides ",
    " supports ",
    " enables ",
    " designed ",
    " focused ",
    " uses ",
)
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
URL_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
SOURCE_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:(?:official|verified)\s+sources|sources|"
    r"(?:офіційні|перевірені)?\s*джерела)\s*:?\s*$",
    re.IGNORECASE,
)
SOURCE_TYPE_PRIORITY: dict[SourceType, int] = {
    "documentation": 0,
    "github": 1,
    "release_notes": 2,
    "web": 3,
}
INSUFFICIENT_EVIDENCE_RESPONSES = {
    "uk": (
        "Не вдалося сформувати надійну відповідь на основі знайдених джерел. "
        "Спробуй уточнити запит або вказати конкретну технологію чи версію."
    ),
    "en": (
        "A reliable answer could not be produced from the retrieved sources. "
        "Try making the request more specific or naming a technology or version."
    ),
}
