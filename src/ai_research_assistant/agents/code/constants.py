import re

MAX_CODE_EVIDENCE_ITEMS = 6
MAX_CODE_EVIDENCE_CONTENT_LENGTH = 2_000
CODE_AGENT_RECURSION_LIMIT = 6
TECHNICAL_TERM_PATTERN = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9_.+#-]{1,}(?!\w)")
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
REPOSITORY_STARS_PATTERN = re.compile(r"^Stars:\s*(\d+)", re.MULTILINE)
PYTHON_FROM_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+([^\n]+)$",
    re.MULTILINE,
)
PYTHON_IMPORT_PATTERN = re.compile(
    r"^\s*import\s+([A-Za-z_][\w.]*)",
    re.MULTILINE,
)
GENERIC_REPOSITORY_TERMS = {
    "an",
    "and",
    "api",
    "build",
    "code",
    "compilation",
    "compile",
    "create",
    "edge",
    "example",
    "examples",
    "for",
    "from",
    "framework",
    "integration",
    "library",
    "minimal",
    "node",
    "on",
    "one",
    "python",
    "rest",
    "sdk",
    "show",
    "the",
    "to",
    "use",
    "using",
    "with",
    "write",
}
INSUFFICIENT_CODE_RESPONSES = {
    "uk": (
        "Не вдалося підготувати надійний приклад коду на основі знайденої "
        "офіційної документації. Уточни технологію, версію або потрібний API."
    ),
    "en": (
        "A reliable code example could not be produced from the retrieved official "
        "documentation. Specify the technology, version, or API you need."
    ),
}
SAFE_STANDARD_LIBRARY_MODULES = {
    "asyncio",
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "json",
    "os",
    "pathlib",
    "re",
    "sys",
    "typing",
    "typing_extensions",
    "uuid",
}
DOCUMENTATION_HOST_HINTS = ("docs.", "documentation.", "reference.", "readthedocs")
NON_DOCUMENTATION_HOSTS = {
    "github.com",
    "opensource.org",
    "pypi.org",
    "pypistats.org",
    "shields.io",
    "www.langchain.com",
    "x.com",
}
