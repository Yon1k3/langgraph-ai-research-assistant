import ast
import builtins
import re

from ai_research_assistant.agents.code.constants import (
    MARKDOWN_LINK_PATTERN,
    PYTHON_FROM_IMPORT_PATTERN,
    PYTHON_IMPORT_PATTERN,
    SAFE_STANDARD_LIBRARY_MODULES,
    URL_PATTERN,
)
from ai_research_assistant.agents.code.evidence import _get_required_code_tokens
from ai_research_assistant.errors import InvalidCodeResultError
from ai_research_assistant.models import CodeSynthesis, EvidenceItem


def _is_code_supported_by_evidence(
    query: str,
    synthesis: CodeSynthesis,
    evidence: list[EvidenceItem],
) -> bool:
    evidence_by_id = {item.source_id: item for item in evidence}

    if any(source_id not in evidence_by_id for source_id in synthesis.used_source_ids):
        return False

    used_evidence = [evidence_by_id[source_id] for source_id in synthesis.used_source_ids]
    evidence_text = "\n".join(
        f"{item.source.title}\n{item.source.url}\n{item.content}" for item in used_evidence
    )
    normalized_evidence = evidence_text.casefold()

    if synthesis.code_language != "python":
        return True

    try:
        syntax_tree = ast.parse(synthesis.code)
    except SyntaxError:
        return False

    if _get_undefined_python_names(syntax_tree):
        return False

    code_tokens = {node.id for node in ast.walk(syntax_tree) if isinstance(node, ast.Name)}
    code_tokens.update(
        node.attr for node in ast.walk(syntax_tree) if isinstance(node, ast.Attribute)
    )
    code_tokens.update(
        alias.asname or alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )

    for required_token in _get_required_code_tokens(query):
        if required_token not in code_tokens or required_token not in evidence_text:
            return False

    for module in PYTHON_IMPORT_PATTERN.findall(synthesis.code):
        top_level_module = module.split(".", maxsplit=1)[0]

        if (
            top_level_module not in SAFE_STANDARD_LIBRARY_MODULES
            and top_level_module.casefold() not in normalized_evidence
        ):
            return False

    for module, imported_names in PYTHON_FROM_IMPORT_PATTERN.findall(synthesis.code):
        top_level_module = module.split(".", maxsplit=1)[0]

        if top_level_module in SAFE_STANDARD_LIBRARY_MODULES:
            continue

        if top_level_module.casefold() not in normalized_evidence:
            return False

        for imported_name in imported_names.strip("() ").split(","):
            symbol = imported_name.strip().split(" as ", maxsplit=1)[0].strip()

            if symbol and symbol != "*" and symbol not in evidence_text:
                return False

    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue

        constructor_name = node.func.id

        if (
            constructor_name[:1].isupper()
            and not node.args
            and not node.keywords
            and f"{constructor_name}()" not in evidence_text
        ):
            return False

    return True


def _attach_supporting_sources(
    query: str,
    synthesis: CodeSynthesis,
    evidence: list[EvidenceItem],
) -> CodeSynthesis:
    """Replace model-selected IDs with the smallest evidence set supporting the code."""

    required_terms = _get_code_evidence_terms(query, synthesis)

    if not required_terms:
        return synthesis

    uncovered = set(required_terms)
    selected_ids: list[str] = []
    remaining = list(evidence)
    api_terms = {
        term
        for term in required_terms
        if "." not in term and term not in SAFE_STANDARD_LIBRARY_MODULES
    }

    while uncovered and remaining:
        candidates = [item for item in remaining if any(term in item.content for term in uncovered)]

        if not candidates:
            break

        best = max(
            candidates,
            key=lambda item: _get_supporting_evidence_rank(
                item,
                uncovered,
                api_terms,
            ),
        )
        covered = {term for term in uncovered if term in best.content}

        if not covered:
            break

        selected_ids.append(best.source_id)
        uncovered -= covered
        remaining.remove(best)

    if uncovered:
        return synthesis

    return synthesis.model_copy(update={"used_source_ids": selected_ids})


def _get_supporting_evidence_rank(
    evidence: EvidenceItem,
    uncovered_terms: set[str],
    api_terms: set[str],
) -> tuple[int, int, int, float]:
    label = f"{evidence.source.title} {evidence.source.url}".casefold()
    label_matches = sum(term.casefold() in label for term in api_terms)
    matched_code = int("Matched content:" in evidence.content)
    content_matches = sum(term in evidence.content for term in uncovered_terms)

    return (
        label_matches,
        matched_code,
        content_matches,
        evidence.score,
    )


def _get_code_evidence_terms(
    query: str,
    synthesis: CodeSynthesis,
) -> list[str]:
    terms = list(_get_required_code_tokens(query))

    if synthesis.code_language != "python":
        return terms

    for module in PYTHON_IMPORT_PATTERN.findall(synthesis.code):
        terms.append(module)

    for module, imported_names in PYTHON_FROM_IMPORT_PATTERN.findall(synthesis.code):
        terms.append(module)
        terms.extend(
            symbol
            for imported_name in imported_names.strip("() ").split(",")
            if (symbol := imported_name.strip().split(" as ", maxsplit=1)[0].strip())
            and symbol != "*"
        )

    return list(dict.fromkeys(terms))


def _get_undefined_python_names(syntax_tree: ast.AST) -> set[str]:
    defined_names = set(dir(builtins))
    loaded_names: set[str] = set()

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded_names.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Param)):
                defined_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Import):
            defined_names.update(
                alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            defined_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.arg):
            defined_names.add(node.arg)

    return loaded_names - defined_names


def _get_code_validation_feedback(
    query: str,
    synthesis: CodeSynthesis,
    evidence: list[EvidenceItem],
) -> list[str]:
    feedback: list[str] = []
    evidence_by_id = {item.source_id: item for item in evidence}
    unknown_source_ids = [
        source_id for source_id in synthesis.used_source_ids if source_id not in evidence_by_id
    ]

    if unknown_source_ids:
        feedback.append("Remove unknown source IDs: " + ", ".join(unknown_source_ids))

    if synthesis.code_language != "python":
        return feedback or ["Use only API elements explicitly present in evidence"]

    try:
        syntax_tree = ast.parse(synthesis.code)
    except SyntaxError as exc:
        feedback.append(f"Fix Python syntax: {exc.msg}")
        return feedback

    undefined_names = sorted(_get_undefined_python_names(syntax_tree))

    if undefined_names:
        feedback.append(
            "Import or define every unresolved Python name: " + ", ".join(undefined_names)
        )

    code_tokens = {node.id for node in ast.walk(syntax_tree) if isinstance(node, ast.Name)}
    code_tokens.update(
        node.attr for node in ast.walk(syntax_tree) if isinstance(node, ast.Attribute)
    )
    code_tokens.update(
        alias.asname or alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    used_evidence_text = "\n".join(
        evidence_by_id[source_id].content
        for source_id in synthesis.used_source_ids
        if source_id in evidence_by_id
    )

    for required_token in _get_required_code_tokens(query):
        if required_token not in code_tokens:
            feedback.append(f"Use the requested Python API token: {required_token}")

        if required_token not in used_evidence_text:
            feedback.append(f"Select an evidence source that explicitly contains: {required_token}")

    if not feedback:
        feedback.append(
            "Use only imports, constructors, and API symbols explicitly present in "
            "the selected evidence"
        )

    return feedback


def _sanitize_explanation(explanation: str) -> str:
    sanitized = MARKDOWN_LINK_PATTERN.sub(r"\1", explanation)
    sanitized = URL_PATTERN.sub("", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

    if not sanitized:
        raise InvalidCodeResultError("Code explanation contained no usable text")

    return sanitized
