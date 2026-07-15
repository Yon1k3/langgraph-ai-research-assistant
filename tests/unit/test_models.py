import pytest
from pydantic import ValidationError

from ai_research_assistant.models import (
    AgentResult,
    CodeCandidate,
    CodeSynthesis,
    GroundedClaim,
    ResearchSynthesis,
    RouteDecision,
    RouteName,
    SourceItem,
    SourceReference,
)

ALL_ROUTES: tuple[RouteName, ...] = (
    "research",
    "code",
    "comparison",
    "direct_answer",
    "clarification",
    "unsupported",
)


@pytest.mark.parametrize("route", ALL_ROUTES)
def test_route_decision_accepts_supported_routes(route: RouteName) -> None:
    clarification_question = (
        "Which technology do you want to research?" if route == "clarification" else None
    )

    decision = RouteDecision(
        route=route,
        confidence=0.8,
        reason="The request matches this route.",
        response_language="en",
        resolved_query="Resolved technical request.",
        clarification_question=clarification_question,
    )

    assert decision.route == route


@pytest.mark.parametrize("confidence", (-0.1, 1.1))
def test_route_decision_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            route="direct_answer",
            confidence=confidence,
            reason="Greeting detected.",
            response_language="en",
            resolved_query="Hello.",
        )


def test_clarification_route_requires_question() -> None:
    with pytest.raises(
        ValidationError,
        match="clarification_question is required",
    ):
        RouteDecision(
            route="clarification",
            confidence=0.6,
            reason="The request is ambiguous.",
            response_language="en",
            resolved_query="Help with a technical task.",
        )


def test_other_routes_reject_clarification_question() -> None:
    with pytest.raises(
        ValidationError,
        match="clarification_question is only allowed",
    ):
        RouteDecision(
            route="research",
            confidence=0.9,
            reason="Research is required.",
            response_language="en",
            resolved_query="Research LangGraph.",
            clarification_question="What exactly do you mean?",
        )


def test_route_decision_rejects_empty_resolved_query() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            route="research",
            confidence=0.9,
            reason="Research is required.",
            response_language="en",
            resolved_query="   ",
        )


def test_research_synthesis_accepts_grounded_answer() -> None:
    synthesis = ResearchSynthesis(
        claims=[
            GroundedClaim(
                statement="LangGraph supports stateful workflows.",
                source_id="src-0123456789ab",
                supporting_quote="LangGraph supports stateful agent workflows.",
            )
        ],
        is_sufficient=True,
    )

    assert synthesis.claims[0].source_id == "src-0123456789ab"


@pytest.mark.parametrize(
    ("claims", "is_sufficient"),
    [
        (
            [],
            True,
        ),
        (
            [
                {
                    "statement": "Grounded answer",
                    "source_id": "src-0123456789ab",
                    "supporting_quote": "A sufficiently long supporting quote.",
                }
            ],
            False,
        ),
        (
            [
                {
                    "statement": "Grounded answer",
                    "source_id": "invalid-source-id",
                    "supporting_quote": "A sufficiently long supporting quote.",
                }
            ],
            True,
        ),
    ],
)
def test_research_synthesis_rejects_invalid_evidence_contract(
    claims: list[dict[str, object]],
    is_sufficient: bool,
) -> None:
    with pytest.raises(ValidationError):
        ResearchSynthesis(
            claims=claims,
            is_sufficient=is_sufficient,
        )


def test_code_synthesis_accepts_complete_grounded_answer() -> None:
    synthesis = CodeSynthesis(
        explanation="Create and compile the graph.",
        code="graph = builder.compile()",
        code_language="python",
        used_source_ids=["src-0123456789ab"],
        is_sufficient=True,
    )

    assert synthesis.code_language == "python"


def test_code_candidate_requires_complete_non_fenced_output() -> None:
    candidate = CodeCandidate(
        explanation="Use the documented API.",
        code="print('ok')",
        code_language="python",
        used_source_ids=["src-0123456789ab"],
    )

    assert candidate.code == "print('ok')"

    with pytest.raises(ValidationError):
        CodeCandidate(
            explanation="Invalid fenced output.",
            code="```python\nprint('bad')\n```",
            code_language="python",
            used_source_ids=["src-0123456789ab"],
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "explanation": "Missing code.",
            "code_language": "python",
            "used_source_ids": ["src-0123456789ab"],
            "is_sufficient": True,
        },
        {
            "explanation": "Includes a Markdown fence.",
            "code": "```python\nprint('unsafe format')\n```",
            "code_language": "python",
            "used_source_ids": ["src-0123456789ab"],
            "is_sufficient": True,
        },
        {
            "code": "print('unexpected')",
            "is_sufficient": False,
        },
    ],
)
def test_code_synthesis_rejects_incomplete_or_inconsistent_output(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CodeSynthesis.model_validate(payload)


def test_agent_result_round_trips_through_checkpoint_safe_record() -> None:
    source = SourceReference.from_source(
        SourceItem(
            title="LangGraph overview",
            url="https://docs.langchain.com/oss/python/langgraph/overview",
            source_type="documentation",
        )
    )
    result = AgentResult(
        answer="LangGraph supports stateful workflows.",
        sources=[source],
        claims=[
            GroundedClaim(
                statement="LangGraph supports stateful workflows.",
                source_id=source.source_id,
                supporting_quote="LangGraph supports durable stateful agent workflows.",
            )
        ],
    )

    restored = AgentResult.from_record(result.to_record())

    assert restored == result
    assert isinstance(result.to_record()["sources"][0]["url"], str)


def test_agent_result_rejects_claim_with_unknown_source_id() -> None:
    with pytest.raises(
        ValidationError,
        match="every grounded claim must reference a result source",
    ):
        AgentResult(
            answer="Unresolvable claim.",
            claims=[
                GroundedClaim(
                    statement="Unresolvable claim.",
                    source_id="src-0123456789ab",
                    supporting_quote="This quote is long enough but has no source.",
                )
            ],
        )
