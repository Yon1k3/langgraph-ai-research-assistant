from typing import Annotated, Literal, Self, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

RouteName: TypeAlias = Literal[
    "research",
    "code",
    "comparison",
    "direct_answer",
    "clarification",
    "unsupported",
]
SourceType: TypeAlias = Literal[
    "documentation",
    "github",
    "web",
    "release_notes",
]
SourceId: TypeAlias = Annotated[str, Field(pattern=r"^src-[0-9a-f]{12}$")]


class SourceRecord(TypedDict):
    """JSON-serializable source metadata stored in graph state."""

    title: str
    url: str
    source_type: SourceType


class SourceItem(BaseModel):
    """Normalized source metadata shown with a final response."""

    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    source_type: SourceType

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    def to_record(self) -> SourceRecord:
        """Convert validated source metadata to checkpoint-safe primitives."""

        return {
            "title": self.title,
            "url": str(self.url),
            "source_type": self.source_type,
        }


class SearchResultItem(BaseModel):
    """Normalized search result used by the research workflow."""

    source: SourceItem
    content: str = Field(min_length=1, max_length=5000)
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceItem(BaseModel):
    """Search evidence available to the grounded synthesis step."""

    source_id: SourceId
    source: SourceItem
    content: str = Field(min_length=1, max_length=5000)
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GroundedClaim(BaseModel):
    """One answer claim linked to an exact excerpt from one evidence record."""

    statement: str = Field(min_length=1, max_length=1_000)
    source_id: SourceId
    supporting_quote: str = Field(min_length=20, max_length=1_000)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearchSynthesis(BaseModel):
    """Structured claims produced only from collected evidence."""

    claims: list[GroundedClaim] = Field(default_factory=list, max_length=6)
    is_sufficient: bool

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> Self:
        """Require grounded claims only when the evidence is sufficient."""

        if self.is_sufficient and not self.claims:
            raise ValueError("claims are required when evidence is sufficient")

        if not self.is_sufficient and self.claims:
            raise ValueError("claims must be empty when evidence is insufficient")

        return self


class ResearchResult(BaseModel):
    """Final answer and real sources collected by the research agent."""

    answer: str = Field(min_length=1)
    sources: list[SourceItem]

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RouteDecision(BaseModel):
    """Structured routing decision produced by the router."""

    route: RouteName
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    response_language: str = Field(min_length=2, max_length=20)
    clarification_question: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_clarification_question(self) -> Self:
        """Validate that clarification data matches the selected route."""

        if self.route == "clarification" and not self.clarification_question:
            raise ValueError("clarification_question is required for the clarification route")

        if self.route != "clarification" and self.clarification_question is not None:
            raise ValueError("clarification_question is only allowed for the clarification route")

        return self
