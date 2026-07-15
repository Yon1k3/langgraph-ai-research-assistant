import hashlib
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
ErrorCategory: TypeAlias = Literal[
    "model_unavailable",
    "invalid_model_output",
    "search_not_configured",
    "search_authentication",
    "search_rate_limit",
    "search_unavailable",
    "invalid_search_response",
    "no_trustworthy_sources",
]


class SourceRecord(TypedDict):
    """JSON-serializable source metadata stored in graph state."""

    title: str
    url: str
    source_type: SourceType


class SourceReferenceRecord(SourceRecord):
    """JSON-serializable source metadata with its stable evidence ID."""

    source_id: str


class GroundedClaimRecord(TypedDict):
    """JSON-serializable claim attribution stored in graph state."""

    statement: str
    source_id: str
    supporting_quote: str


class AgentResultRecord(TypedDict):
    """Canonical JSON-safe result shared by all specialist routes."""

    answer: str
    sources: list[SourceReferenceRecord]
    claims: list[GroundedClaimRecord]


class ErrorInfoRecord(TypedDict):
    """Safe runtime error metadata exposed to the application graph."""

    category: ErrorCategory
    message: str


def build_source_id(url: str) -> str:
    """Build a stable non-secret ID from a normalized source URL."""

    source_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"src-{source_digest}"


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


class SourceReference(BaseModel):
    """A verified source linked to the stable ID used by grounded claims."""

    source_id: SourceId
    source: SourceItem

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_source(cls, source: SourceItem) -> Self:
        """Create a reference using the canonical URL-derived source ID."""

        return cls(
            source_id=build_source_id(str(source.url)),
            source=source,
        )

    @property
    def title(self) -> str:
        return self.source.title

    @property
    def url(self) -> HttpUrl:
        return self.source.url

    @property
    def source_type(self) -> SourceType:
        return self.source.source_type

    def to_record(self) -> SourceReferenceRecord:
        """Convert the reference to checkpoint-safe primitives."""

        return {
            "source_id": self.source_id,
            **self.source.to_record(),
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

    def to_record(self) -> GroundedClaimRecord:
        """Convert the claim to checkpoint-safe primitives."""

        return {
            "statement": self.statement,
            "source_id": self.source_id,
            "supporting_quote": self.supporting_quote,
        }


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


class AgentResult(BaseModel):
    """Canonical final result produced by any application route."""

    answer: str = Field(min_length=1)
    sources: list[SourceReference] = Field(default_factory=list)
    claims: list[GroundedClaim] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_attribution_contract(self) -> Self:
        """Require unique sources and resolvable source IDs for every claim."""

        source_ids = [source.source_id for source in self.sources]
        source_urls = [str(source.url) for source in self.sources]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("agent result sources must have unique source IDs")

        if len(source_urls) != len(set(source_urls)):
            raise ValueError("agent result sources must have unique URLs")

        unknown_source_ids = [
            claim.source_id for claim in self.claims if claim.source_id not in source_ids
        ]

        if unknown_source_ids:
            raise ValueError("every grounded claim must reference a result source")

        return self

    def to_record(self) -> AgentResultRecord:
        """Convert the result to checkpoint-safe primitives."""

        return {
            "answer": self.answer,
            "sources": [source.to_record() for source in self.sources],
            "claims": [claim.to_record() for claim in self.claims],
        }

    @classmethod
    def from_record(cls, record: AgentResultRecord) -> Self:
        """Restore and validate a result loaded from graph state."""

        sources = [
            SourceReference(
                source_id=source["source_id"],
                source=SourceItem.model_validate(
                    {
                        "title": source["title"],
                        "url": source["url"],
                        "source_type": source["source_type"],
                    }
                ),
            )
            for source in record["sources"]
        ]
        claims = [GroundedClaim.model_validate(claim) for claim in record["claims"]]

        return cls(
            answer=record["answer"],
            sources=sources,
            claims=claims,
        )


class ResearchResult(AgentResult):
    """Canonical result produced by the Research Agent."""


class ErrorInfo(BaseModel):
    """Safe error details that may be shown to the user and debug UI."""

    category: ErrorCategory
    message: str = Field(min_length=1, max_length=500)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    def to_record(self) -> ErrorInfoRecord:
        """Convert safe error metadata to checkpoint-safe primitives."""

        return {
            "category": self.category,
            "message": self.message,
        }


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
