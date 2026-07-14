from typing import Literal, Self, TypeAlias

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


class SourceItem(BaseModel):
    """Normalized source metadata shown with a final response."""

    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    source_type: SourceType

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchResultItem(BaseModel):
    """Normalized search result used by the research workflow."""

    source: SourceItem
    content: str = Field(min_length=1, max_length=5000)
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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
