"""
llm/llm_schema/deliverables.py
================================
Pydantic schemas for the four phase deliverables.

Each deliverable is produced by a dedicated LLM call inside its
generator tool using .with_structured_output(). The schema drives
both the LLM output format and the deterministic markdown renderer.

Phase order and prerequisites:
    1. DiscoveryDeliverable       — no prerequisites
    2. ClusteringDeliverable      — requires DiscoveryDeliverable
    3. GapAnalysisDeliverable     — requires ClusteringDeliverable
    4. WritingOutlineDeliverable  — requires GapAnalysisDeliverable

Design rules:
    - All fields have descriptions — these are read by the LLM
      when generating structured output
    - No Optional fields without a clear default — ambiguity in
      schema = ambiguity in LLM output
    - Nested models for complex structures — flat is not always better
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ==================================================================
# PHASE 1 — Discovery Deliverable
# ==================================================================

class CorpusOverview(BaseModel):
    """Summary of the ingested research corpus."""
    total_sources: int = Field(
        description="Total number of sources in the corpus"
    )
    source_breakdown: dict[str, int] = Field(
        description="Count per source type: {url: 2, pdf: 5, image: 1}"
    )
    languages: list[str] = Field(
        description="Languages present in the corpus as ISO 639-1 codes"
    )
    key_authors: list[str] = Field(
        default_factory=list,
        description="Prominent authors identified across sources"
    )
    date_range: Optional[str] = Field(
        default=None,
        description="Approximate date range of sources, e.g. '2019-2024'"
    )
    dominant_themes: list[str] = Field(
        description="Top 3-5 recurring themes identified in the corpus"
    )


class Constraints(BaseModel):
    """Practical constraints on the research output."""
    target_venue: Optional[str] = Field(
        default=None,
        description="Target journal, conference, or publication venue"
    )
    page_limit: Optional[str] = Field(
        default=None,
        description="Page or word count limit if specified"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Submission or completion deadline"
    )
    citation_style: Optional[str] = Field(
        default=None,
        description="Required citation style: APA, MLA, Chicago, IEEE"
    )


class DiscoveryDeliverable(BaseModel):
    """
    Phase 1 output: establishes the research foundation.

    Captures the researcher's field, intent, corpus snapshot,
    practical constraints, and target output type.
    This is the prerequisite for all subsequent phases.
    """
    field_summary: str = Field(
        description=(
            "2-3 sentence summary of the academic field and sub-field "
            "this research belongs to"
        )
    )
    research_intent: str = Field(
        description=(
            "Clear statement of the research question or objective. "
            "Should be specific enough to guide gap analysis."
        )
    )
    corpus_overview: CorpusOverview = Field(
        description="Structured summary of the ingested research material"
    )
    constraints: Constraints = Field(
        description="Practical constraints: venue, length, deadline, citation style"
    )
    target_output: str = Field(
        description=(
            "Type of research output: e.g. 'Conference paper (8 pages)', "
            "'PhD thesis chapter', 'Systematic literature review'"
        )
    )


# ==================================================================
# PHASE 2 — Clustering Deliverable
# ==================================================================

class Cluster(BaseModel):
    """A thematic cluster of related sources in the corpus."""
    cluster_id: str = Field(
        description="Short identifier, e.g. 'cluster_attention_mechanisms'"
    )
    label: str = Field(
        description="Human-readable cluster name, e.g. 'Attention Mechanisms'"
    )
    description: str = Field(
        description="2-3 sentence description of what unites these sources"
    )
    source_ids: list[str] = Field(
        description="List of source_ids belonging to this cluster"
    )
    key_concepts: list[str] = Field(
        description="Top 3-5 concepts or keywords defining this cluster"
    )
    representative_source: Optional[str] = Field(
        default=None,
        description="The single most representative source in this cluster"
    )


class Relationship(BaseModel):
    """A relationship between two clusters."""
    cluster_a: str = Field(description="cluster_id of the first cluster")
    cluster_b: str = Field(description="cluster_id of the second cluster")
    relationship_type: Literal[
        "builds_on", "contradicts", "complements",
        "overlaps", "applies_to"
    ] = Field(description="Type of relationship between the clusters")
    description: str = Field(
        description="One sentence explaining how these clusters relate"
    )


class ClusteringDeliverable(BaseModel):
    """
    Phase 2 output: a taxonomy of the research corpus.

    Groups sources into thematic clusters, maps relationships
    between clusters, and identifies orphan sources that don't
    fit any cluster (often the most interesting gaps).

    Prerequisite: DiscoveryDeliverable must be accepted.
    """
    clusters: list[Cluster] = Field(
        description="List of thematic clusters. Aim for 3-7 clusters."
    )
    cross_cluster_relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships between clusters that reveal the field's structure"
    )
    orphan_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Source origins that don't fit any cluster. "
            "These often point to emerging or underexplored areas."
        )
    )
    taxonomy_summary: str = Field(
        description=(
            "2-3 sentence narrative of what the clustering reveals "
            "about the field's structure"
        )
    )


# ==================================================================
# PHASE 3 — Gap Analysis Deliverable
# ==================================================================

class Gap(BaseModel):
    """A single identified research gap."""
    gap_id: str = Field(
        description="Short identifier, e.g. 'gap_multilingual_reasoning'"
    )
    title: str = Field(
        description="Concise gap title, e.g. 'Multilingual Reasoning in LLMs'"
    )
    description: str = Field(
        description=(
            "2-3 sentences describing what is unknown, unstudied, "
            "or methodologically weak in this area"
        )
    )
    evidence: list[str] = Field(
        description=(
            "List of source citations or cluster references that "
            "support the existence of this gap"
        )
    )
    gap_type: Literal[
        "empirical", "theoretical", "methodological",
        "application", "replication"
    ] = Field(
        description="Classification of the gap type"
    )
    feasibility: Literal["high", "medium", "low"] = Field(
        description=(
            "Estimated feasibility of addressing this gap given "
            "the researcher's corpus and constraints"
        )
    )
    novelty: Literal["high", "medium", "low"] = Field(
        description="Estimated novelty of addressing this gap"
    )


class GapAnalysisDeliverable(BaseModel):
    """
    Phase 3 output: identifies and prioritizes research gaps.

    Surfaces unanswered questions and methodological weaknesses
    visible from the cluster structure. The researcher picks one
    gap to pursue — this becomes the thesis of the writing outline.

    Prerequisite: ClusteringDeliverable must be accepted.
    """
    gaps: list[Gap] = Field(
        description="All identified gaps. Aim for 3-6 well-defined gaps."
    )
    chosen_gap: Optional[Gap] = Field(
        default=None,
        description=(
            "The gap the researcher has chosen to pursue. "
            "Set after the researcher selects from the gaps list."
        )
    )
    rationale: Optional[str] = Field(
        default=None,
        description=(
            "Researcher's reasoning for choosing this gap over others. "
            "Captured during the gap selection conversation."
        )
    )
    related_work_summary: str = Field(
        description=(
            "Paragraph summarizing existing work most relevant to the "
            "chosen gap (or all gaps if none chosen yet)"
        )
    )


# ==================================================================
# PHASE 4 — Writing Outline Deliverable
# ==================================================================

class CitationEntry(BaseModel):
    """A single citation to be used in the paper."""
    citation_key: str = Field(
        description="Short key, e.g. 'vaswani2017attention'"
    )
    origin: str = Field(
        description="Source URL or filename this citation comes from"
    )
    suggested_context: str = Field(
        description="One sentence on where/why to cite this source"
    )


class OutlineSection(BaseModel):
    """A single section in the paper outline."""
    section_number: str = Field(
        description="e.g. '1', '2.1', '3.2.1'"
    )
    title: str = Field(
        description="Section title"
    )
    paragraph_intents: list[str] = Field(
        description=(
            "One sentence per paragraph describing what that paragraph "
            "will argue or demonstrate"
        )
    )
    citations: list[CitationEntry] = Field(
        default_factory=list,
        description="Sources to cite in this section"
    )
    estimated_length: Optional[str] = Field(
        default=None,
        description="Approximate length: e.g. '300-400 words', '1 page'"
    )


class WritingOutlineDeliverable(BaseModel):
    """
    Phase 4 output: the complete writing roadmap.

    This IS the final deliverable of the research planning journey.
    It provides a structured, citation-mapped outline the researcher
    can follow to write their paper.

    Prerequisite: GapAnalysisDeliverable must be accepted.
    """
    title_options: list[str] = Field(
        description=(
            "3 candidate paper titles ranging from descriptive to "
            "provocative. Researcher picks or adapts one."
        )
    )
    abstract_draft: str = Field(
        description=(
            "150-250 word abstract draft covering: motivation, gap, "
            "approach, contribution, and implications"
        )
    )
    sections: list[OutlineSection] = Field(
        description=(
            "Complete section-by-section outline with paragraph intents "
            "and citations. Typically 5-8 top-level sections."
        )
    )
    citation_style: Literal["APA", "MLA", "Chicago", "IEEE"] = Field(
        description="Citation style to use throughout the paper"
    )
    estimated_total_length: Optional[str] = Field(
        default=None,
        description="Total estimated paper length, e.g. '8 pages', '12,000 words'"
    )