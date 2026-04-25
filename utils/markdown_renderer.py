"""
utils/markdown_renderer.py
===========================
Deterministic Pydantic → markdown rendering for all four deliverables.

No LLM involvement. Pure functions. Same input always produces same output.
Called inside deliverable generator tools after structured output is produced.

Used by:
    llm/workflow/research_planner/tool/generate_*_deliverable.py
    api/routers/research_planner.py (GET /deliverables endpoint)
"""

from llm.llm_schema.deliverables import (
    ClusteringDeliverable,
    DiscoveryDeliverable,
    GapAnalysisDeliverable,
    WritingOutlineDeliverable,
)


def render_discovery_markdown(d: DiscoveryDeliverable) -> str:
    co = d.corpus_overview
    c = d.constraints

    lines = [
        "## 📚 Discovery",
        "",
        "### Field",
        d.field_summary,
        "",
        "### Research Intent",
        d.research_intent,
        "",
        "### Target Output",
        d.target_output,
        "",
        "### Corpus Overview",
        f"- **Total sources:** {co.total_sources}",
        f"- **Languages:** {', '.join(co.languages) if co.languages else 'N/A'}",
        f"- **Date range:** {co.date_range or 'N/A'}",
    ]

    if co.source_breakdown:
        lines.append("- **Source breakdown:**")
        for src_type, count in co.source_breakdown.items():
            lines.append(f"  - {src_type}: {count}")

    if co.dominant_themes:
        lines.append("")
        lines.append("### Dominant Themes")
        for theme in co.dominant_themes:
            lines.append(f"- {theme}")

    if co.key_authors:
        lines.append("")
        lines.append("### Key Authors")
        for author in co.key_authors:
            lines.append(f"- {author}")

    lines.append("")
    lines.append("### Constraints")
    lines.append(f"- **Venue:** {c.target_venue or 'Not specified'}")
    lines.append(f"- **Length:** {c.page_limit or 'Not specified'}")
    lines.append(f"- **Deadline:** {c.deadline or 'Not specified'}")
    lines.append(f"- **Citation style:** {c.citation_style or 'Not specified'}")

    return "\n".join(lines)


def render_clustering_markdown(d: ClusteringDeliverable) -> str:
    lines = [
        "## 🗂️ Clustering",
        "",
        "### Taxonomy Summary",
        d.taxonomy_summary,
        "",
        "### Clusters",
    ]

    for cluster in d.clusters:
        lines.extend([
            f"",
            f"#### {cluster.label}",
            f"{cluster.description}",
            f"",
            f"- **Key concepts:** {', '.join(cluster.key_concepts)}",
            f"- **Sources:** {len(cluster.source_ids)} sources",
        ])
        if cluster.representative_source:
            lines.append(
                f"- **Representative:** {cluster.representative_source}"
            )

    if d.cross_cluster_relationships:
        lines.extend(["", "### Cross-Cluster Relationships"])
        for rel in d.cross_cluster_relationships:
            lines.append(
                f"- **{rel.cluster_a}** *{rel.relationship_type}* "
                f"**{rel.cluster_b}**: {rel.description}"
            )

    if d.orphan_sources:
        lines.extend(["", "### Orphan Sources"])
        lines.append("*These sources don't fit any cluster — potential gaps:*")
        for src in d.orphan_sources:
            lines.append(f"- {src}")

    return "\n".join(lines)


def render_gap_analysis_markdown(d: GapAnalysisDeliverable) -> str:
    lines = [
        "## 🔍 Gap Analysis",
        "",
        "### Related Work Summary",
        d.related_work_summary,
        "",
        "### Identified Gaps",
    ]

    for gap in d.gaps:
        lines.extend([
            "",
            f"#### {gap.title}",
            f"{gap.description}",
            "",
            f"- **Type:** {gap.gap_type}",
            f"- **Feasibility:** {gap.feasibility}",
            f"- **Novelty:** {gap.novelty}",
        ])
        if gap.evidence:
            lines.append(f"- **Evidence:** {', '.join(gap.evidence)}")

    if d.chosen_gap:
        lines.extend([
            "",
            "### ✅ Chosen Gap",
            f"**{d.chosen_gap.title}**",
            "",
            d.chosen_gap.description,
        ])
        if d.rationale:
            lines.extend(["", "**Rationale:**", d.rationale])

    return "\n".join(lines)


def render_writing_outline_markdown(d: WritingOutlineDeliverable) -> str:
    lines = [
        "## ✍️ Writing Outline",
        "",
        "### Title Options",
    ]

    for i, title in enumerate(d.title_options, 1):
        lines.append(f"{i}. {title}")

    lines.extend([
        "",
        "### Abstract Draft",
        d.abstract_draft,
        "",
        f"**Citation style:** {d.citation_style}",
    ])

    if d.estimated_total_length:
        lines.append(f"**Estimated length:** {d.estimated_total_length}")

    lines.extend(["", "### Outline"])

    for section in d.sections:
        lines.extend([
            "",
            f"#### {section.section_number}. {section.title}",
        ])
        if section.estimated_length:
            lines.append(f"*{section.estimated_length}*")
        lines.append("")
        for intent in section.paragraph_intents:
            lines.append(f"- {intent}")
        if section.citations:
            lines.append("")
            lines.append("**Citations:**")
            for cite in section.citations:
                lines.append(
                    f"- `{cite.citation_key}` — {cite.suggested_context}"
                )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Convenience dispatcher
# ------------------------------------------------------------------

def render_deliverable_markdown(phase: str, deliverable) -> str:
    """
    Render any deliverable to markdown by phase name.

    Args:
        phase: phase name string ("discovery", "clustering", etc.)
        deliverable: the Pydantic deliverable object

    Returns:
        rendered markdown string
    """
    renderers = {
        "discovery": render_discovery_markdown,
        "clustering": render_clustering_markdown,
        "gap_analysis": render_gap_analysis_markdown,
        "writing_outline": render_writing_outline_markdown,
    }
    renderer = renderers.get(phase)
    if renderer is None:
        raise ValueError(f"Unknown phase: {phase}")
    return renderer(deliverable)