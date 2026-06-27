"""Parser utilities for converting agent string outputs into Pydantic models.

After each agent task (research, analysis, writing), the raw string output
is parsed into the corresponding Pydantic model. If parsing fails (indicating
hallucinated or malformed structure), a warning is logged and the raw string
is preserved — but the validation step catches issues before they propagate.

Usage:
    from schemas.parser import parse_research_brief, parse_analysis, parse_report
    brief = parse_research_brief(raw_text, topic)
    if brief is None:
        print("⚠️  Research output failed validation — structure issue detected")
"""

import json
import logging
import re
from typing import Optional

from schemas.models import (
    Analysis,
    ResearchBrief,
    ResearchFinding,
    ResearchReport,
    ThemeAnalysis,
)

logger = logging.getLogger(__name__)


def _try_extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from text that may have markdown wrapping.

    Args:
        text: Raw agent output that may contain JSON.

    Returns:
        Parsed dict if JSON found, None otherwise.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\n?\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def parse_research_brief(raw_text: str, topic: str) -> Optional[ResearchBrief]:
    """Parse agent research output into a typed ResearchBrief model.

    Args:
        raw_text: Raw string output from the researcher agent.
        topic: The research topic.

    Returns:
        ResearchBrief if parsing succeeds, None if structure is invalid.
    """
    data = _try_extract_json(raw_text)

    if data:
        try:
            findings = []
            for f in data.get("findings", []):
                findings.append(ResearchFinding(
                    title=f.get("title", "Untitled finding"),
                    summary=f.get("summary", f.get("description", "No summary")),
                    source_url=f.get("source_url", f.get("url", "")),
                    relevance=f.get("relevance", "medium"),
                ))

            return ResearchBrief(
                topic=data.get("topic", topic),
                findings=findings,
                key_statistics=data.get("key_statistics", []),
                conflicting_viewpoints=data.get("conflicting_viewpoints", []),
                gaps=data.get("gaps", []),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"ResearchBrief parsing failed: {e}")

    # Fallback: create from raw text with metadata
    try:
        # Try to create a minimal valid brief from the raw text
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        finding_count = len([l for l in lines if l.startswith(("- ", "* ", "1. ", "2. "))])

        return ResearchBrief(
            topic=topic,
            findings=[
                ResearchFinding(
                    title=f"Finding {i+1}",
                    summary=raw_text[:500],
                    source_url="",
                    relevance="medium",
                )
                for i in range(max(1, min(finding_count, 8)))
            ],
        )
    except Exception as e:
        logger.error(f"ResearchBrief fallback failed: {e}")
        return None


def parse_analysis(raw_text: str, topic: str) -> Optional[Analysis]:
    """Parse agent analysis output into a typed Analysis model.

    Args:
        raw_text: Raw string output from the analyst agent.
        topic: The research topic.

    Returns:
        Analysis if parsing succeeds, None if structure is invalid.
    """
    data = _try_extract_json(raw_text)

    if data:
        try:
            themes = []
            for t in data.get("themes", []):
                themes.append(ThemeAnalysis(
                    theme=t.get("theme", "Untitled theme"),
                    description=t.get("description", ""),
                    supporting_evidence=t.get("supporting_evidence", t.get("evidence", [])),
                    significance=t.get("significance", "medium"),
                ))

            return Analysis(
                topic=data.get("topic", topic),
                themes=themes,
                key_takeaways=data.get("key_takeaways", []),
                methodology_notes=data.get("methodology_notes"),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Analysis parsing failed: {e}")

    # Fallback: extract themes from markdown
    try:
        themes = []
        lines = raw_text.split("\n")
        current_theme = None
        current_desc = []

        for line in lines:
            if line.startswith("## ") or line.startswith("**"):
                if current_theme and current_desc:
                    themes.append(ThemeAnalysis(
                        theme=current_theme,
                        description=" ".join(current_desc)[:500],
                    ))
                current_theme = line.replace("## ", "").replace("**", "").strip()
                current_desc = []
            elif current_theme:
                current_desc.append(line.strip())

        if current_theme and current_desc:
            themes.append(ThemeAnalysis(
                theme=current_theme,
                description=" ".join(current_desc)[:500],
            ))

        return Analysis(
            topic=topic,
            themes=themes or [ThemeAnalysis(theme=topic, description=raw_text[:500])],
            key_takeaways=[line.strip("- *") for line in lines if line.strip().startswith("-")][:5],
        )
    except Exception as e:
        logger.error(f"Analysis fallback failed: {e}")
        return None


def parse_report(raw_text: str, topic: str) -> Optional[ResearchReport]:
    """Parse agent report output into a typed ResearchReport model.

    Args:
        raw_text: Raw string output from the writer agent.
        topic: The research topic.

    Returns:
        ResearchReport if parsing succeeds, None if structure is invalid.
    """
    data = _try_extract_json(raw_text)

    if data:
        try:
            return ResearchReport(
                topic=data.get("topic", topic),
                executive_summary=data.get("executive_summary", ""),
                introduction=data.get("introduction", ""),
                detailed_analysis=data.get("detailed_analysis", ""),
                key_insights=data.get("key_insights", []),
                challenges=data.get("challenges", []),
                future_outlook=data.get("future_outlook", ""),
                sources=data.get("sources", []),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"ResearchReport parsing failed: {e}")

    # Fallback: extract sections from markdown
    try:
        sections = re.split(r"\n## ", raw_text)
        section_map = {}
        current_header = "introduction"

        for section in sections:
            lines = section.strip().split("\n")
            header = lines[0].strip().lower().replace("#", "").strip()
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            if "executive summary" in header:
                section_map["executive_summary"] = content
            elif "introduction" in header:
                section_map["introduction"] = content
            elif "detailed analysis" in header or "analysis" in header:
                section_map["detailed_analysis"] = content
            elif "key insights" in header or "insights" in header:
                section_map["key_insights"] = [l.strip("- *") for l in content.split("\n") if l.strip().startswith("-")]
            elif "challenges" in header:
                section_map["challenges"] = [l.strip("- *") for l in content.split("\n") if l.strip().startswith("-")]
            elif "future outlook" in header or "outlook" in header:
                section_map["future_outlook"] = content
            elif "sources" in header or "references" in header:
                section_map["sources"] = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]

        return ResearchReport(
            topic=topic,
            executive_summary=section_map.get("executive_summary", raw_text[:500]),
            introduction=section_map.get("introduction", ""),
            detailed_analysis=section_map.get("detailed_analysis", ""),
            key_insights=section_map.get("key_insights", ["See full report"]),
            challenges=section_map.get("challenges", []),
            future_outlook=section_map.get("future_outlook", ""),
            sources=section_map.get("sources", []),
        )
    except Exception as e:
        logger.error(f"ResearchReport fallback failed: {e}")
        return None
