"""CrewAI crew definition for the research agent pipeline.

Orchestrates three agents (Researcher → Writer → Editor) to research
any topic and generate a professional report.
"""

import os
from pathlib import Path
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@CrewBase
class ResearchCrew:
    """Research agent crew that researches topics and generates reports."""

    agents_config = str(Path(__file__).parent / "config" / "agents.yaml")
    tasks_config = str(Path(__file__).parent / "config" / "tasks.yaml")

    def __init__(self, verbose: bool = True):
        self._verbose = verbose
        self._tools = self._get_tools()

    def _get_tools(self) -> list:
        """Get the appropriate search tools based on available API keys."""
        tools = []

        # Always add website scraping (installed with crewai[tools])
        from crewai_tools import ScrapeWebsiteTool

        tools.append(ScrapeWebsiteTool())

        # Try to use SerperDevTool if SERPER_API_KEY is set (better search results)
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key:
            try:
                from crewai_tools import SerperDevTool

                tools.append(SerperDevTool())
                print("  🔍 Using SerperDevTool for web search (SERPER_API_KEY found)")
            except ImportError:
                print("  ⚠️  SerperDevTool not available")

        # Fall back to DuckDuckGo (no API key required, always available)
        if not serper_key:
            from research_agent.tools.custom_tool import (
                DuckDuckGoSearchTool,
                DuckDuckGoSearchResults,
            )

            tools.append(DuckDuckGoSearchTool())
            tools.append(DuckDuckGoSearchResults())
            print("  🔍 Using DuckDuckGo for web search (no API key needed)")

        return tools

    @agent
    def senior_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["senior_research_analyst"],
            tools=self._tools,
            verbose=self._verbose,
        )

    @agent
    def technical_content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_content_writer"],
            verbose=self._verbose,
        )

    @agent
    def quality_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["quality_editor"],
            verbose=self._verbose,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config["writing_task"])

    @task
    def editing_task(self) -> Task:
        return Task(config=self.tasks_config["editing_task"])

    @crew
    def crew(self) -> Crew:
        """Create the research crew with sequential process."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self._verbose,
        )

    def run(self, topic: str) -> str:
        """Run the research pipeline on a given topic.

        Args:
            topic: The research topic to investigate.

        Returns:
            The final report content as a string.
        """
        print(f"\n{'='*60}")
        print(f"  AI Research Agent")
        print(f"  Topic: {topic}")
        print(f"{'='*60}\n")

        inputs = {"topic": topic}
        result = self.crew().kickoff(inputs=inputs)

        return result.raw if hasattr(result, "raw") else str(result)
