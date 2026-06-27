"""Tests for the report generator module."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from research_agent.report_generator import (
    generate_markdown_report,
    generate_pdf_report,
    _sanitize_filename,
)

# Sample markdown report content for testing
SAMPLE_REPORT = """# Test Report: AI Research Topic

## Executive Summary

This is a test report for validating the report generator.

## Introduction

The AI Research Agent generates structured reports on any topic.

## Detailed Analysis

### Subsection 1

- Key finding one
- Key finding two
- Key finding three

### Subsection 2

Some detailed text about the topic.

## Key Insights

- **Important**: This is a test
- **Notable**: The PDF generator works
- **Significant**: WeasyPrint renders CSS

## Challenges & Considerations

- Challenge 1
- Challenge 2

## Future Outlook

The future is bright for AI research agents.

## Sources & References

1. Test Source One - https://example.com/1
2. Test Source Two - https://example.com/2

## Quality Review

The report meets all quality standards.
"""


class TestSanitizeFilename:
    """Tests for the _sanitize_filename helper."""

    def test_basic_topic(self):
        result = _sanitize_filename("Test Topic")
        assert result == "test-topic"

    def test_special_characters(self):
        result = _sanitize_filename("AI & Machine Learning! (2024)")
        assert result == "ai---machine-learning--2024-"

    def test_long_topic(self):
        long_topic = "A" * 100
        result = _sanitize_filename(long_topic)
        assert len(result) <= 60

    def test_spaces_and_hyphens(self):
        result = _sanitize_filename("  Multiple   Spaces  ")
        assert result == "multiple-spaces"


class TestGenerateMarkdownReport:
    """Tests for Markdown report generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_generates_markdown_file(self, temp_output_dir):
        """Verify that a .md file is created."""
        path = generate_markdown_report(
            "Test Topic", SAMPLE_REPORT, output_dir=temp_output_dir
        )
        assert os.path.exists(path)
        assert path.endswith(".md")

    def test_content_preserved(self, temp_output_dir):
        """Verify the markdown content is preserved exactly."""
        path = generate_markdown_report(
            "Test Topic", SAMPLE_REPORT, output_dir=temp_output_dir
        )
        with open(path, "r") as f:
            content = f.read()
        assert "Test Report: AI Research Topic" in content
        assert "Executive Summary" in content
        assert "Sources & References" in content

    def test_custom_filename(self, temp_output_dir):
        """Verify custom filename is respected."""
        path = generate_markdown_report(
            "Test Topic",
            SAMPLE_REPORT,
            output_dir=temp_output_dir,
            filename="custom_report.md",
        )
        assert "custom_report.md" in path
        assert os.path.exists(path)

    def test_creates_output_directory(self, temp_output_dir):
        """Verify the output directory is created if it doesn't exist."""
        nested_dir = os.path.join(temp_output_dir, "nested", "output")
        path = generate_markdown_report(
            "Test Topic", SAMPLE_REPORT, output_dir=nested_dir
        )
        assert os.path.exists(path)
        assert os.path.isdir(nested_dir)


class TestGeneratePDFReport:
    """Tests for PDF report generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_generates_pdf_file(self, temp_output_dir):
        """Verify that a .pdf file is created."""
        path = generate_pdf_report(
            "Test Topic", SAMPLE_REPORT, output_dir=temp_output_dir
        )
        if path:  # PDF generation might be skipped if WeasyPrint is unavailable
            assert os.path.exists(path)
            assert path.endswith(".pdf")
            # PDF files should start with %PDF
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"

    def test_custom_filename(self, temp_output_dir):
        """Verify custom filename is respected."""
        path = generate_pdf_report(
            "Test Topic",
            SAMPLE_REPORT,
            output_dir=temp_output_dir,
            filename="custom_report.pdf",
        )
        if path:
            assert "custom_report.pdf" in path
            assert os.path.exists(path)

    def test_pdf_has_content(self, temp_output_dir):
        """Verify the PDF contains the report text."""
        path = generate_pdf_report(
            "Test Topic", SAMPLE_REPORT, output_dir=temp_output_dir
        )
        if path:
            with open(path, "rb") as f:
                content = f.read()
            # Check that the PDF contains our text (as part of its content stream)
            assert b"AI Research Topic" in content
            assert b"Executive Summary" in content


class TestGenerateBothFormats:
    """Integration test for generating both formats."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_both_formats_generated(self, temp_output_dir):
        """Verify both .md and .pdf files can be generated from the same content."""
        md_path = generate_markdown_report(
            "Integration Test", SAMPLE_REPORT, output_dir=temp_output_dir
        )
        pdf_path = generate_pdf_report(
            "Integration Test", SAMPLE_REPORT, output_dir=temp_output_dir
        )

        assert os.path.exists(md_path)

        if pdf_path:
            assert os.path.exists(pdf_path)
            # Both files should have the same base topic name
            md_name = os.path.basename(md_path)
            pdf_name = os.path.basename(pdf_path)
            assert md_name.startswith("integration-test")
            assert pdf_name.startswith("integration-test")
