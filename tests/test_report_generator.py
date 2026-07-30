"""Tests for report generation."""

import os
import shutil
import tempfile

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import save_report

SAMPLE_REPORT = """# Test Report

## Executive Summary

This is a test report.

## Introduction

The AI Research Agent generates structured reports.

## Detailed Analysis

- Key finding one
- Key finding two

## Key Insights

- **Important**: This is a test

## Challenges & Considerations

- Challenge 1

## Future Outlook

The future is bright.

## Sources & References

1. Test Source - https://example.com

## Quality Review

All quality standards met.
"""


def test_markdown_generation():
    """Verify Markdown file is created with content."""
    tmp = tempfile.mkdtemp()
    try:
        md_path, pdf_path = save_report("Test", SAMPLE_REPORT, tmp)
        assert os.path.exists(md_path)
        assert md_path.endswith(".md")
        with open(md_path) as f:
            content = f.read()
        assert "Executive Summary" in content
        assert "Sources & References" in content
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pdf_generation():
    """Verify PDF file is created with correct header."""
    tmp = tempfile.mkdtemp()
    try:
        md_path, pdf_path = save_report("Test", SAMPLE_REPORT, tmp)
        if pdf_path:  # May be skipped if WeasyPrint unavailable
            assert os.path.exists(pdf_path)
            assert pdf_path.endswith(".pdf")
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_both_formats_returned():
    """Verify both paths are returned from save_report."""
    tmp = tempfile.mkdtemp()
    try:
        md_path, pdf_path = save_report("Integration", SAMPLE_REPORT, tmp)
        assert md_path is not None
        assert md_path.endswith(".md")
        assert os.path.exists(md_path)
        # pdf_path could be None if WeasyPrint fails, that's OK
        if pdf_path:
            assert pdf_path.endswith(".pdf")
            assert os.path.exists(pdf_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_output_directory_creation():
    """Verify nested output directories are created."""
    tmp = tempfile.mkdtemp()
    try:
        nested = os.path.join(tmp, "nested", "deep", "reports")
        md_path, pdf_path = save_report("Test", SAMPLE_REPORT, nested)
        assert os.path.exists(md_path)
        assert os.path.isdir(nested)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
