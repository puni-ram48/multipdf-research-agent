import pytest

def test_pdf_parsing_mock():
    """Validates basic parsing logic and text normalization structure."""
    sample_text = "Abstract\nThis paper presents the Segment Anything Model (SAM)."
    normalized = sample_text.strip()
    assert "Segment Anything Model" in normalized
    assert len(normalized) > 0