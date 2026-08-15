"""Tests for document reading with vision fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.documents import (
    JPEG_THRESHOLD_BYTES,
    MAX_IMAGE_EDGE_PX,
    TEXT_THRESHOLD_CHARS,
    VISION_ATTEMPTS,
    DocumentError,
    DocumentResult,
    PageResult,
    describe_image,
    read_document,
    read_document_json,
    render_page_image,
)

pymupdf = pytest.importorskip("pymupdf")


def vision_client(replies):
    """A client whose vision calls return `replies` in order.

    Also answers models.list, because describe_image resolves a vision model
    from the same client when none is passed explicitly.
    """
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="kimi-k2.6")]
    )
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])
        for r in replies
    ]
    return client


def make_pdf(tmp_path, text="", pages=1):
    """Build a real PDF; empty text yields an image-only-style blank page."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


class TestTextExtraction:
    def test_text_pdf_needs_no_vision(self, tmp_path) -> None:
        body = "Machine learning report. " * 20
        result = read_document(str(make_pdf(tmp_path, body)), client=vision_client([]))
        assert result.pages[0].source == "text"
        assert "Machine learning" in result.text


class TestVisionFallback:
    def test_page_without_text_is_transcribed(self, tmp_path) -> None:
        """The real failure: a scanned page extracts to zero characters.

        Text-only extraction reports success and returns nothing, so the model
        believes it read the file.
        """
        client = vision_client(["CERTIFICATE of COMPLETION\nJane Doe"])
        result = read_document(str(make_pdf(tmp_path)), client=client)

        assert result.pages[0].source == "vision"
        assert "CERTIFICATE" in result.text

    def test_empty_vision_reply_is_retried(self, tmp_path) -> None:
        """Vision intermittently returns nothing on dense scans.

        The same page returned 0 chars then 914 chars on retry, so a single
        empty reply must not be treated as a blank page.
        """
        client = vision_client(["", "", "Recovered on the third attempt"])
        result = read_document(str(make_pdf(tmp_path)), client=client)

        assert result.pages[0].source == "vision"
        assert "Recovered" in result.text
        assert client.chat.completions.create.call_count == 3

    def test_persistently_empty_page_is_reported_not_faked(self, tmp_path) -> None:
        """A genuinely blank page must say so rather than look transcribed."""
        client = vision_client([""] * VISION_ATTEMPTS)
        result = read_document(str(make_pdf(tmp_path)), client=client)

        page = result.pages[0]
        assert page.source == "empty"
        assert page.text == ""
        assert "no text" in page.note.lower()

    def test_no_legible_text_sentinel_is_honoured(self, tmp_path) -> None:
        client = vision_client(["[no legible text]"])
        result = read_document(str(make_pdf(tmp_path)), client=client)
        assert result.pages[0].source == "empty"

    def test_vision_can_be_disabled(self, tmp_path) -> None:
        client = vision_client(["should not be called"])
        result = read_document(str(make_pdf(tmp_path)), use_vision=False, client=client)
        assert result.pages[0].source == "empty"
        assert client.chat.completions.create.call_count == 0

    def test_vision_page_budget_is_enforced(self, tmp_path) -> None:
        """A 200-page scan must not silently become 200 vision calls."""
        client = vision_client(["text"] * 10)
        result = read_document(
            str(make_pdf(tmp_path, pages=5)), max_vision_pages=2, client=client
        )
        transcribed = [p for p in result.pages if p.source == "vision"]
        assert len(transcribed) == 2
        assert any("max_vision_pages" in p.note for p in result.pages)


class TestImageSizing:
    def test_oversized_page_is_downscaled(self, tmp_path) -> None:
        """A 150-DPI scan reached 3686x5219 -> a 23 MB PNG the API rejects
        with a bare 400 that reads like a model failure."""
        doc = pymupdf.open()
        # A very large page, so 150 DPI blows past the ceiling.
        doc.new_page(width=1700, height=2400)
        path = tmp_path / "big.pdf"
        doc.save(str(path))
        doc.close()

        opened = pymupdf.open(str(path))
        b64, media = render_page_image(opened[0])
        opened.close()

        raw_bytes = len(b64) * 3 // 4
        assert raw_bytes < JPEG_THRESHOLD_BYTES * 2
        assert media in ("image/png", "image/jpeg")

    def test_small_page_stays_png(self, tmp_path) -> None:
        doc = pymupdf.open()
        doc.new_page(width=200, height=200)
        path = tmp_path / "small.pdf"
        doc.save(str(path))
        doc.close()

        opened = pymupdf.open(str(path))
        _, media = render_page_image(opened[0])
        opened.close()
        assert media == "image/png"

    def test_ceiling_is_a_sane_value(self) -> None:
        assert 512 <= MAX_IMAGE_EDGE_PX <= 4096


class TestUnsupportedAndMissing:
    def test_missing_file(self) -> None:
        with pytest.raises(DocumentError, match="not found"):
            read_document("/nonexistent/file.pdf")

    def test_unsupported_type_lists_what_works(self, tmp_path) -> None:
        path = tmp_path / "thing.xyz"
        path.write_text("data")
        with pytest.raises(DocumentError, match="Supported"):
            read_document(str(path))

    def test_plain_text_passes_through(self, tmp_path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("# Heading\n\nSome content.")
        result = read_document(str(path))
        assert "Heading" in result.text

    def test_json_wrapper_reports_errors_not_raises(self) -> None:
        payload = json.loads(read_document_json(path="/nonexistent/x.pdf"))
        assert payload["status"] == "error"


class TestNoVisionModel:
    def test_missing_vision_model_explains_itself(self) -> None:
        with patch(
            "llm_forge.chat.ollama_provider.default_vision_model", return_value=None
        ), patch("llm_forge.chat.ollama_provider._client", return_value=MagicMock()):
            with pytest.raises(DocumentError, match="vision-capable"):
                describe_image("Zm9v", "prompt")


class TestResultShape:
    def test_reports_which_pages_used_vision(self) -> None:
        result = DocumentResult(
            path="x.pdf",
            page_count=2,
            pages=[PageResult(1, "a", "text"), PageResult(2, "b", "vision")],
        )
        assert result.as_dict()["pages_needing_vision"] == [2]

    def test_threshold_is_documented_value(self) -> None:
        assert TEXT_THRESHOLD_CHARS == 100
