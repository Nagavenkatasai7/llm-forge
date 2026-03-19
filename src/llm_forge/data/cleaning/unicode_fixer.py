"""Unicode normalization and text encoding repair for llm-forge.

Uses the ftfy library to fix mojibake, curly quotes, HTML entities, and
other common encoding problems found in web-scraped corpora.  Also strips
invisible characters (zero-width spaces, control characters, BOM) and
applies Unicode NFC normalization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.unicode_fixer")

# ---------------------------------------------------------------------------
# Optional dependency: ftfy
# ---------------------------------------------------------------------------

try:
    import ftfy

    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Invisible / control character pattern
# ---------------------------------------------------------------------------
# Remove zero-width spaces, BOM, soft hyphens, and C0/C1 control characters
# BUT preserve normal whitespace: space (0x20), tab (0x09), LF (0x0A), CR (0x0D)
_INVISIBLE_RE = re.compile(
    r"["
    r"\u00ad"  # soft hyphen
    r"\u200b-\u200f"  # zero-width space, ZWNJ, ZWJ, LRM, RLM
    r"\u202a-\u202e"  # LRE, RLE, PDF, LRO, RLO
    r"\u2060-\u2069"  # word joiner, invisible times/separator/plus, etc.
    r"\ufeff"  # BOM / zero-width no-break space
    r"\ufff9-\ufffb"  # interlinear annotations
    r"\x00-\x08"  # C0 control chars (NUL through BS, excluding TAB)
    r"\x0b"  # vertical tab
    r"\x0c"  # form feed
    r"\x0e-\x1f"  # C0 control chars (SO through US, excluding CR/LF)
    r"\x7f"  # DEL
    r"\x80-\x9f"  # C1 control chars
    r"]"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fix_unicode(text: str) -> str:
    """Fix encoding issues, remove invisible characters, and NFC-normalize.

    Parameters
    ----------
    text:
        Raw text that may contain mojibake, HTML entities, curly quotes,
        invisible characters, or non-NFC codepoints.

    Returns
    -------
    str
        Cleaned, NFC-normalized text.

    Notes
    -----
    If the ``ftfy`` library is not installed, this function still removes
    invisible characters and applies NFC normalization, but cannot repair
    mojibake or HTML entities.
    """
    if not text:
        return text

    # Step 1: ftfy text repair (mojibake, entities, curly quotes)
    if _FTFY_AVAILABLE:
        text = ftfy.fix_text(
            text,
            unescape_html=True,
            uncurl_quotes=True,
            fix_character_width=True,
            fix_line_breaks=True,
            fix_surrogates=True,
            remove_terminal_escapes=True,
            fix_encoding=True,
            normalization="NFC",
        )
    else:
        logger.debug(
            "ftfy is not installed; skipping mojibake repair. Install it with: pip install ftfy"
        )

    # Step 2: Remove invisible / control characters (preserving \t \n \r and space)
    text = _INVISIBLE_RE.sub("", text)

    # Step 3: Unicode NFC normalization (belt-and-suspenders; ftfy already does
    # this when available, but we ensure it for the non-ftfy path too)
    text = unicodedata.normalize("NFC", text)

    return text


def process_dataset(
    dataset: Dataset,
    text_field: str = "text",
) -> Dataset:
    """Apply :func:`fix_unicode` to every record in a HuggingFace Dataset.

    Parameters
    ----------
    dataset:
        A ``datasets.Dataset`` instance.
    text_field:
        Name of the column containing the text to clean.

    Returns
    -------
    Dataset
        A new Dataset with the text field cleaned in-place.

    Raises
    ------
    ValueError
        If *text_field* does not exist in the dataset.
    """
    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    initial_count = len(dataset)
    logger.info(
        "Applying unicode fixing to %d records (field='%s')...",
        initial_count,
        text_field,
    )

    if not _FTFY_AVAILABLE:
        logger.warning(
            "ftfy is not installed. Only invisible-char removal and NFC "
            "normalization will be applied. Install ftfy for full repair: "
            "pip install ftfy"
        )

    def _apply(example: dict) -> dict:
        example[text_field] = fix_unicode(example[text_field])
        return example

    dataset = dataset.map(
        _apply,
        desc="Fixing unicode",
        num_proc=1,  # ftfy is CPU-bound but not GIL-free; keep sequential
    )

    logger.info(
        "Unicode fixing complete. Processed %d records.",
        initial_count,
    )
    return dataset
