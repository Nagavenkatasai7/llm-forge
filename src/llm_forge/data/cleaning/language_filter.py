"""Language identification and filtering using FastText lid.176.bin.

Detects the language of each document and filters datasets to retain only
documents in the specified target languages above a confidence threshold.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.language_filter")

# ---------------------------------------------------------------------------
# Optional dependency: FastText
# ---------------------------------------------------------------------------

try:
    import fasttext

    # Suppress FastText's own warnings about loading a .bin model
    fasttext.FastText.eprint = lambda *args, **kwargs: None
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_MODEL_CACHE_DIR = Path.home() / ".cache" / "llm_forge" / "models"

# HuggingFace Hub URL for the model (preferred, more reliable)
_HF_MODEL_URL = (
    "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin"
)

# Fallback: Original Facebook Research URL
_FB_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

_model_instance: object | None = None


def _download_model(dest_path: Path) -> Path:
    """Download the FastText language identification model.

    Tries the HuggingFace mirror first, then falls back to the Facebook URL.

    Parameters
    ----------
    dest_path:
        Where to save the downloaded model file.

    Returns
    -------
    Path
        The path to the downloaded model.
    """
    import urllib.error
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    urls = [_HF_MODEL_URL, _FB_MODEL_URL]

    for url in urls:
        logger.info("Downloading FastText LID model from %s ...", url)
        try:
            # Download to a temp file first, then rename (atomic on same FS)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_path.parent, suffix=".bin.tmp")
            os.close(tmp_fd)
            urllib.request.urlretrieve(url, tmp_path)
            os.rename(tmp_path, str(dest_path))
            logger.info("Model saved to %s", dest_path)
            return dest_path
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Failed to download from %s: %s", url, exc)
            # Clean up partial download
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            continue

    raise RuntimeError(
        "Could not download the FastText language identification model. "
        "Please download lid.176.bin manually and place it at: "
        f"{dest_path}\n"
        f"  wget {_FB_MODEL_URL} -O {dest_path}"
    )


def _get_model() -> object:
    """Load or download the FastText LID model (cached singleton).

    Returns
    -------
    fasttext.FastText._FastText
        The loaded FastText model.

    Raises
    ------
    ImportError
        If fasttext is not installed.
    RuntimeError
        If the model cannot be downloaded.
    """
    global _model_instance

    if not _FASTTEXT_AVAILABLE:
        raise ImportError(
            "FastText is required for language detection. "
            "Install it with: pip install fasttext-wheel\n"
            "  (or: pip install fasttext)"
        )

    if _model_instance is not None:
        return _model_instance

    model_path = _MODEL_CACHE_DIR / "lid.176.bin"

    if not model_path.exists():
        logger.info("FastText LID model not found at %s; downloading...", model_path)
        _download_model(model_path)

    logger.info("Loading FastText LID model from %s", model_path)
    _model_instance = fasttext.load_model(str(model_path))
    return _model_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Regex to clean text before feeding to FastText (newlines confuse it)
_NEWLINE_RE = re.compile(r"[\n\r]+")


def detect_language(text: str) -> tuple[str, float]:
    """Detect the language of a text string.

    Uses the first 200 characters of the text for speed. Newlines are
    replaced with spaces before detection since FastText expects single-line
    input.

    Parameters
    ----------
    text:
        The text to identify.

    Returns
    -------
    tuple[str, float]
        A ``(language_code, confidence)`` pair where *language_code* is an
        ISO-639-1 code (e.g. ``"en"``, ``"fr"``) and *confidence* is a
        float in [0, 1].
    """
    if not text or not text.strip():
        return ("und", 0.0)  # undetermined

    model = _get_model()

    # Use first 200 chars for speed; replace newlines for FastText
    snippet = _NEWLINE_RE.sub(" ", text[:200]).strip()

    if not snippet:
        return ("und", 0.0)

    predictions = model.predict(snippet, k=1)
    # predictions is ([labels], [probabilities])
    label = predictions[0][0]  # e.g. "__label__en"
    confidence = float(predictions[1][0])

    # Strip the __label__ prefix
    lang_code = label.replace("__label__", "")

    return (lang_code, confidence)


def filter_by_language(
    dataset: Dataset,
    languages: list[str],
    threshold: float = 0.8,
    text_field: str = "text",
) -> Dataset:
    """Filter a dataset to retain only documents in specified languages.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` instance.
    languages:
        List of ISO-639-1 language codes to keep (e.g. ``["en", "de"]``).
    threshold:
        Minimum confidence score to accept the language detection result.
        Documents below this threshold are discarded.
    text_field:
        Name of the column containing the text.

    Returns
    -------
    Dataset
        A filtered dataset containing only documents in the target languages
        with confidence above the threshold.

    Raises
    ------
    ValueError
        If *text_field* does not exist in the dataset.
    ImportError
        If fasttext is not installed.
    """
    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Normalize language codes to lowercase
    languages_lower = {lang.lower() for lang in languages}

    initial_count = len(dataset)
    logger.info(
        "Filtering %d records for languages %s (threshold=%.2f)...",
        initial_count,
        sorted(languages_lower),
        threshold,
    )

    # Counters for stats
    stats = {"kept": 0, "rejected_language": 0, "rejected_confidence": 0}

    def _keep(example: dict) -> bool:
        text = example[text_field]
        lang, conf = detect_language(text)

        if lang.lower() not in languages_lower:
            stats["rejected_language"] += 1
            return False

        if conf < threshold:
            stats["rejected_confidence"] += 1
            return False

        stats["kept"] += 1
        return True

    dataset = dataset.filter(
        _keep,
        desc="Filtering by language",
        num_proc=1,  # FastText model is not fork-safe
    )

    final_count = len(dataset)
    logger.info(
        "Language filtering complete: %d -> %d records "
        "(rejected %d wrong language, %d low confidence).",
        initial_count,
        final_count,
        stats["rejected_language"],
        stats["rejected_confidence"],
    )

    return dataset
