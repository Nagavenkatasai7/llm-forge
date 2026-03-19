"""Universal data loader supporting multiple file formats and sources."""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
    ".parquet",
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
}


class DataLoader:
    """Universal data connector that loads data from files, directories, URLs, or HuggingFace."""

    def __init__(
        self,
        path: str,
        streaming: bool = False,
        num_workers: int = 4,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.path = path
        self.streaming = streaming
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed

    def load(self) -> Dataset:
        """Load data from the configured path, auto-detecting the source type."""
        path = self.path

        if path.startswith(("http://", "https://")):
            return self._load_from_url(path)

        local_path = Path(path)
        if local_path.exists():
            if local_path.is_dir():
                return self._load_from_directory(local_path)
            return self._load_from_file(local_path)

        return self._load_from_huggingface(path)

    def _load_from_file(self, path: Path) -> Dataset:
        """Load a single file based on its extension."""
        ext = path.suffix.lower()
        logger.info(f"Loading file: {path} (format: {ext})")

        if ext == ".jsonl":
            return self._load_jsonl(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext == ".csv":
            return load_dataset("csv", data_files=str(path), split="train")
        elif ext == ".tsv":
            return load_dataset("csv", data_files=str(path), split="train", delimiter="\t")
        elif ext == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        elif ext == ".txt" or ext == ".md":
            return self._load_text(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        elif ext in (".html", ".htm"):
            return self._load_html(path)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def _load_from_directory(self, path: Path) -> Dataset:
        """Recursively load all supported files from a directory."""
        logger.info(f"Loading directory: {path}")
        all_records: list[dict[str, Any]] = []

        files = sorted(
            f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            raise FileNotFoundError(
                f"No supported files found in {path}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        logger.info(f"Found {len(files)} files to load")

        for file_path in files:
            try:
                ds = self._load_from_file(file_path)
                for record in ds:
                    record["_source_file"] = str(file_path)
                    all_records.append(record)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")

        dataset = Dataset.from_list(all_records)
        return self._apply_limits(dataset)

    def _load_with_timeout(self, dataset_name: str, **kwargs: Any) -> Dataset | DatasetDict:
        """Load dataset with timeout protection.

        Uses SIGALRM on Unix to abort loads that hang longer than 30 minutes.
        On Windows, falls back to a plain ``load_dataset`` call (no timeout).
        """
        timeout = 1800  # 30 minutes max for dataset loading

        def _timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(
                f"Dataset loading timed out after {timeout}s. "
                f"Try: streaming=True in your config, or download the dataset manually."
            )

        if platform.system() != "Windows":
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                result = load_dataset(dataset_name, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        else:
            return load_dataset(dataset_name, **kwargs)

    def _load_from_huggingface(self, dataset_name: str) -> Dataset:
        """Load a dataset from HuggingFace Hub."""
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")

        try:
            ds = self._load_with_timeout(
                dataset_name,
                streaming=self.streaming,
                num_proc=self.num_workers if not self.streaming else None,
            )
        except TimeoutError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load '{dataset_name}'. "
                f"Check if the path is a valid file, directory, or HuggingFace dataset ID. "
                f"Error: {e}"
            ) from e

        if isinstance(ds, DatasetDict):
            if "train" in ds:
                dataset = ds["train"]
            else:
                first_split = next(iter(ds.keys()))
                logger.warning(f"No 'train' split found, using '{first_split}'")
                dataset = ds[first_split]
        else:
            dataset = ds

        return self._apply_limits(dataset)

    def _load_from_url(self, url: str) -> Dataset:
        """Download and load data from a URL."""
        import tempfile
        import urllib.request

        logger.info(f"Downloading from URL: {url}")

        suffix = Path(url.split("?")[0]).suffix or ".jsonl"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            urllib.request.urlretrieve(url, str(tmp_path))
            return self._load_from_file(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink()

    def _load_jsonl(self, path: Path) -> Dataset:
        """Load JSONL file."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        return self._apply_limits(Dataset.from_list(records))

    def _load_json(self, path: Path) -> Dataset:
        """Load JSON file (array or single object)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._apply_limits(Dataset.from_list(data))
        elif isinstance(data, dict):
            if any(isinstance(v, list) for v in data.values()):
                return self._apply_limits(Dataset.from_dict(data))
            return self._apply_limits(Dataset.from_list([data]))
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")

    def _load_text(self, path: Path) -> Dataset:
        """Load plain text file as completion-format data."""
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        records = [{"text": p} for p in paragraphs]
        return self._apply_limits(Dataset.from_list(records))

    def _load_pdf(self, path: Path) -> Dataset:
        """Load PDF file using pymupdf."""
        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF loading. Install with: pip install llm-forge[cleaning]"
            )

        doc = pymupdf.open(str(path))
        records = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                records.append(
                    {
                        "text": text,
                        "_page_number": page_num + 1,
                        "_source_file": str(path),
                    }
                )
        doc.close()

        if not records:
            logger.warning(f"No text extracted from PDF: {path}")
            return Dataset.from_list([{"text": ""}])

        return self._apply_limits(Dataset.from_list(records))

    def _load_docx(self, path: Path) -> Dataset:
        """Load DOCX file using python-docx."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX loading. "
                "Install with: pip install llm-forge[cleaning]"
            )

        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        records = [{"text": p, "_source_file": str(path)} for p in paragraphs]

        if not records:
            logger.warning(f"No text extracted from DOCX: {path}")
            return Dataset.from_list([{"text": ""}])

        return self._apply_limits(Dataset.from_list(records))

    def _load_html(self, path: Path) -> Dataset:
        """Load HTML file using trafilatura for main content extraction."""
        try:
            import trafilatura
        except ImportError:
            raise ImportError(
                "trafilatura is required for HTML loading. "
                "Install with: pip install llm-forge[cleaning]"
            )

        html_content = path.read_text(encoding="utf-8")
        text = trafilatura.extract(html_content)

        if not text:
            logger.warning(f"No main content extracted from HTML: {path}")
            return Dataset.from_list([{"text": ""}])

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        records = [{"text": p, "_source_file": str(path)} for p in paragraphs]
        return self._apply_limits(Dataset.from_list(records))

    def _apply_limits(self, dataset: Dataset) -> Dataset:
        """Apply max_samples limit and shuffle."""
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.shuffle(seed=self.seed).select(range(self.max_samples))
        return dataset

    def load_streaming(self) -> Iterator[dict[str, Any]]:
        """Load data as a streaming iterator for large datasets."""
        path = self.path

        if path.startswith(("http://", "https://")):
            ds = self.load()
            yield from ds
            return

        local_path = Path(path)
        if local_path.exists() and local_path.suffix == ".jsonl":
            with open(local_path, encoding="utf-8") as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        break
            return

        try:
            ds = load_dataset(path, streaming=True)
            if isinstance(ds, DatasetDict):
                ds = ds["train"]
            count = 0
            for item in ds:
                yield item
                count += 1
                if self.max_samples and count >= self.max_samples:
                    break
        except Exception:
            ds = self.load()
            yield from ds
