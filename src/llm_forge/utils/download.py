"""HuggingFace Hub download utilities for llm-forge.

Provides model and dataset download functions with progress bars, caching,
and Hub connectivity checks.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from llm_forge.utils.logging import get_logger

logger = get_logger("utils.download")

# Default cache directory follows HuggingFace convention
_DEFAULT_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------


def _make_progress(console: Console | None = None) -> Progress:
    """Create a Rich progress bar configured for file downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console or Console(),
        transient=False,
    )


# ---------------------------------------------------------------------------
# Hub connectivity
# ---------------------------------------------------------------------------


def check_model_exists(model_name: str) -> bool:
    """Verify that a model repository exists on the HuggingFace Hub.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"meta-llama/Llama-2-7b-hf"``).

    Returns
    -------
    bool
        *True* if the repo exists and is accessible.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.model_info(model_name)
        return True
    except Exception:
        return False


def check_dataset_exists(dataset_name: str) -> bool:
    """Verify that a dataset repository exists on the HuggingFace Hub.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. ``"tatsu-lab/alpaca"``).

    Returns
    -------
    bool
        *True* if the repo exists and is accessible.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.dataset_info(dataset_name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def download_model(
    model_name: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    console: Console | None = None,
) -> Path:
    """Download a model from the HuggingFace Hub.

    Uses ``huggingface_hub.snapshot_download`` to fetch all model files
    into a local cache directory.  Files already present are skipped.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    revision:
        Git revision (branch, tag, or commit hash).  Defaults to ``"main"``.
    cache_dir:
        Local directory for cached downloads.
    token:
        HuggingFace API token.  Falls back to the ``HF_TOKEN`` environment
        variable or the locally-stored token.
    console:
        Optional Rich console for progress output.

    Returns
    -------
    Path
        Local path to the downloaded model snapshot.

    Raises
    ------
    ValueError
        If the model does not exist on the Hub.
    RuntimeError
        If the download fails.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    _console = console or Console()
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    token = token or os.environ.get("HF_TOKEN")

    _console.print(f"[bold]Downloading model:[/bold] {model_name}")
    if revision:
        _console.print(f"  Revision: {revision}")
    _console.print(f"  Cache dir: {cache}")

    try:
        with _make_progress(_console) as progress:
            task = progress.add_task(f"Downloading {model_name}", total=None)

            local_path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=str(cache),
                token=token,
                ignore_patterns=[
                    "*.md",
                    "*.txt",
                    ".gitattributes",
                    "original/",
                ],
            )

            progress.update(task, completed=True)

        _console.print(f"[green]Download complete:[/green] {local_path}")
        logger.info("Model %s downloaded to %s", model_name, local_path)
        return Path(local_path)

    except RepositoryNotFoundError:
        raise ValueError(
            f"Model '{model_name}' not found on HuggingFace Hub. "
            "Check the model name and your access permissions."
        )
    except GatedRepoError:
        raise ValueError(
            f"Model '{model_name}' is a gated repository. "
            "Accept the license at https://huggingface.co/{model_name} "
            "and provide your HF token via HF_TOKEN env var or `huggingface-cli login`."
        )
    except EntryNotFoundError as exc:
        raise RuntimeError(f"Required file not found in model repo: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to download model '{model_name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------


def download_dataset(
    dataset_name: str,
    split: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    console: Console | None = None,
) -> Path:
    """Download a dataset from the HuggingFace Hub.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. ``"tatsu-lab/alpaca"``).
    split:
        Specific split to download (e.g. ``"train"``).  If *None*,
        all splits are downloaded.
    cache_dir:
        Local directory for cached downloads.
    token:
        HuggingFace API token.
    console:
        Optional Rich console for progress output.

    Returns
    -------
    Path
        Local path to the downloaded dataset cache directory.

    Raises
    ------
    ValueError
        If the dataset does not exist on the Hub.
    RuntimeError
        If the download fails.
    """
    _console = console or Console()
    cache = Path(cache_dir) if cache_dir else None
    token = token or os.environ.get("HF_TOKEN")

    _console.print(f"[bold]Downloading dataset:[/bold] {dataset_name}")
    if split:
        _console.print(f"  Split: {split}")

    try:
        from datasets import load_dataset

        with _make_progress(_console) as progress:
            task = progress.add_task(f"Downloading {dataset_name}", total=None)

            ds = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(cache) if cache else None,
                token=token,
            )

            progress.update(task, completed=True)

        # Determine the cache path
        if cache:
            dataset_cache_path = cache
        else:
            dataset_cache_path = Path(
                os.environ.get(
                    "HF_DATASETS_CACHE",
                    Path.home() / ".cache" / "huggingface" / "datasets",
                )
            )

        if split and hasattr(ds, "num_rows"):
            _console.print(
                f"[green]Download complete:[/green] {ds.num_rows:,} examples ({split} split)"
            )
        elif hasattr(ds, "keys"):
            splits_info = ", ".join(f"{s}: {ds[s].num_rows:,}" for s in ds)
            _console.print(f"[green]Download complete:[/green] {splits_info}")
        else:
            _console.print("[green]Download complete.[/green]")

        logger.info("Dataset %s downloaded", dataset_name)
        return dataset_cache_path

    except FileNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found on HuggingFace Hub.")
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset '{dataset_name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_model_size_on_hub(model_name: str) -> float | None:
    """Query the HuggingFace Hub for approximate model size in GB.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.

    Returns
    -------
    float or None
        Approximate total size in GB, or *None* if unavailable.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(model_name, files_metadata=True)
        total_bytes = sum(
            (s.size or 0)
            for s in (info.siblings or [])
            if s.rfilename.endswith((".safetensors", ".bin", ".pt"))
        )
        if total_bytes > 0:
            return round(total_bytes / (1024**3), 2)
        return None
    except Exception:
        return None


def list_model_files(model_name: str) -> list[str]:
    """List files in a HuggingFace model repository.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.

    Returns
    -------
    list[str]
        List of relative file paths in the repository.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(model_name)
        return [s.rfilename for s in (info.siblings or [])]
    except Exception:
        return []
