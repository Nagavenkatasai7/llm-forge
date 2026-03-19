"""Smart project detection and scaffolding for LLM Forge."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path

# Project type detection markers
PROJECT_MARKERS: dict[str, list[str]] = {
    "nodejs": ["package.json", "node_modules"],
    "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "ruby": ["Gemfile"],
    "dotnet": ["*.csproj", "*.sln"],
    "llmforge": [".llmforge", "configs"],
}

# .gitignore content for scaffolded projects
_GITIGNORE_CONTENT = """\
# LLM Forge
outputs/
*.safetensors
*.gguf
*.pt
*.bin
.llmforge/
data/
*.jsonl
*.parquet
*.csv
!examples/data/*.jsonl
!examples/data/*.txt
.env
*.key
*.pem
wandb/
"""


def _package_root() -> Path:
    """Return the root directory of the llm-forge *source tree*.

    This walks from this file up to the ``src/`` directory, then to the
    project root which contains ``configs/`` and ``examples/``.
    """
    # __file__ is  …/src/llm_forge/chat/project_setup.py
    # project root is  …/ (4 levels up from this file)
    return Path(__file__).resolve().parent.parent.parent.parent


def detect_project_type(directory: str | Path) -> dict:
    """Scan a directory and determine what kind of project it is.

    Returns a dict with:
    - is_empty: bool
    - is_llmforge: bool (already has .llmforge/)
    - detected_types: list of project types found
    - files_count: total files
    - has_git: bool
    - recommended_mode: "root" or "subdirectory"
    - existing_files: list of key files found
    """
    d = Path(directory).resolve()

    if not d.exists():
        return {
            "is_empty": True,
            "is_llmforge": False,
            "detected_types": [],
            "files_count": 0,
            "has_git": False,
            "recommended_mode": "root",
            "existing_files": [],
        }

    # Gather top-level entries (not hidden dot-dirs except .llmforge / .git)
    all_entries = list(d.iterdir())
    # Count only non-hidden real files recursively (cap at 5000 to avoid slowness)
    files_count = 0
    for item in d.rglob("*"):
        if item.is_file():
            files_count += 1
            if files_count >= 5000:
                break

    has_git = (d / ".git").is_dir()

    # Detect project types
    detected_types: list[str] = []
    existing_files: list[str] = []

    entry_names = [e.name for e in all_entries]

    for proj_type, markers in PROJECT_MARKERS.items():
        for marker in markers:
            if "*" in marker:
                # Glob-based marker (e.g. *.csproj)
                for entry_name in entry_names:
                    if fnmatch.fnmatch(entry_name, marker):
                        if proj_type not in detected_types:
                            detected_types.append(proj_type)
                        existing_files.append(entry_name)
            else:
                if (d / marker).exists():
                    if proj_type not in detected_types:
                        detected_types.append(proj_type)
                    existing_files.append(marker)

    is_llmforge = "llmforge" in detected_types
    # Consider directory empty if it has no non-hidden entries, or only .git
    visible_entries = [e for e in all_entries if not e.name.startswith(".")]
    is_empty = len(visible_entries) == 0

    # Determine recommended mode
    if is_empty or is_llmforge:
        recommended_mode = "root"
    else:
        recommended_mode = "subdirectory"

    return {
        "is_empty": is_empty,
        "is_llmforge": is_llmforge,
        "detected_types": detected_types,
        "files_count": files_count,
        "has_git": has_git,
        "recommended_mode": recommended_mode,
        "existing_files": sorted(set(existing_files)),
    }


def _starter_config_source(name: str) -> Path | None:
    """Locate a starter config in the package's configs directory."""
    configs_dir = _package_root() / "configs"
    src = configs_dir / name
    if src.is_file():
        return src
    return None


def _example_data_dir() -> Path | None:
    """Locate the examples/data directory shipped with the package."""
    examples = _package_root() / "examples" / "data"
    if examples.is_dir():
        return examples
    return None


def scaffold_project(
    directory: str | Path,
    mode: str = "auto",  # "root", "subdirectory", or "auto"
    purpose: str = "general",
    include_examples: bool = True,
    auto_approve: bool = False,
) -> dict:
    """Create the LLM Forge project structure.

    Returns a dict with:
    - created_files: list of files created
    - created_dirs: list of directories created
    - base_dir: where the forge directory is
    - mode: "root" or "subdirectory"
    - skipped_files: list of files that already existed (not overwritten)
    - status: "ok" or "error"
    """
    # Determine mode
    if mode == "auto":
        detection = detect_project_type(directory)
        if detection["is_empty"] or detection["is_llmforge"]:
            mode = "root"
        else:
            mode = "subdirectory"

    # Set base directory
    base = Path(directory).resolve()
    parent_dir = base  # The original directory the user pointed at
    if mode == "subdirectory":
        base = base / "llm-forge"

    created_files: list[str] = []
    created_dirs: list[str] = []
    skipped_files: list[str] = []

    # Helper: create a directory safely
    def _mkdir(p: Path) -> None:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(p))

    # Helper: write a file safely (NEVER overwrite)
    def _write(p: Path, content: str) -> None:
        if p.exists():
            skipped_files.append(str(p))
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        created_files.append(str(p))

    # Helper: copy a file safely (NEVER overwrite)
    def _copy(src: Path, dst: Path) -> None:
        if dst.exists():
            skipped_files.append(str(dst))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        created_files.append(str(dst))

    try:
        # 1. Create directories
        _mkdir(base)
        _mkdir(base / "configs")
        _mkdir(base / "data")
        _mkdir(base / "outputs")
        _mkdir(base / ".llmforge")

        if include_examples:
            _mkdir(base / "examples" / "data")

        # 2. Copy starter configs from package presets
        lora_src = _starter_config_source("example_lora.yaml")
        if lora_src is not None:
            _copy(lora_src, base / "configs" / "starter_lora.yaml")

        qlora_src = _starter_config_source("example_qlora.yaml")
        if qlora_src is not None:
            _copy(qlora_src, base / "configs" / "starter_qlora.yaml")

        # 3. Copy the default config as config.yaml at the root
        if lora_src is not None:
            _copy(lora_src, base / "config.yaml")

        # 4. Copy example data files
        if include_examples:
            examples_src = _example_data_dir()
            if examples_src is not None:
                for sample_file in sorted(examples_src.iterdir()):
                    if sample_file.is_file():
                        _copy(sample_file, base / "examples" / "data" / sample_file.name)

        # 5. Write .gitignore
        _write(base / ".gitignore", _GITIGNORE_CONTENT)

        # 6. If subdirectory mode, add llm-forge/ to parent .gitignore
        if mode == "subdirectory":
            parent_gitignore = parent_dir / ".gitignore"
            if parent_gitignore.exists():
                existing = parent_gitignore.read_text()
                if "llm-forge/" not in existing:
                    with open(parent_gitignore, "a") as f:
                        f.write("\n# LLM Forge training directory\nllm-forge/\n")
                    # We don't count this as a created file since it's an append
            # Don't create parent .gitignore if it doesn't exist

        return {
            "created_files": created_files,
            "created_dirs": created_dirs,
            "skipped_files": skipped_files,
            "base_dir": str(base),
            "mode": mode,
            "status": "ok",
        }

    except Exception as e:
        return {
            "created_files": created_files,
            "created_dirs": created_dirs,
            "skipped_files": skipped_files,
            "base_dir": str(base),
            "mode": mode,
            "status": "error",
            "error": str(e),
        }


def get_setup_plan(directory: str | Path, mode: str = "auto") -> dict:
    """Show what scaffold_project would do without doing it.

    Returns:
    - files_to_create: list of paths
    - directories_to_create: list of paths
    - mode: root or subdirectory
    - total_size_estimate: bytes
    """
    # Determine mode
    if mode == "auto":
        detection = detect_project_type(directory)
        if detection["is_empty"] or detection["is_llmforge"]:
            mode = "root"
        else:
            mode = "subdirectory"

    base = Path(directory).resolve()
    if mode == "subdirectory":
        base = base / "llm-forge"

    directories_to_create: list[str] = []
    files_to_create: list[str] = []
    total_size_estimate = 0

    # Directories
    for d in ["configs", "data", "outputs", ".llmforge", "examples/data"]:
        dp = base / d
        if not dp.exists():
            directories_to_create.append(str(dp))

    # Starter configs
    for src_name, dst_name in [
        ("example_lora.yaml", "configs/starter_lora.yaml"),
        ("example_qlora.yaml", "configs/starter_qlora.yaml"),
    ]:
        dst = base / dst_name
        if not dst.exists():
            src = _starter_config_source(src_name)
            if src is not None:
                files_to_create.append(str(dst))
                total_size_estimate += src.stat().st_size

    # config.yaml at root
    config_yaml = base / "config.yaml"
    if not config_yaml.exists():
        lora_src = _starter_config_source("example_lora.yaml")
        if lora_src is not None:
            files_to_create.append(str(config_yaml))
            total_size_estimate += lora_src.stat().st_size

    # Example data
    examples_src = _example_data_dir()
    if examples_src is not None:
        for sample_file in sorted(examples_src.iterdir()):
            if sample_file.is_file():
                dst = base / "examples" / "data" / sample_file.name
                if not dst.exists():
                    files_to_create.append(str(dst))
                    total_size_estimate += sample_file.stat().st_size

    # .gitignore
    gitignore = base / ".gitignore"
    if not gitignore.exists():
        files_to_create.append(str(gitignore))
        total_size_estimate += len(_GITIGNORE_CONTENT.encode())

    return {
        "files_to_create": files_to_create,
        "directories_to_create": directories_to_create,
        "mode": mode,
        "total_size_estimate": total_size_estimate,
    }
