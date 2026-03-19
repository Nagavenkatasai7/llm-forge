"""Preset resolution and config merging for the llm-forge pipeline.

Loads built-in or user-defined preset YAML files and deep-merges
user overrides on top to produce a final configuration dictionary.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from llm_forge.utils.logging import get_logger

logger = get_logger("pipeline.preset_resolver")

# ---------------------------------------------------------------------------
# Built-in presets directory
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"


# ============================================================================
# PresetResolver
# ============================================================================


class PresetResolver:
    """Resolve preset names to full configuration dictionaries.

    Presets are YAML files that provide sensible default configurations
    for common training scenarios (e.g. LoRA, QLoRA, full fine-tuning).
    Users can reference a preset and selectively override individual fields.

    Supports a chain of preset inheritance via an ``_inherit`` key in the
    preset YAML, allowing presets to build on top of other presets.

    Parameters
    ----------
    preset_dirs : list[str | Path], optional
        Additional directories to search for preset files, beyond the
        built-in presets directory.
    """

    def __init__(
        self,
        preset_dirs: list[str | Path] | None = None,
    ) -> None:
        self._search_dirs: list[Path] = [_PRESETS_DIR]
        if preset_dirs:
            for d in preset_dirs:
                p = Path(d).resolve()
                if p.exists() and p.is_dir():
                    self._search_dirs.append(p)
                else:
                    logger.warning("Preset directory does not exist: %s", p)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve(self, preset_name: str) -> dict[str, Any]:
        """Load a preset by name and return the fully expanded config dict.

        Parameters
        ----------
        preset_name : str
            Preset name (without ``.yaml`` extension), or a path to a YAML file.

        Returns
        -------
        dict
            The fully resolved configuration dictionary, with all inherited
            presets merged in order.

        Raises
        ------
        FileNotFoundError
            If the preset cannot be found in any search directory.
        """
        preset_path = self._find_preset(preset_name)
        if preset_path is None:
            available = self.list_presets()
            available_str = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"Preset '{preset_name}' not found in any search directory. "
                f"Available presets: {available_str}"
            )

        logger.info("Resolving preset: %s (%s)", preset_name, preset_path)
        return self._load_with_inheritance(preset_path, visited=set())

    def merge_with_overrides(
        self,
        preset_config: dict[str, Any],
        user_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep-merge user overrides on top of the preset configuration.

        Parameters
        ----------
        preset_config : dict
            Base configuration from a resolved preset.
        user_overrides : dict
            User-provided overrides to apply on top.

        Returns
        -------
        dict
            The merged configuration dictionary.  The original inputs are
            not modified.
        """
        result = copy.deepcopy(preset_config)
        _deep_merge(result, user_overrides)
        return result

    def list_presets(self) -> list[str]:
        """Return names of all available presets across search directories.

        Returns
        -------
        list[str]
            Sorted list of preset names (without extensions).
        """
        names: set = set()
        for search_dir in self._search_dirs:
            if search_dir.exists():
                for p in search_dir.glob("*.yaml"):
                    if p.is_file() and not p.name.startswith("_"):
                        names.add(p.stem)
                for p in search_dir.glob("*.yml"):
                    if p.is_file() and not p.name.startswith("_"):
                        names.add(p.stem)
        return sorted(names)

    def resolve_and_override(
        self,
        preset_name: str,
        user_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience method: resolve a preset then apply user overrides.

        Parameters
        ----------
        preset_name : str
            Preset name or path.
        user_overrides : dict, optional
            Additional user overrides.

        Returns
        -------
        dict
            The final merged configuration dictionary.
        """
        preset_cfg = self.resolve(preset_name)
        if user_overrides:
            return self.merge_with_overrides(preset_cfg, user_overrides)
        return preset_cfg

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _find_preset(self, name: str) -> Path | None:
        """Locate a preset file by name or path."""
        # Check if it is an explicit file path
        explicit = Path(name)
        if explicit.exists() and explicit.is_file():
            return explicit.resolve()

        # Strip extension for searching
        stem = name.removesuffix(".yaml").removesuffix(".yml")

        for search_dir in self._search_dirs:
            for ext in (".yaml", ".yml"):
                candidate = search_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

        return None

    def _load_with_inheritance(
        self,
        preset_path: Path,
        visited: set,
    ) -> dict[str, Any]:
        """Load a preset YAML, recursively resolving ``_inherit`` chains.

        Parameters
        ----------
        preset_path : Path
            Path to the preset YAML file.
        visited : set
            Set of already-visited paths to detect circular inheritance.

        Returns
        -------
        dict
            The fully resolved configuration.
        """
        resolved = preset_path.resolve()
        if resolved in visited:
            raise ValueError(f"Circular preset inheritance detected involving: {resolved}")
        visited.add(resolved)

        with open(preset_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Preset file must be a YAML mapping, got {type(raw).__name__}: {preset_path}"
            )

        # Handle inheritance
        inherit_name = raw.pop("_inherit", None)
        if inherit_name is not None:
            parent_path = self._find_preset(str(inherit_name))
            if parent_path is None:
                raise FileNotFoundError(
                    f"Inherited preset '{inherit_name}' not found (referenced by {preset_path})"
                )
            parent_cfg = self._load_with_inheritance(parent_path, visited)
            _deep_merge(parent_cfg, raw)
            return parent_cfg

        return raw


# ---------------------------------------------------------------------------
# Utility: deep merge
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in place.

    - Dicts are merged recursively.
    - Lists in *override* replace lists in *base* entirely.
    - Scalar values in *override* replace scalars in *base*.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
