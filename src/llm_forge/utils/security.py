"""Security utilities for llm-forge.

Provides safetensors validation, pickle safety checks, environment
configuration loading, and sensitive-value masking for logs.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("utils.security")


# ---------------------------------------------------------------------------
# Sensitive key patterns
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key)"),
    re.compile(r"(?i)(secret)"),
    re.compile(r"(?i)(token)"),
    re.compile(r"(?i)(password)"),
    re.compile(r"(?i)(credential)"),
    re.compile(r"(?i)(auth)"),
    re.compile(r"(?i)(private[_-]?key)"),
    re.compile(r"(?i)(access[_-]?key)"),
    re.compile(r"(?i)(wandb[_-]?key)"),
    re.compile(r"(?i)(hf[_-]?token)"),
    re.compile(r"(?i)(openai)"),
    re.compile(r"(?i)(anthropic)"),
    re.compile(r"(?i)(aws[_-]?secret)"),
]

_MASK = "********"


# ---------------------------------------------------------------------------
# Safetensors validation
# ---------------------------------------------------------------------------


def validate_safetensors(path: str | Path) -> dict[str, Any]:
    """Validate the integrity of a safetensors file.

    Checks:
    1. File exists and is readable.
    2. Header is valid JSON with expected structure.
    3. File size matches declared tensor sizes.
    4. Computes SHA-256 hash for integrity verification.

    Parameters
    ----------
    path:
        Path to the ``.safetensors`` file.

    Returns
    -------
    dict[str, Any]
        Validation result with keys:
        - ``valid`` (bool): Whether the file passed all checks.
        - ``num_tensors`` (int): Number of tensors in the file.
        - ``file_size_mb`` (float): File size in megabytes.
        - ``sha256`` (str): Hex digest of the file's SHA-256 hash.
        - ``errors`` (list[str]): Any validation errors found.
        - ``tensor_names`` (list[str]): Names of tensors found.
        - ``metadata`` (dict): Metadata section from the header.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    errors: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "num_tensors": 0,
        "file_size_mb": 0.0,
        "sha256": "",
        "errors": errors,
        "tensor_names": [],
        "metadata": {},
    }

    # -- Existence check --
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    if not path.is_file():
        errors.append(f"Path is not a regular file: {path}")
        return result

    file_size = path.stat().st_size
    result["file_size_mb"] = round(file_size / (1024 * 1024), 2)

    if file_size < 8:
        errors.append("File is too small to be a valid safetensors file (< 8 bytes).")
        return result

    try:
        with open(path, "rb") as f:
            # Safetensors format: first 8 bytes = uint64 header size (little-endian)
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            # Sanity: header should not exceed file size
            if header_size > file_size - 8:
                errors.append(
                    f"Declared header size ({header_size:,} bytes) exceeds "
                    f"remaining file size ({file_size - 8:,} bytes)."
                )
                return result

            # Reasonable upper bound for header size (256 MB)
            if header_size > 256 * 1024 * 1024:
                errors.append(f"Header size suspiciously large: {header_size:,} bytes.")
                return result

            # Parse header JSON
            import json

            header_bytes = f.read(header_size)
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"Failed to parse header JSON: {exc}")
                return result

            if not isinstance(header, dict):
                errors.append("Header is not a JSON object.")
                return result

            # Extract metadata
            metadata = header.pop("__metadata__", {})
            result["metadata"] = metadata

            # Validate tensor entries
            tensor_names: list[str] = []
            expected_data_end = 0

            for name, info in header.items():
                if not isinstance(info, dict):
                    errors.append(f"Tensor '{name}' entry is not a dict.")
                    continue
                if "dtype" not in info:
                    errors.append(f"Tensor '{name}' missing 'dtype' field.")
                if "shape" not in info:
                    errors.append(f"Tensor '{name}' missing 'shape' field.")
                if "data_offsets" not in info:
                    errors.append(f"Tensor '{name}' missing 'data_offsets' field.")
                    continue

                offsets = info["data_offsets"]
                if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                    errors.append(f"Tensor '{name}' has invalid data_offsets: {offsets}")
                    continue

                start, end = offsets
                if end < start:
                    errors.append(f"Tensor '{name}' data_offsets end ({end}) < start ({start}).")
                expected_data_end = max(expected_data_end, end)
                tensor_names.append(name)

            result["tensor_names"] = sorted(tensor_names)
            result["num_tensors"] = len(tensor_names)

            # Verify data section size
            actual_data_size = file_size - 8 - header_size
            if expected_data_end > actual_data_size:
                errors.append(
                    f"Tensor data requires {expected_data_end:,} bytes but "
                    f"only {actual_data_size:,} bytes available."
                )

    except OSError as exc:
        errors.append(f"I/O error reading file: {exc}")
        return result

    # -- SHA-256 hash --
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        result["sha256"] = sha.hexdigest()
    except OSError as exc:
        errors.append(f"Failed to compute SHA-256: {exc}")

    result["valid"] = len(errors) == 0
    if result["valid"]:
        logger.info(
            "Safetensors file validated: %s (%d tensors, %.1f MB)",
            path.name,
            result["num_tensors"],
            result["file_size_mb"],
        )
    else:
        logger.warning("Safetensors validation failed for %s: %s", path, errors)

    return result


# ---------------------------------------------------------------------------
# Pickle safety
# ---------------------------------------------------------------------------


_DANGEROUS_PICKLE_OPCODES = {
    b"\x80",  # PROTO
    b"c",  # GLOBAL - can import arbitrary modules
    b"\x93",  # STACK_GLOBAL
    b"R",  # REDUCE - calls arbitrary callables
    b"b",  # BUILD - can call __setstate__
    b"i",  # INST - instantiate class
    b"o",  # OBJ
}

# Known safe module prefixes in PyTorch pickles
_SAFE_MODULES = {
    "torch",
    "torch._utils",
    "torch.nn",
    "collections",
    "_codecs",
    "numpy",
    "numpy.core",
}


def check_pickle_safety(path: str | Path) -> dict[str, Any]:
    """Analyze a pickle (.bin, .pt, .pkl) file for potential security risks.

    This performs a heuristic scan of the pickle bytecode to detect
    potentially dangerous operations such as arbitrary code execution.

    Parameters
    ----------
    path:
        Path to the pickle file.

    Returns
    -------
    dict[str, Any]
        Safety report:
        - ``safe`` (bool): Whether the file appears safe.
        - ``risk_level`` (str): ``"low"``, ``"medium"``, or ``"high"``.
        - ``warnings`` (list[str]): Specific warnings.
        - ``recommendation`` (str): Suggested action.
        - ``file_size_mb`` (float): File size in megabytes.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    warnings_list: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = path.stat().st_size
    result: dict[str, Any] = {
        "safe": True,
        "risk_level": "low",
        "warnings": warnings_list,
        "recommendation": "",
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }

    # General warning about pickle format
    warnings_list.append(
        "Pickle files can execute arbitrary code during deserialization. "
        "Prefer safetensors format whenever possible."
    )

    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        result["risk_level"] = "high"
        warnings_list.append(
            f"File has '{suffix}' extension. Raw pickle files pose a high "
            "security risk. Convert to safetensors before use."
        )
    elif suffix in (".bin", ".pt", ".pth"):
        # PyTorch checkpoint files use pickle internally
        result["risk_level"] = "medium"
        warnings_list.append(
            f"File has '{suffix}' extension (PyTorch format). "
            "These use pickle serialization internally."
        )

    # Scan first 10MB for suspicious patterns
    scan_size = min(file_size, 10 * 1024 * 1024)
    suspicious_imports: list[str] = []

    try:
        with open(path, "rb") as f:
            data = f.read(scan_size)

        # Look for GLOBAL opcode followed by module.callable patterns
        # The GLOBAL opcode format is: c<module>\n<name>\n
        import_pattern = re.compile(rb"c([a-zA-Z_][\w.]*)\n([a-zA-Z_][\w]*)\n")
        for match in import_pattern.finditer(data):
            module = match.group(1).decode("utf-8", errors="replace")
            name = match.group(2).decode("utf-8", errors="replace")
            full_ref = f"{module}.{name}"

            # Check if the module is known-safe
            module_root = module.split(".")[0]
            if module_root not in _SAFE_MODULES:
                suspicious_imports.append(full_ref)

        # Check for common attack patterns
        dangerous_patterns = [
            (b"os.system", "os.system call detected"),
            (b"subprocess", "subprocess module reference detected"),
            (b"exec(", "exec() call detected"),
            (b"eval(", "eval() call detected"),
            (b"__import__", "__import__ reference detected"),
            (b"builtins", "builtins module reference detected"),
            (b"shutil.rmtree", "shutil.rmtree detected"),
            (b"socket", "socket module reference detected"),
            (b"requests.get", "HTTP request detected"),
            (b"urllib", "urllib module reference detected"),
        ]

        for pattern, description in dangerous_patterns:
            if pattern in data:
                suspicious_imports.append(description)
                result["risk_level"] = "high"

    except OSError as exc:
        warnings_list.append(f"Could not scan file contents: {exc}")

    if suspicious_imports:
        result["safe"] = False
        result["risk_level"] = "high"
        warnings_list.append(f"Suspicious imports/patterns found: {', '.join(suspicious_imports)}")
        result["recommendation"] = (
            "HIGH RISK: This file contains suspicious code references. "
            "Do NOT load this file. Convert to safetensors from a trusted source."
        )
    elif result["risk_level"] == "medium":
        result["recommendation"] = (
            "MEDIUM RISK: Standard PyTorch checkpoint. Consider converting to "
            "safetensors for safer loading: "
            "`from safetensors.torch import save_file`"
        )
    else:
        result["recommendation"] = (
            "LOW RISK: No obviously dangerous patterns detected. However, "
            "safetensors format is always preferred for security."
        )

    logger.info(
        "Pickle safety check for %s: risk_level=%s, safe=%s",
        path.name,
        result["risk_level"],
        result["safe"],
    )

    return result


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def load_env_config(
    env_file: str | Path = ".env",
    override: bool = False,
) -> dict[str, str]:
    """Load configuration from a ``.env`` file.

    Parses a simple ``KEY=VALUE`` format file and optionally sets the
    values as environment variables.

    Parameters
    ----------
    env_file:
        Path to the ``.env`` file.
    override:
        If *True*, overwrite existing environment variables.  By default,
        existing variables take precedence.

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs from the file.
    """
    env_path = Path(env_file)
    loaded: dict[str, str] = {}

    if not env_path.exists():
        logger.debug("No .env file found at %s", env_path.absolute())
        return loaded

    try:
        with open(env_path, encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Handle export prefix
                if line.startswith("export "):
                    line = line[7:].strip()

                if "=" not in line:
                    logger.warning(
                        ".env line %d: no '=' found, skipping: %s",
                        line_number,
                        line[:50],
                    )
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Remove surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]

                if not key:
                    continue

                # Validate key format (alphanumeric + underscores)
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                    logger.warning(
                        ".env line %d: invalid key format '%s', skipping",
                        line_number,
                        key[:50],
                    )
                    continue

                loaded[key] = value

                # Set environment variable
                if override or key not in os.environ:
                    os.environ[key] = value

    except OSError as exc:
        logger.error("Failed to read .env file: %s", exc)

    logger.info("Loaded %d variables from %s", len(loaded), env_path)
    return loaded


# ---------------------------------------------------------------------------
# Sensitive value masking
# ---------------------------------------------------------------------------


def mask_sensitive_values(
    config_dict: dict[str, Any],
    mask: str = _MASK,
    additional_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deep copy of a config dict with sensitive values masked.

    Parameters
    ----------
    config_dict:
        The configuration dictionary to mask.
    mask:
        Replacement string for sensitive values.
    additional_patterns:
        Extra regex patterns for key names to consider sensitive.

    Returns
    -------
    dict[str, Any]
        A new dictionary with sensitive values replaced by ``mask``.
    """
    patterns = list(_SENSITIVE_PATTERNS)
    if additional_patterns:
        patterns.extend(re.compile(p) for p in additional_patterns)

    def _is_sensitive(key: str) -> bool:
        return any(pat.search(key) for pat in patterns)

    def _mask_recursive(obj: Any, parent_key: str = "") -> Any:
        if isinstance(obj, dict):
            masked = {}
            for k, v in obj.items():
                str_key = str(k)
                if _is_sensitive(str_key) and isinstance(v, str) and v:
                    masked[k] = mask
                else:
                    masked[k] = _mask_recursive(v, parent_key=str_key)
            return masked
        elif isinstance(obj, (list, tuple)):
            result = [_mask_recursive(item, parent_key) for item in obj]
            return type(obj)(result) if isinstance(obj, tuple) else result
        elif isinstance(obj, str) and _is_sensitive(parent_key) and obj:
            return mask
        else:
            return deepcopy(obj) if isinstance(obj, (dict, list)) else obj

    return _mask_recursive(config_dict)


def is_safe_path(path: str | Path, allowed_roots: list[Path] | None = None) -> bool:
    """Check whether a file path is safe (no directory traversal, etc.).

    Parameters
    ----------
    path:
        The path to validate.
    allowed_roots:
        If provided, the resolved path must be under one of these roots.

    Returns
    -------
    bool
        *True* if the path is considered safe.
    """
    p = Path(path).resolve()

    # Block obvious traversal attempts
    parts = str(path).replace("\\", "/").split("/")
    if ".." in parts:
        logger.warning("Path traversal detected: %s", path)
        return False

    # Check against allowed roots
    if allowed_roots:
        for root in allowed_roots:
            try:
                p.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        logger.warning(
            "Path %s is not under any allowed root: %s",
            path,
            [str(r) for r in allowed_roots],
        )
        return False

    return True
