"""System execution tools for LLM Forge.

Provides secure, permission-gated tools that give the LLM Forge manager
the ability to execute actions on the user's machine: running commands,
reading/writing files, converting documents, installing packages, and
fetching URLs.

Security is enforced via the ``PermissionSystem`` class which categorises
every tool call as ALWAYS_ALLOW, ASK_FIRST, or ALWAYS_BLOCK and validates
arguments against a blocked-patterns list before execution.
"""

from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Project root is resolved once at import time.  Callers can override via
# ``set_project_dir()`` if the working directory differs from the project.
_project_dir: Path = Path.cwd()

# Extensions that are almost certainly binary data.
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svg",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".whl",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".gguf",
        ".safetensors",
        ".pt",
        ".pth",
        ".onnx",
        ".pb",
        ".tflite",
        ".model",
    }
)

# Shell commands/patterns that are always blocked.
_BLOCKED_COMMAND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\s*"),  # rm -rf /
    re.compile(r"\bsudo\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b\s+.*\bof=/dev/"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\b>\s*/dev/sd[a-z]"),
    re.compile(r"\brm\s+(-\w*\s+)*-\w*r\w*\s+~\s*"),  # rm -rf ~
    re.compile(r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/home\b"),
    re.compile(r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/Users\b"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"),  # fork bomb
    re.compile(r"--no-preserve-root"),  # rm --no-preserve-root bypass
    re.compile(r"\$\("),  # subshell injection via $(...)
    re.compile(r"`[^`]+`"),  # subshell injection via backticks
    # Block command chaining with dangerous commands
    re.compile(r"[;&|]+\s*(?:rm|sudo|shutdown|reboot|mkfs)\b"),
    re.compile(r"\b(?:rm|sudo|shutdown|reboot|mkfs)\b.*[;&|]+"),
    # --- Additional hardening (security audit 2026-03-19) ---
    # Block newline/carriage-return injection (VULN-01: CRITICAL)
    re.compile(r"[\n\r]"),
    # Block eval/exec which can decode and run arbitrary payloads (VULN-02)
    re.compile(r"\beval\b"),
    re.compile(r"\bexec\b"),
    # Block piping to shell interpreters (VULN-03: HIGH)
    re.compile(r"\|\s*(?:sh|bash|zsh|dash|ksh|csh|tcsh|fish|python[23]?|perl|ruby|node)\b"),
    # Block curl/wget piped to shell (VULN-04: HIGH)
    re.compile(r"\b(?:curl|wget)\b.*\|\s*(?:sh|bash|zsh|python)"),
    # Block environment variable exfiltration of secrets (VULN-19/20: HIGH)
    re.compile(r"\b(?:env|printenv|set)\s*$", re.MULTILINE),
    re.compile(r"\b(?:env|printenv|set)\s*[|;>&]"),
    re.compile(
        r"\$(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|API_KEY|SECRET|TOKEN|PASSWORD)", re.IGNORECASE
    ),
    # Block hex/octal/base64 decode tricks
    re.compile(r"\bbase64\b.*-d"),
    re.compile(r"\bxxd\b.*-r"),
    # Block process substitution
    re.compile(r"<\("),
    re.compile(r">\("),
    # Block here-string/here-doc abuse with dangerous commands
    re.compile(r"<<<"),
]

# Known malicious PyPI packages.
_BLOCKED_PACKAGES: frozenset[str] = frozenset(
    {
        "os-sys",
        "python-binance",
        "colourama",
        "pipsqlite",
        "coloramma",
        "pipcolorama",
        "requests-toolbet",
        "discordd",
    }
)

# Maximum download size (50 MB).
_MAX_DOWNLOAD_BYTES: int = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def set_project_dir(path: str | Path) -> None:
    """Override the project directory used for path-safety checks."""
    global _project_dir  # noqa: PLW0603
    _project_dir = Path(path).resolve()


def _resolve_project_dir() -> Path:
    return _project_dir


# ---------------------------------------------------------------------------
# Permission System
# ---------------------------------------------------------------------------


class PermissionSystem:
    """Gate-keeper that decides whether a tool call is allowed.

    Categories:
        ALWAYS_ALLOW  - safe read-only or low-risk operations.
        ASK_FIRST     - operations that mutate state; require explicit approval
                        (unless ``auto_approve`` is *True*).
        ALWAYS_BLOCK  - destructive patterns that are never allowed.
    """

    ALWAYS_ALLOW: list[str] = ["read_file", "convert_document", "fetch_url"]
    ASK_FIRST: list[str] = ["run_command", "write_file", "install_package"]
    ALWAYS_BLOCK: list[re.Pattern[str]] = _BLOCKED_COMMAND_PATTERNS

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve
        self.approved_commands: set[str] = set()

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a proposed tool call."""

        # --- ALWAYS_BLOCK patterns (only for run_command) ---
        if tool_name == "run_command":
            cmd = args.get("command", "")
            for pattern in self.ALWAYS_BLOCK:
                if pattern.search(cmd):
                    return (
                        False,
                        f"BLOCKED: command matches dangerous pattern ({pattern.pattern}). "
                        "This operation is never allowed.",
                    )

        # --- ALWAYS_ALLOW ---
        if tool_name in self.ALWAYS_ALLOW:
            return True, "Tool is in the always-allow list."

        # --- ASK_FIRST ---
        if tool_name in self.ASK_FIRST:
            # If auto-approve is on, allow immediately.
            if self.auto_approve:
                return True, "Auto-approved."

            # Check if user already approved an identical command.
            sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if sig in self.approved_commands:
                return True, "Previously approved by user."

            return (
                False,
                f"Tool '{tool_name}' requires user approval. "
                "Pass auto_approve=True or call approve() first.",
            )

        # Unknown tool name — be safe, deny.
        return False, f"Unknown execution tool: {tool_name}"

    def approve(self, tool_name: str, args: dict) -> None:
        """Record that the user has approved a specific call."""
        sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        self.approved_commands.add(sig)


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------


def _is_path_allowed(path: str | Path, *, allow_home: bool = False) -> bool:
    """Return *True* if *path* is inside the project directory (or home when allowed).

    Uses ``Path.is_relative_to`` (Python 3.9+) instead of string-prefix
    matching to avoid false positives when directory names share a common
    prefix (e.g. ``/home/user/app`` vs ``/home/user/application-secret``).
    """
    resolved = Path(path).expanduser().resolve()
    project = _resolve_project_dir()
    home = Path.home().resolve()

    if resolved.is_relative_to(project):
        return True
    if allow_home and resolved.is_relative_to(home):
        return True
    # Also allow /tmp and platform temp dirs
    import tempfile

    tmp = Path(tempfile.gettempdir()).resolve()
    return resolved.is_relative_to(tmp)


def _is_binary_file(path: Path) -> bool:
    """Heuristic check: extension in the binary list or first 8 KB contain null bytes."""
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Tool 1: run_command
# ---------------------------------------------------------------------------


def run_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
) -> str:
    """Execute a shell command securely via subprocess.

    Returns JSON with ``stdout``, ``stderr``, ``return_code``.
    """
    # Clamp timeout
    timeout = max(1, min(timeout, 600))

    # Validate cwd
    if cwd is not None:
        cwd_path = Path(cwd).expanduser().resolve()
        if not _is_path_allowed(cwd_path, allow_home=True):
            return json.dumps(
                {
                    "error": f"Working directory '{cwd}' is outside the project and home directories.",
                    "status": "blocked",
                }
            )
        if not cwd_path.is_dir():
            return json.dumps(
                {"error": f"Working directory does not exist: {cwd}", "status": "error"}
            )
    else:
        cwd_path = _resolve_project_dir()

    # Blocked patterns checked by PermissionSystem.check() before we get here,
    # but double-check as a safety net.
    for pattern in _BLOCKED_COMMAND_PATTERNS:
        if pattern.search(command):
            return json.dumps(
                {
                    "error": "Command matches a blocked pattern and cannot be executed.",
                    "status": "blocked",
                }
            )

    try:
        # Scrub sensitive environment variables before executing commands
        # to prevent credential exfiltration (VULN-20).
        safe_env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            not in {
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "AWS_SECRET_ACCESS_KEY",
                "GITHUB_TOKEN",
                "API_KEY",
                "SECRET_KEY",
            }
        }
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd_path),
            env=safe_env,
        )
        return json.dumps(
            {
                "stdout": result.stdout[-10_000:] if result.stdout else "",
                "stderr": result.stderr[-5_000:] if result.stderr else "",
                "return_code": result.returncode,
                "status": "ok",
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "error": f"Command timed out after {timeout} seconds.",
                "status": "timeout",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


# ---------------------------------------------------------------------------
# Tool 2: read_file
# ---------------------------------------------------------------------------


# Sensitive file names that should never be exposed to the LLM.
_SENSITIVE_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".api_key",
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "credentials.json",
        ".netrc",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        ".pgpass",
        ".my.cnf",
    }
)

# Sensitive directory components that should be blocked from read access.
_SENSITIVE_PATH_PARTS: frozenset[str] = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".gcloud",
    }
)


def read_file(path: str, max_lines: int = 500) -> str:
    """Read a file's contents securely.

    Returns JSON with ``content``, ``line_count``, ``size_bytes``, ``encoding``.
    """
    file_path = Path(path).expanduser().resolve()

    if not _is_path_allowed(file_path, allow_home=True):
        return json.dumps(
            {
                "error": f"Path '{path}' is outside the allowed directories.",
                "status": "blocked",
            }
        )

    # Block reading of sensitive files (API keys, SSH keys, credentials)
    if file_path.name in _SENSITIVE_FILE_NAMES:
        return json.dumps(
            {
                "error": f"Access to sensitive file '{file_path.name}' is blocked.",
                "status": "blocked",
            }
        )
    # Block reading from sensitive directories
    if any(part in _SENSITIVE_PATH_PARTS for part in file_path.parts):
        return json.dumps(
            {
                "error": "Access to sensitive directory is blocked.",
                "status": "blocked",
            }
        )

    if not file_path.exists():
        return json.dumps({"error": f"File not found: {path}", "status": "not_found"})

    if not file_path.is_file():
        return json.dumps({"error": f"Path is not a file: {path}", "status": "error"})

    if _is_binary_file(file_path):
        return json.dumps(
            {
                "error": f"File appears to be binary ({file_path.suffix}); skipped.",
                "status": "binary_skipped",
                "size_bytes": file_path.stat().st_size,
            }
        )

    size_bytes = file_path.stat().st_size

    # Try UTF-8 first, fall back to latin-1
    encoding = "utf-8"
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = file_path.read_text(encoding="latin-1")

    lines = text.splitlines()
    total_lines = len(lines)
    truncated = False

    if total_lines > max_lines:
        lines = lines[:max_lines]
        truncated = True

    content = "\n".join(lines)
    result: dict = {
        "content": content,
        "line_count": total_lines,
        "size_bytes": size_bytes,
        "encoding": encoding,
        "status": "ok",
    }
    if truncated:
        result["truncated"] = True
        result["note"] = (
            f"File has {total_lines} lines; showing first {max_lines}. "
            "Increase max_lines to see more."
        )
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 3: write_file
# ---------------------------------------------------------------------------


_MAX_WRITE_BYTES: int = 10 * 1024 * 1024  # 10 MB max write size

_VALID_WRITE_MODES: frozenset[str] = frozenset({"create", "overwrite", "append"})


def write_file(path: str, content: str, mode: str = "create") -> str:
    """Create, overwrite, or append to a file.

    Modes:
        create    - fail if the file already exists (safe default).
        overwrite - overwrite an existing file.
        append    - append to an existing file.
    """
    # Validate mode to prevent fall-through to overwrite (VULN-14)
    if mode not in _VALID_WRITE_MODES:
        return json.dumps(
            {
                "error": f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(_VALID_WRITE_MODES))}",
                "status": "error",
            }
        )

    file_path = Path(path).expanduser().resolve()

    if not _is_path_allowed(file_path, allow_home=False):
        return json.dumps(
            {
                "error": f"Path '{path}' is outside the project directory.",
                "status": "blocked",
            }
        )

    # Block writes to sensitive files (VULN-18: prevent LLM reading API keys)
    if file_path.name in (".api_key", ".env", "credentials.json", ".netrc"):
        return json.dumps(
            {
                "error": f"Cannot write to sensitive file: {file_path.name}",
                "status": "blocked",
            }
        )

    # Enforce maximum write size to prevent disk-fill DoS (VULN-11)
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > _MAX_WRITE_BYTES:
        return json.dumps(
            {
                "error": f"Content too large ({content_bytes} bytes, max {_MAX_WRITE_BYTES}).",
                "status": "too_large",
            }
        )

    if mode == "create" and file_path.exists():
        return json.dumps(
            {
                "error": f"File already exists: {path}. Use mode='overwrite' or mode='append'.",
                "status": "exists",
            }
        )

    if mode == "append" and not file_path.exists():
        return json.dumps(
            {
                "error": f"File does not exist for append: {path}.",
                "status": "not_found",
            }
        )

    # Ensure parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "append":
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            file_path.write_text(content, encoding="utf-8")

        return json.dumps(
            {
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8")),
                "status": "ok",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


# ---------------------------------------------------------------------------
# Tool 4: convert_document
# ---------------------------------------------------------------------------


def convert_document(input_path: str, output_format: str = "txt") -> str:
    """Convert DOCX, PDF, HTML, or MD to plain text.

    Returns JSON with ``output_path``, ``extracted_text_preview``,
    ``char_count``, ``word_count``.
    """
    src = Path(input_path).expanduser().resolve()

    # Security: restrict to project directory and home (VULN-09)
    if not _is_path_allowed(src, allow_home=True):
        return json.dumps(
            {
                "error": f"Path '{input_path}' is outside the allowed directories.",
                "status": "blocked",
            }
        )

    if not src.exists():
        return json.dumps({"error": f"File not found: {input_path}", "status": "not_found"})

    suffix = src.suffix.lower()
    text = ""

    try:
        if suffix in (".md", ".txt"):
            text = src.read_text(encoding="utf-8")

        elif suffix == ".docx":
            text = _convert_docx(src)

        elif suffix == ".pdf":
            text = _convert_pdf(src)

        elif suffix in (".html", ".htm"):
            text = _convert_html(src)

        else:
            return json.dumps(
                {
                    "error": f"Unsupported file type: {suffix}. Supported: .docx, .pdf, .html, .md, .txt",
                    "status": "unsupported",
                }
            )

    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}", "status": "error"})

    # Write output alongside the source
    out_path = src.with_suffix(f".{output_format}")

    try:
        out_path.write_text(text, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Failed to write output: {e}", "status": "error"})

    preview = text[:500]
    words = text.split()

    return json.dumps(
        {
            "output_path": str(out_path),
            "extracted_text_preview": preview,
            "char_count": len(text),
            "word_count": len(words),
            "status": "ok",
        }
    )


def _convert_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        import docx  # python-docx

        doc = docx.Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        # macOS fallback: textutil
        if sys.platform == "darwin":
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        raise ImportError(
            "python-docx is required for DOCX conversion. Install with: pip install python-docx"
        )


def _convert_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        return "\n\n".join(pages)
    except ImportError:
        # Try pdftotext command-line fallback
        try:
            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            pass
        raise ImportError(
            "pdfplumber is required for PDF conversion. Install with: pip install pdfplumber"
        )


def _convert_html(path: Path) -> str:
    """Extract text from an HTML file."""
    raw = path.read_text(encoding="utf-8")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        pass

    # Crude fallback: strip HTML tags with regex
    clean = re.sub(r"<[^>]+>", " ", raw)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Tool 5: install_package
# ---------------------------------------------------------------------------


def install_package(package_name: str) -> str:
    """Install a Python package from PyPI.

    Returns JSON with ``status`` and ``package_version``.
    """
    # Normalise
    name = package_name.strip()

    # Security: reject arbitrary URLs and direct references (VULN-13)
    if "://" in name or name.startswith("git+"):
        return json.dumps(
            {
                "error": "Only PyPI package names are allowed (no URLs).",
                "status": "blocked",
            }
        )
    # Block PEP 440 direct references (package @ URL) and local paths
    if "@" in name:
        return json.dumps(
            {
                "error": "Direct references (@ notation) are not allowed.",
                "status": "blocked",
            }
        )
    # Block local file paths
    if name.startswith(("/", "./", "../", "~", "\\")):
        return json.dumps(
            {
                "error": "Local file paths are not allowed. Use PyPI package names only.",
                "status": "blocked",
            }
        )
    # Block pip flags embedded in the package name (e.g. "--index-url evil.com pkg")
    if name.startswith("-") or " -" in name:
        return json.dumps(
            {
                "error": "Package names cannot contain pip flags.",
                "status": "blocked",
            }
        )
    # Only allow valid PyPI package name characters (PEP 508)
    base_name_match = re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?", name)
    if not base_name_match:
        return json.dumps(
            {
                "error": f"Invalid package name: {name}",
                "status": "blocked",
            }
        )

    # Security: blocked packages
    base_name = re.split(r"[<>=!~\[@]", name)[0].strip().lower()
    if base_name in _BLOCKED_PACKAGES:
        return json.dumps(
            {
                "error": f"Package '{base_name}' is on the blocked list (known malware).",
                "status": "blocked",
            }
        )

    # Check if already installed
    try:
        version = importlib.metadata.version(base_name)
        return json.dumps(
            {
                "status": "already_installed",
                "package": base_name,
                "package_version": version,
            }
        )
    except importlib.metadata.PackageNotFoundError:
        pass

    # Prefer uv, fall back to pip
    installers: list[list[str]] = []
    import shutil

    if shutil.which("uv"):
        installers.append(["uv", "pip", "install", name])
    installers.append([sys.executable, "-m", "pip", "install", "-q", name])

    for cmd in installers:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Re-check version after install
                try:
                    version = importlib.metadata.version(base_name)
                except Exception:
                    version = "unknown"
                return json.dumps(
                    {
                        "status": "installed",
                        "package": base_name,
                        "package_version": version,
                    }
                )
        except Exception:
            continue

    return json.dumps(
        {
            "error": f"Failed to install '{name}' with all available installers.",
            "status": "error",
        }
    )


# ---------------------------------------------------------------------------
# SSRF-safe redirect handler (security audit 2026-03-19)
# ---------------------------------------------------------------------------


class _SSRFSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that blocks redirects to private/loopback IPs.

    Prevents SSRF attacks where a public URL redirects (302) to an internal
    service such as ``http://169.254.169.254/`` (cloud metadata) or
    ``http://127.0.0.1:8080/admin``.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        redirect_hostname = parsed.hostname or ""

        # Block non-http(s) scheme redirects
        if parsed.scheme not in ("http", "https"):
            raise urllib.error.URLError(f"Redirect to non-http scheme blocked: {parsed.scheme}")

        # Block redirects to localhost aliases
        if redirect_hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            raise urllib.error.URLError(f"Redirect to localhost blocked: {redirect_hostname}")

        # Resolve the redirect target and check for private IPs
        try:
            addrinfos = socket.getaddrinfo(redirect_hostname, None)
            for _family, _type, _proto, _canonname, sockaddr in addrinfos:
                ip_str = sockaddr[0]
                addr = ipaddress.ip_address(ip_str)
                if addr.is_private or addr.is_link_local or addr.is_loopback or addr.is_reserved:
                    raise urllib.error.URLError(
                        f"Redirect to private/reserved IP blocked: {ip_str}"
                    )
        except socket.gaierror:
            raise urllib.error.URLError(f"Could not resolve redirect target: {redirect_hostname}")

        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ---------------------------------------------------------------------------
# Tool 6: fetch_url
# ---------------------------------------------------------------------------


def fetch_url(url: str, output_path: str | None = None) -> str:
    """Download content from a URL.

    For HTML pages, extracts text content. For other content types, saves raw
    bytes to *output_path* (or a temp file).

    Returns JSON with ``content_preview``, ``content_type``, ``size_bytes``,
    ``saved_path``.
    """
    # Security: only HTTP/HTTPS
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return json.dumps(
            {
                "error": f"Only http/https URLs are allowed (got '{parsed.scheme}://').",
                "status": "blocked",
            }
        )

    # Security: block localhost unless explicitly allowed
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return json.dumps(
            {
                "error": "Localhost URLs are not allowed for security reasons.",
                "status": "blocked",
            }
        )

    # Security: block private/link-local IPs (SSRF protection)
    # Resolve once and validate all addresses before making the request.
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
        for _family, _type, _proto, _canonname, sockaddr in addrinfos:
            ip_str = sockaddr[0]
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_link_local or addr.is_loopback or addr.is_reserved:
                return json.dumps(
                    {
                        "error": (
                            f"URL resolves to private/reserved IP ({ip_str}). "
                            "Blocked to prevent SSRF attacks."
                        ),
                        "status": "blocked",
                    }
                )
    except socket.gaierror:
        return json.dumps(
            {
                "error": f"Could not resolve hostname: {hostname}",
                "status": "error",
            }
        )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LLM-Forge/0.3 (https://github.com/llm-forge)"},
        )
        # Use a custom opener that validates redirect targets against SSRF
        # (blocks redirects to private/loopback/link-local IPs).
        opener = urllib.request.build_opener(_SSRFSafeRedirectHandler)
        with opener.open(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "application/octet-stream")

            # Check content-length header first
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_DOWNLOAD_BYTES:
                return json.dumps(
                    {
                        "error": f"Content too large ({int(content_length)} bytes, max {_MAX_DOWNLOAD_BYTES}).",
                        "status": "too_large",
                    }
                )

            # Read in chunks up to max
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    return json.dumps(
                        {
                            "error": f"Download exceeded maximum size ({_MAX_DOWNLOAD_BYTES} bytes).",
                            "status": "too_large",
                        }
                    )
                chunks.append(chunk)
            data = b"".join(chunks)

        size_bytes = len(data)
        saved_path: str | None = None

        # If it looks like HTML, extract text
        if "text/html" in content_type:
            try:
                html_text = data.decode("utf-8", errors="replace")
            except Exception:
                html_text = data.decode("latin-1")

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                text = re.sub(r"<[^>]+>", " ", html_text)
                text = re.sub(r"\s+", " ", text).strip()

            if output_path:
                out = Path(output_path).expanduser().resolve()
                if not _is_path_allowed(out, allow_home=False):
                    return json.dumps(
                        {
                            "error": f"Output path '{output_path}' is outside the project directory.",
                            "status": "blocked",
                        }
                    )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
                saved_path = str(out)

            return json.dumps(
                {
                    "content_preview": text[:1000],
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "saved_path": saved_path,
                    "status": "ok",
                }
            )

        # Non-HTML: save raw bytes
        if output_path:
            out = Path(output_path).expanduser().resolve()
            if not _is_path_allowed(out, allow_home=False):
                return json.dumps(
                    {
                        "error": f"Output path '{output_path}' is outside the project directory.",
                        "status": "blocked",
                    }
                )
        else:
            import tempfile

            fd, tmp_name = tempfile.mkstemp(suffix=Path(parsed.path).suffix or ".bin")
            os.close(fd)
            out = Path(tmp_name)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        saved_path = str(out)

        # Text preview for text types
        preview = ""
        if "text/" in content_type:
            try:
                preview = data.decode("utf-8", errors="replace")[:1000]
            except Exception:
                preview = "(binary content)"
        else:
            preview = f"(binary content, {size_bytes} bytes)"

        return json.dumps(
            {
                "content_preview": preview,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "saved_path": saved_path,
                "status": "ok",
            }
        )

    except urllib.error.URLError as e:
        return json.dumps({"error": f"URL error: {e}", "status": "error"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


# ---------------------------------------------------------------------------
# Tool Definitions (JSON schemas for Claude API)
# ---------------------------------------------------------------------------

EXECUTION_TOOLS: list[dict] = [
    {
        "name": "run_command",
        "description": (
            "Execute a shell command on the user's machine. Returns stdout, stderr, "
            "and return code. Blocked commands (rm -rf /, sudo, shutdown, etc.) are "
            "rejected. Requires user approval unless auto_approve is enabled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (defaults to project root)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 600)",
                    "default": 120,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file's contents. Returns the text content, line count, size, "
            "and encoding. Binary files are skipped. Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (default 500)",
                    "default": 500,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create, overwrite, or append to a file within the project directory. "
            "mode='create' (default) fails if the file exists. mode='overwrite' "
            "replaces content. mode='append' adds to the end."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (within project directory)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "Write mode (default: create)",
                    "default": "create",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "convert_document",
        "description": (
            "Convert a document (DOCX, PDF, HTML, MD, TXT) to plain text. "
            "Returns the extracted text preview, character count, and word count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the document file",
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format (default: txt)",
                    "default": "txt",
                },
            },
            "required": ["input_path"],
        },
    },
    {
        "name": "install_package",
        "description": (
            "Install a Python package from PyPI. Checks if already installed first. "
            "Only PyPI names allowed (no arbitrary URLs). Known malware packages "
            "are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "PyPI package name (e.g., 'requests', 'python-docx')",
                },
            },
            "required": ["package_name"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Download content from a URL. For HTML pages, extracts text. "
            "For other content, saves to output_path. Max 50MB. "
            "Only http/https allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http or https only)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to save downloaded content (optional)",
                },
            },
            "required": ["url"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def execute_execution_tool(name: str, input_data: dict) -> str:
    """Dispatch an execution tool call by name. Returns JSON string."""
    if name == "run_command":
        return run_command(
            command=input_data["command"],
            cwd=input_data.get("cwd"),
            timeout=input_data.get("timeout", 120),
        )
    elif name == "read_file":
        return read_file(
            path=input_data["path"],
            max_lines=input_data.get("max_lines", 500),
        )
    elif name == "write_file":
        return write_file(
            path=input_data["path"],
            content=input_data["content"],
            mode=input_data.get("mode", "create"),
        )
    elif name == "convert_document":
        return convert_document(
            input_path=input_data["input_path"],
            output_format=input_data.get("output_format", "txt"),
        )
    elif name == "install_package":
        return install_package(
            package_name=input_data["package_name"],
        )
    elif name == "fetch_url":
        return fetch_url(
            url=input_data["url"],
            output_path=input_data.get("output_path"),
        )
    else:
        return json.dumps({"error": f"Unknown execution tool: {name}"})


# Set of all execution tool names for quick membership testing.
EXECUTION_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in EXECUTION_TOOLS)
