"""Tests for the system execution tools module.

Covers read_file, write_file, run_command, convert_document,
install_package, fetch_url, and the PermissionSystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_forge.chat.execution import (
    PermissionSystem,
    convert_document,
    execute_execution_tool,
    fetch_url,
    install_package,
    read_file,
    run_command,
    set_project_dir,
    write_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_project_to_tmp(tmp_path: Path) -> None:
    """Point the execution module's project directory at the test tmp_path
    so that path-safety checks pass for files created during tests."""
    set_project_dir(tmp_path)


# ===================================================================
# read_file
# ===================================================================


class TestReadFile:
    """Tests for the read_file tool."""

    def test_read_file_exists(self, tmp_path: Path) -> None:
        """Read a real file that exists — returns content and metadata."""
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!\nSecond line.\n", encoding="utf-8")

        result = json.loads(read_file(str(f)))
        assert result["status"] == "ok"
        assert "Hello, world!" in result["content"]
        assert result["line_count"] == 2  # two non-empty lines
        assert result["encoding"] == "utf-8"
        assert result["size_bytes"] > 0

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        """Reading a file that does not exist returns not_found."""
        result = json.loads(read_file(str(tmp_path / "missing.txt")))
        assert result["status"] == "not_found"
        assert "error" in result

    def test_read_file_binary_skipped(self, tmp_path: Path) -> None:
        """Binary files are detected and skipped."""
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = json.loads(read_file(str(f)))
        assert result["status"] == "binary_skipped"
        assert "binary" in result["error"].lower()

    def test_read_file_truncation(self, tmp_path: Path) -> None:
        """Large files are truncated to max_lines."""
        f = tmp_path / "big.txt"
        lines = [f"line {i}" for i in range(1000)]
        f.write_text("\n".join(lines), encoding="utf-8")

        result = json.loads(read_file(str(f), max_lines=10))
        assert result["status"] == "ok"
        assert result["truncated"] is True
        assert result["line_count"] == 1000
        # Content should contain only first 10 lines
        content_lines = result["content"].split("\n")
        assert len(content_lines) == 10

    def test_read_file_latin1_fallback(self, tmp_path: Path) -> None:
        """Files that are not valid UTF-8 fall back to latin-1."""
        f = tmp_path / "latin.txt"
        f.write_bytes(b"caf\xe9\n")

        result = json.loads(read_file(str(f)))
        assert result["status"] == "ok"
        assert result["encoding"] == "latin-1"


# ===================================================================
# write_file
# ===================================================================


class TestWriteFile:
    """Tests for the write_file tool."""

    def test_write_file_create(self, tmp_path: Path) -> None:
        """Create a new file successfully."""
        path = str(tmp_path / "new_file.txt")
        result = json.loads(write_file(path, "Hello!", mode="create"))
        assert result["status"] == "ok"
        assert result["bytes_written"] == len(b"Hello!")
        assert Path(path).read_text() == "Hello!"

    def test_write_file_no_overwrite(self, tmp_path: Path) -> None:
        """mode='create' fails if the file already exists."""
        path = tmp_path / "existing.txt"
        path.write_text("original")

        result = json.loads(write_file(str(path), "replacement", mode="create"))
        assert result["status"] == "exists"
        assert "already exists" in result["error"]
        # Original content unchanged
        assert path.read_text() == "original"

    def test_write_file_overwrite(self, tmp_path: Path) -> None:
        """mode='overwrite' replaces the file content."""
        path = tmp_path / "overwrite_me.txt"
        path.write_text("old content")

        result = json.loads(write_file(str(path), "new content", mode="overwrite"))
        assert result["status"] == "ok"
        assert path.read_text() == "new content"

    def test_write_file_append(self, tmp_path: Path) -> None:
        """mode='append' adds content to the end of a file."""
        path = tmp_path / "appendable.txt"
        path.write_text("start")

        result = json.loads(write_file(str(path), " end", mode="append"))
        assert result["status"] == "ok"
        assert path.read_text() == "start end"

    def test_write_file_creates_parents(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        path = str(tmp_path / "deep" / "nested" / "dir" / "file.txt")
        result = json.loads(write_file(path, "content", mode="create"))
        assert result["status"] == "ok"
        assert Path(path).exists()


# ===================================================================
# run_command
# ===================================================================


class TestRunCommand:
    """Tests for the run_command tool."""

    def test_run_command_simple(self) -> None:
        """Run a simple echo command."""
        result = json.loads(run_command("echo hello"))
        assert result["status"] == "ok"
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]

    def test_run_command_blocked(self) -> None:
        """Dangerous commands like 'rm -rf /' are blocked."""
        result = json.loads(run_command("rm -rf /"))
        assert result["status"] == "blocked"
        assert "error" in result

    def test_run_command_blocked_sudo(self) -> None:
        """Commands with sudo are blocked."""
        result = json.loads(run_command("sudo apt-get install something"))
        assert result["status"] == "blocked"

    def test_run_command_timeout(self) -> None:
        """Commands that exceed the timeout are killed."""
        result = json.loads(run_command("sleep 10", timeout=1))
        assert result["status"] == "timeout"
        assert "timed out" in result["error"].lower()

    def test_run_command_stderr(self) -> None:
        """Commands that write to stderr capture it."""
        result = json.loads(run_command("echo oops >&2"))
        assert result["status"] == "ok"
        assert "oops" in result["stderr"]

    def test_run_command_nonzero_exit(self) -> None:
        """Commands that fail return a non-zero exit code."""
        result = json.loads(run_command("exit 42"))
        assert result["status"] == "ok"
        assert result["return_code"] == 42

    def test_run_command_max_timeout_clamped(self) -> None:
        """Timeout is clamped to 600 seconds maximum."""
        # We just check it doesn't error — the timeout is internally clamped.
        result = json.loads(run_command("echo clamped", timeout=99999))
        assert result["status"] == "ok"
        assert "clamped" in result["stdout"]


# ===================================================================
# convert_document
# ===================================================================


class TestConvertDocument:
    """Tests for the convert_document tool."""

    def test_convert_document_txt(self, tmp_path: Path) -> None:
        """Converting a .txt file returns its content directly."""
        f = tmp_path / "readme.txt"
        f.write_text("This is plain text content.\nSecond line.", encoding="utf-8")

        result = json.loads(convert_document(str(f)))
        assert result["status"] == "ok"
        assert "plain text content" in result["extracted_text_preview"]
        assert result["char_count"] > 0
        assert result["word_count"] >= 5

    def test_convert_document_markdown(self, tmp_path: Path) -> None:
        """Converting a .md file returns its content (markdown is already text)."""
        f = tmp_path / "notes.md"
        f.write_text("# Heading\n\nSome paragraph text.", encoding="utf-8")

        result = json.loads(convert_document(str(f)))
        assert result["status"] == "ok"
        assert "Heading" in result["extracted_text_preview"]

    def test_convert_document_html(self, tmp_path: Path) -> None:
        """Converting an HTML file strips tags and returns text."""
        f = tmp_path / "page.html"
        f.write_text(
            "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>",
            encoding="utf-8",
        )

        result = json.loads(convert_document(str(f)))
        assert result["status"] == "ok"
        assert "Title" in result["extracted_text_preview"]
        assert "Paragraph text" in result["extracted_text_preview"]
        # Tags should be stripped
        assert "<h1>" not in result["extracted_text_preview"]

    def test_convert_document_not_found(self, tmp_path: Path) -> None:
        """Converting a non-existent file returns not_found."""
        result = json.loads(convert_document(str(tmp_path / "ghost.pdf")))
        assert result["status"] == "not_found"

    def test_convert_document_unsupported(self, tmp_path: Path) -> None:
        """Unsupported file types return an error."""
        f = tmp_path / "data.xyz"
        f.write_text("stuff")
        result = json.loads(convert_document(str(f)))
        assert result["status"] == "unsupported"


# ===================================================================
# install_package
# ===================================================================


class TestInstallPackage:
    """Tests for the install_package tool."""

    def test_install_package_already_installed(self) -> None:
        """Checking a package that is already installed returns already_installed."""
        result = json.loads(install_package("pytest"))
        assert result["status"] == "already_installed"
        assert result["package"] == "pytest"
        assert result["package_version"]  # non-empty version string

    def test_install_package_blocked_url(self) -> None:
        """Arbitrary URLs are blocked."""
        result = json.loads(install_package("https://evil.com/malware.tar.gz"))
        assert result["status"] == "blocked"

    def test_install_package_blocked_malware(self) -> None:
        """Known malware packages are blocked."""
        result = json.loads(install_package("os-sys"))
        assert result["status"] == "blocked"
        assert "blocked list" in result["error"]

    def test_install_package_git_url_blocked(self) -> None:
        """git+ URLs are blocked."""
        result = json.loads(install_package("git+https://github.com/example/pkg.git"))
        assert result["status"] == "blocked"


# ===================================================================
# fetch_url
# ===================================================================


class TestFetchUrl:
    """Tests for the fetch_url tool."""

    def test_fetch_url_blocked_file_protocol(self) -> None:
        """file:// protocol is blocked."""
        result = json.loads(fetch_url("file:///etc/passwd"))
        assert result["status"] == "blocked"
        assert "http" in result["error"].lower() or "allowed" in result["error"].lower()

    def test_fetch_url_blocked_localhost(self) -> None:
        """Localhost URLs are blocked."""
        result = json.loads(fetch_url("http://localhost:8080/secret"))
        assert result["status"] == "blocked"
        assert "localhost" in result["error"].lower()

    def test_fetch_url_blocked_ftp(self) -> None:
        """FTP protocol is blocked."""
        result = json.loads(fetch_url("ftp://example.com/file.txt"))
        assert result["status"] == "blocked"

    def test_fetch_url_blocked_127(self) -> None:
        """127.0.0.1 is treated as localhost and blocked."""
        result = json.loads(fetch_url("http://127.0.0.1/secret"))
        assert result["status"] == "blocked"


# ===================================================================
# PermissionSystem
# ===================================================================


class TestPermissionSystem:
    """Tests for the PermissionSystem gate-keeper."""

    def test_permission_system_always_allow(self) -> None:
        """Tools in ALWAYS_ALLOW are permitted without approval."""
        ps = PermissionSystem(auto_approve=False)
        allowed, reason = ps.check("read_file", {"path": "/some/file"})
        assert allowed is True

    def test_permission_system_always_allow_convert(self) -> None:
        """convert_document is in ALWAYS_ALLOW."""
        ps = PermissionSystem(auto_approve=False)
        allowed, _ = ps.check("convert_document", {"input_path": "/some/file.pdf"})
        assert allowed is True

    def test_permission_system_always_allow_fetch(self) -> None:
        """fetch_url is in ALWAYS_ALLOW."""
        ps = PermissionSystem(auto_approve=False)
        allowed, _ = ps.check("fetch_url", {"url": "https://example.com"})
        assert allowed is True

    def test_permission_system_ask_first(self) -> None:
        """Tools in ASK_FIRST are blocked without approval (auto_approve=False)."""
        ps = PermissionSystem(auto_approve=False)
        allowed, reason = ps.check("run_command", {"command": "echo hello"})
        assert allowed is False
        assert "approval" in reason.lower()

    def test_permission_system_ask_first_auto_approve(self) -> None:
        """Tools in ASK_FIRST are allowed when auto_approve=True."""
        ps = PermissionSystem(auto_approve=True)
        allowed, reason = ps.check("run_command", {"command": "echo hello"})
        assert allowed is True
        assert "auto" in reason.lower()

    def test_permission_system_blocked(self) -> None:
        """Dangerous commands are always blocked even with auto_approve."""
        ps = PermissionSystem(auto_approve=True)
        allowed, reason = ps.check("run_command", {"command": "rm -rf /"})
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_permission_system_blocked_sudo(self) -> None:
        """sudo commands are always blocked."""
        ps = PermissionSystem(auto_approve=True)
        allowed, reason = ps.check("run_command", {"command": "sudo reboot"})
        assert allowed is False

    def test_permission_system_approve_remembers(self) -> None:
        """After calling approve(), the same command is allowed next time."""
        ps = PermissionSystem(auto_approve=False)
        args = {"command": "echo hello"}

        allowed1, _ = ps.check("run_command", args)
        assert allowed1 is False

        ps.approve("run_command", args)
        allowed2, _ = ps.check("run_command", args)
        assert allowed2 is True

    def test_permission_system_write_file_ask_first(self) -> None:
        """write_file requires approval."""
        ps = PermissionSystem(auto_approve=False)
        allowed, _ = ps.check("write_file", {"path": "/tmp/test", "content": "x"})
        assert allowed is False

    def test_permission_system_install_package_ask_first(self) -> None:
        """install_package requires approval."""
        ps = PermissionSystem(auto_approve=False)
        allowed, _ = ps.check("install_package", {"package_name": "requests"})
        assert allowed is False

    def test_permission_system_unknown_tool(self) -> None:
        """Unknown tool names are denied."""
        ps = PermissionSystem(auto_approve=True)
        allowed, reason = ps.check("destroy_everything", {})
        assert allowed is False
        assert "unknown" in reason.lower()


# ===================================================================
# execute_execution_tool dispatcher
# ===================================================================


class TestExecuteExecutionToolDispatcher:
    """Tests for the execute_execution_tool routing function."""

    def test_dispatch_read_file(self, tmp_path: Path) -> None:
        """Dispatcher correctly routes read_file."""
        f = tmp_path / "dispatch_test.txt"
        f.write_text("dispatch content")
        result = json.loads(execute_execution_tool("read_file", {"path": str(f)}))
        assert result["status"] == "ok"
        assert "dispatch content" in result["content"]

    def test_dispatch_unknown_tool(self) -> None:
        """Dispatcher returns error for unknown tool names."""
        result = json.loads(execute_execution_tool("not_a_tool", {}))
        assert "error" in result

    def test_dispatch_run_command(self) -> None:
        """Dispatcher correctly routes run_command."""
        result = json.loads(
            execute_execution_tool("run_command", {"command": "echo dispatch_test"})
        )
        assert result["status"] == "ok"
        assert "dispatch_test" in result["stdout"]

    def test_dispatch_write_file(self, tmp_path: Path) -> None:
        """Dispatcher correctly routes write_file."""
        path = str(tmp_path / "dispatch_write.txt")
        result = json.loads(
            execute_execution_tool(
                "write_file",
                {"path": path, "content": "written via dispatcher"},
            )
        )
        assert result["status"] == "ok"
        assert Path(path).read_text() == "written via dispatcher"
