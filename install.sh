#!/usr/bin/env bash
# LLM Forge Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash
#
# This script:
# 1. Finds Python 3.10-3.13 on your system (PyTorch has no 3.14 wheels)
# 2. Creates ~/.llm-forge/ with an isolated environment
# 3. Installs llm-forge-new from PyPI
# 4. Makes 'llm-forge' available as a command
# 5. Works on macOS, Linux, and WSL

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# Overridable so the installer can be tested against a scratch directory
# instead of only ever being provable by overwriting the user's real install.
INSTALL_DIR="${LLM_FORGE_HOME:-$HOME/.llm-forge}"

echo ""
echo -e "${BOLD}${CYAN}╭─────────────────────────────────────╮${RESET}"
echo -e "${BOLD}${CYAN}│   LLM Forge — Installer             │${RESET}"
echo -e "${BOLD}${CYAN}│   Build your own AI model            │${RESET}"
echo -e "${BOLD}${CYAN}╰─────────────────────────────────────╯${RESET}"
echo ""

# -----------------------------------------------------------------------
# Step 1: Find Python 3.10-3.13
# -----------------------------------------------------------------------

find_python() {
    local candidates=(
        "python3.13" "python3.12" "python3.11" "python3.10"
        "/opt/homebrew/bin/python3.13" "/opt/homebrew/bin/python3.12"
        "/opt/homebrew/bin/python3.11" "/opt/homebrew/bin/python3.10"
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3.13" "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11" "/usr/local/bin/python3.10"
        "/usr/local/bin/python3"
        "$HOME/.pyenv/shims/python3"
        "$HOME/.local/bin/python3"
        "/usr/bin/python3.12" "/usr/bin/python3.11" "/usr/bin/python3.10"
        "python3" "python"
    )

    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null 2>&1 || [ -x "$cmd" ]; then
            local version
            version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major minor
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            # Upper bound matters: PyTorch publishes no wheels for Python 3.14+,
            # so a bare `python3` pointing at 3.14 would install and then fail
            # at the first `import torch`.
            if [ "$major" = "3" ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo -e "${DIM}Searching for Python 3.10-3.13...${RESET}"

PYTHON_CMD=$(find_python) || true

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Python 3.10-3.13 not found on your system.${RESET}"
    echo -e "${DIM}(PyTorch has no wheels for 3.14+ yet, so a newer Python will not work.)${RESET}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "Install it with Homebrew:"
        echo -e "  ${BOLD}brew install python@3.12${RESET}"
        echo ""
        echo -e "No Homebrew? Install it first:"
        echo -e "  ${BOLD}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${RESET}"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo -e "Install it:"
        echo -e "  ${BOLD}sudo apt update && sudo apt install -y python3.12 python3.12-venv${RESET}  (Ubuntu/Debian)"
        echo -e "  ${BOLD}sudo dnf install -y python3.12${RESET}  (Fedora)"
    else
        echo -e "Download from: ${BOLD}https://www.python.org/downloads/${RESET}"
    fi
    echo ""
    echo "Then re-run this installer."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${RESET} ($(command -v "$PYTHON_CMD" || echo "$PYTHON_CMD"))"

# -----------------------------------------------------------------------
# Step 2: Create isolated environment
# -----------------------------------------------------------------------

echo -e "${DIM}Setting up at $INSTALL_DIR ...${RESET}"

if [ -d "$INSTALL_DIR/venv" ]; then
    echo -e "${DIM}Upgrading existing installation...${RESET}"
    rm -rf "$INSTALL_DIR/venv"
fi

mkdir -p "$INSTALL_DIR/bin"
"$PYTHON_CMD" -m venv "$INSTALL_DIR/venv"

# -----------------------------------------------------------------------
# Step 3: Install llm-forge-new
# -----------------------------------------------------------------------

# Where to install from. Defaults to the published main branch, but can be
# pointed at a branch or a local clone -- useful for testing a fix before it is
# merged, which otherwise requires pushing to main to try it at all.
#
#   LLM_FORGE_REF=my-branch    curl ... | bash     # install a branch
#   LLM_FORGE_SOURCE=~/my/repo bash install.sh     # install a local clone
LLM_FORGE_REF="${LLM_FORGE_REF:-main}"

if [ -n "$LLM_FORGE_SOURCE" ]; then
    SOURCE_DESC="local clone at $LLM_FORGE_SOURCE"
    PIP_TARGET="$LLM_FORGE_SOURCE[chat]"
else
    SOURCE_DESC="github @ $LLM_FORGE_REF"
    PIP_TARGET="llm-forge-new[chat] @ git+https://github.com/Nagavenkatasai7/llm-forge.git@$LLM_FORGE_REF"
fi

echo -e "${DIM}Installing LLM Forge from $SOURCE_DESC ...${RESET}"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q 2>/dev/null
"$INSTALL_DIR/venv/bin/pip" install "$PIP_TARGET" -q

if ! "$INSTALL_DIR/venv/bin/python" -c "import llm_forge" 2>/dev/null; then
    echo -e "${RED}Installation failed. Please report:${RESET}"
    echo "  https://github.com/Nagavenkatasai7/llm-forge/issues"
    exit 1
fi

VERSION=$("$INSTALL_DIR/venv/bin/python" -c "import llm_forge; print(llm_forge.__version__)")

# -----------------------------------------------------------------------
# Step 4: Create launcher and add to PATH
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# API key setup — LLM Forge ships no credentials of its own.
# -----------------------------------------------------------------------

ENV_FILE="$INSTALL_DIR/.env"

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${GREEN}Using ANTHROPIC_API_KEY from your environment.${RESET}"
elif [ -n "$OLLAMA_API_KEY" ]; then
    echo -e "${GREEN}Using OLLAMA_API_KEY from your environment.${RESET}"
elif [ -f "$ENV_FILE" ] && grep -qE "^(ANTHROPIC|OLLAMA|OPENAI)_API_KEY=" "$ENV_FILE" 2>/dev/null; then
    echo -e "${GREEN}Using the API key already saved at $ENV_FILE${RESET}"
elif [ -t 0 ]; then
    echo ""
    echo -e "${BOLD}Anthropic API key${RESET} ${DIM}(powers the conversational assistant)${RESET}"
    echo -e "${DIM}Create one at https://console.anthropic.com/settings/keys${RESET}"
    echo -e "${DIM}Press Enter to skip — the offline wizard works without a key.${RESET}"
    printf "  Paste key: "
    read -r USER_KEY
    if [ -n "$USER_KEY" ]; then
        mkdir -p "$INSTALL_DIR"
        touch "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        # Drop any previous value, then append the new one.
        grep -v "^ANTHROPIC_API_KEY=" "$ENV_FILE" > "$ENV_FILE.tmp" 2>/dev/null || true
        mv "$ENV_FILE.tmp" "$ENV_FILE"
        echo "ANTHROPIC_API_KEY=$USER_KEY" >> "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        echo -e "${GREEN}Saved to $ENV_FILE${RESET}"
    else
        echo -e "${DIM}Skipped. Run 'llm-forge setup' for the offline guided setup.${RESET}"
    fi
else
    # Piped install (curl | bash) has no stdin to prompt on.
    echo -e "${YELLOW}No API key configured.${RESET}"
    echo -e "${DIM}Set one later:  export ANTHROPIC_API_KEY=sk-ant-...${RESET}"
    echo -e "${DIM}Ollama Cloud also works:  export OLLAMA_API_KEY=...${RESET}"
    echo -e "${DIM}Or run 'llm-forge setup' for the offline guided setup.${RESET}"
fi

# Create launcher
cat > "$INSTALL_DIR/bin/llm-forge" << 'LAUNCHER'
#!/usr/bin/env bash
exec "$HOME/.llm-forge/venv/bin/llm-forge" "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/bin/llm-forge"

# Determine shell config — CREATE it if it doesn't exist
detect_shell_config() {
    local current_shell
    current_shell=$(basename "$SHELL" 2>/dev/null || echo "bash")

    case "$current_shell" in
        zsh)
            echo "$HOME/.zshrc"
            ;;
        bash)
            # Prefer .bashrc on Linux, .bash_profile on macOS
            if [[ "$OSTYPE" == "darwin"* ]]; then
                echo "$HOME/.bash_profile"
            else
                echo "$HOME/.bashrc"
            fi
            ;;
        fish)
            echo "$HOME/.config/fish/config.fish"
            ;;
        *)
            # Fallback: check what exists
            if [ -f "$HOME/.zshrc" ]; then
                echo "$HOME/.zshrc"
            elif [ -f "$HOME/.bashrc" ]; then
                echo "$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                echo "$HOME/.bash_profile"
            else
                # Create .bashrc as default
                echo "$HOME/.bashrc"
            fi
            ;;
    esac
}

SHELL_CONFIG=$(detect_shell_config)

# Add PATH + env sourcing to shell config
LLM_BLOCK='# LLM Forge
export PATH="$HOME/.llm-forge/bin:$PATH"'

if ! grep -q "llm-forge/bin" "$SHELL_CONFIG" 2>/dev/null; then
    echo "" >> "$SHELL_CONFIG"
    echo "$LLM_BLOCK" >> "$SHELL_CONFIG"
    echo -e "${DIM}Added to $SHELL_CONFIG${RESET}"
fi

# Also add to other common configs
for extra_config in "$HOME/.zprofile" "$HOME/.profile"; do
    if [ -f "$extra_config" ]; then
        if ! grep -q "llm-forge/bin" "$extra_config" 2>/dev/null; then
            echo "" >> "$extra_config"
            echo "$LLM_BLOCK" >> "$extra_config"
        fi
    fi
done

# Activate in current session
export PATH="$HOME/.llm-forge/bin:$PATH"
# API keys live in ~/.llm-forge/.env and are read by llm_forge.chat.api_keys

# -----------------------------------------------------------------------
# Step 6: Verify it works
# -----------------------------------------------------------------------

if command -v llm-forge &>/dev/null; then
    VERIFY="verified"
else
    VERIFY="needs_source"
fi

# -----------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------

echo ""
echo -e "${BOLD}${GREEN}╭─────────────────────────────────────────╮${RESET}"
echo -e "${BOLD}${GREEN}│   LLM Forge v${VERSION} installed!            │${RESET}"
echo -e "${BOLD}${GREEN}╰─────────────────────────────────────────╯${RESET}"
echo ""

if [ "$VERIFY" = "verified" ]; then
    echo -e "Ready to go! Just type:"
    echo ""
    echo -e "  ${BOLD}${CYAN}llm-forge${RESET}"
else
    echo -e "Almost done! Run this once to activate:"
    echo ""
    echo -e "  ${BOLD}source $SHELL_CONFIG${RESET}"
    echo ""
    echo -e "Then type:"
    echo ""
    echo -e "  ${BOLD}${CYAN}llm-forge${RESET}"
    echo ""
    echo -e "${DIM}(Or just open a new terminal — it activates automatically.)${RESET}"
fi

echo ""
if [ -f "$ENV_FILE" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OLLAMA_API_KEY" ]; then
    echo -e "${GREEN}API key configured — ready to use.${RESET}"
else
    echo -e "${DIM}Set an API key (Anthropic, Ollama, or OpenAI) to use the assistant,${RESET}"
    echo -e "${DIM}or run 'llm-forge setup' for the offline wizard.${RESET}"
fi
echo ""
echo -e "${DIM}Type 'llm-forge' to start building your AI model.${RESET}"
echo ""
