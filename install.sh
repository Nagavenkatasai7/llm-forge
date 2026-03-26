#!/usr/bin/env bash
# LLM Forge Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash
#
# This script:
# 1. Finds Python 3.10+ on your system
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

INSTALL_DIR="$HOME/.llm-forge"

echo ""
echo -e "${BOLD}${CYAN}╭─────────────────────────────────────╮${RESET}"
echo -e "${BOLD}${CYAN}│   LLM Forge — Installer             │${RESET}"
echo -e "${BOLD}${CYAN}│   Build your own AI model            │${RESET}"
echo -e "${BOLD}${CYAN}╰─────────────────────────────────────╯${RESET}"
echo ""

# -----------------------------------------------------------------------
# Step 1: Find Python 3.10+
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
            if [ "$major" = "3" ] && [ "$minor" -ge 10 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo -e "${DIM}Searching for Python 3.10+...${RESET}"

PYTHON_CMD=$(find_python) || true

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Python 3.10+ not found on your system.${RESET}"
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

echo -e "${DIM}Installing LLM Forge v3.0.0 (multi-agent orchestration)...${RESET}"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q 2>/dev/null
"$INSTALL_DIR/venv/bin/pip" install "llm-forge-new[chat] @ git+https://github.com/Nagavenkatasai7/llm-forge.git@main" -q

if ! "$INSTALL_DIR/venv/bin/python" -c "import llm_forge" 2>/dev/null; then
    echo -e "${RED}Installation failed. Please report:${RESET}"
    echo "  https://github.com/Nagavenkatasai7/llm-forge/issues"
    exit 1
fi

VERSION=$("$INSTALL_DIR/venv/bin/python" -c "import llm_forge; print(llm_forge.__version__)")

# -----------------------------------------------------------------------
# Step 4: Create launcher and add to PATH
# -----------------------------------------------------------------------

# Create launcher script in bin/
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
PATH_LINE='export PATH="$HOME/.llm-forge/bin:$PATH"'

# Add to shell config (create file if needed)
if ! grep -q "llm-forge/bin" "$SHELL_CONFIG" 2>/dev/null; then
    echo "" >> "$SHELL_CONFIG"
    echo "# LLM Forge" >> "$SHELL_CONFIG"
    echo "$PATH_LINE" >> "$SHELL_CONFIG"
    echo -e "${DIM}Added to $SHELL_CONFIG${RESET}"
fi

# Also try to add to other common configs that might be sourced
for extra_config in "$HOME/.zprofile" "$HOME/.profile"; do
    if [ -f "$extra_config" ]; then
        if ! grep -q "llm-forge/bin" "$extra_config" 2>/dev/null; then
            echo "" >> "$extra_config"
            echo "# LLM Forge" >> "$extra_config"
            echo "$PATH_LINE" >> "$extra_config"
        fi
    fi
done

# Activate in current session
export PATH="$HOME/.llm-forge/bin:$PATH"

# -----------------------------------------------------------------------
# Step 5: Verify it works
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
echo -e "${BOLD}API Keys Required:${RESET}"
echo -e "  ${CYAN}export ANTHROPIC_API_KEY=sk-ant-...${RESET}  (get at console.anthropic.com)"
echo -e "  ${CYAN}export GOOGLE_API_KEY=AIza...${RESET}       (get at aistudio.google.com/apikey)"
echo ""
echo -e "${DIM}Then type 'llm-forge' to start building your AI model.${RESET}"
echo ""
