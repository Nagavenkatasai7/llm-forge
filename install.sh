#!/usr/bin/env bash
# LLM Forge Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash
#
# This script:
# 1. Finds or installs Python 3.10+
# 2. Creates an isolated environment at ~/.llm-forge/
# 3. Installs llm-forge-new from PyPI
# 4. Adds 'llm-forge' to your PATH
# 5. Done — type 'llm-forge' anywhere

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

INSTALL_DIR="$HOME/.llm-forge"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

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
    # Check common Python locations in priority order
    local candidates=(
        "python3.13"
        "python3.12"
        "python3.11"
        "python3.10"
        "/opt/homebrew/bin/python3.13"
        "/opt/homebrew/bin/python3.12"
        "/opt/homebrew/bin/python3.11"
        "/opt/homebrew/bin/python3.10"
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11"
        "/usr/local/bin/python3.10"
        "/usr/local/bin/python3"
        "$HOME/.pyenv/shims/python3"
        "python3"
    )

    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major minor
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo -e "${DIM}Searching for Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+...${RESET}"

PYTHON_CMD=$(find_python) || true

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Python 3.10+ not found.${RESET}"
    echo ""

    # Detect OS and suggest install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install Python with Homebrew:"
        echo ""
        echo -e "  ${BOLD}brew install python@3.12${RESET}"
        echo ""
        echo "Don't have Homebrew? Install it first:"
        echo ""
        echo -e "  ${BOLD}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${RESET}"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "Install Python:"
        echo ""
        echo -e "  ${BOLD}sudo apt update && sudo apt install python3.12 python3.12-venv${RESET}  (Ubuntu/Debian)"
        echo -e "  ${BOLD}sudo dnf install python3.12${RESET}  (Fedora)"
    else
        echo "Install Python 3.10+ from: https://www.python.org/downloads/"
    fi
    echo ""
    echo "Then re-run this installer."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${RESET} ($PYTHON_CMD)"

# -----------------------------------------------------------------------
# Step 2: Create isolated environment
# -----------------------------------------------------------------------

echo -e "${DIM}Setting up LLM Forge at $INSTALL_DIR...${RESET}"

if [ -d "$INSTALL_DIR/venv" ]; then
    echo -e "${DIM}Existing installation found. Upgrading...${RESET}"
    rm -rf "$INSTALL_DIR/venv"
fi

mkdir -p "$INSTALL_DIR"
"$PYTHON_CMD" -m venv "$INSTALL_DIR/venv"

# -----------------------------------------------------------------------
# Step 3: Install llm-forge-new
# -----------------------------------------------------------------------

echo -e "${DIM}Installing LLM Forge from PyPI...${RESET}"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q 2>/dev/null
"$INSTALL_DIR/venv/bin/pip" install llm-forge-new -q

# Verify installation
if ! "$INSTALL_DIR/venv/bin/python" -c "import llm_forge" 2>/dev/null; then
    echo -e "${RED}Installation failed. Please report this issue:${RESET}"
    echo "  https://github.com/Nagavenkatasai7/llm-forge/issues"
    exit 1
fi

VERSION=$("$INSTALL_DIR/venv/bin/python" -c "import llm_forge; print(llm_forge.__version__)")

# -----------------------------------------------------------------------
# Step 4: Create launcher script in PATH
# -----------------------------------------------------------------------

# Create the launcher
cat > "$INSTALL_DIR/bin-llm-forge" << 'LAUNCHER'
#!/usr/bin/env bash
exec "$HOME/.llm-forge/venv/bin/llm-forge" "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/bin-llm-forge"

# Determine shell config file
SHELL_CONFIG=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
fi

# Add to PATH if not already there
PATH_LINE='export PATH="$HOME/.llm-forge:$PATH"'
NEEDS_PATH_UPDATE=true

# Create a symlink-friendly bin directory
mkdir -p "$INSTALL_DIR"
ln -sf "$INSTALL_DIR/bin-llm-forge" "$INSTALL_DIR/llm-forge"

if [ -n "$SHELL_CONFIG" ]; then
    if grep -q ".llm-forge" "$SHELL_CONFIG" 2>/dev/null; then
        NEEDS_PATH_UPDATE=false
    else
        echo "" >> "$SHELL_CONFIG"
        echo "# LLM Forge" >> "$SHELL_CONFIG"
        echo "$PATH_LINE" >> "$SHELL_CONFIG"
    fi
fi

# Also add to current session
export PATH="$HOME/.llm-forge:$PATH"

# -----------------------------------------------------------------------
# Step 5: Done!
# -----------------------------------------------------------------------

echo ""
echo -e "${BOLD}${GREEN}╭─────────────────────────────────────╮${RESET}"
echo -e "${BOLD}${GREEN}│   LLM Forge v${VERSION} installed!       │${RESET}"
echo -e "${BOLD}${GREEN}╰─────────────────────────────────────╯${RESET}"
echo ""

if [ "$NEEDS_PATH_UPDATE" = true ] && [ -n "$SHELL_CONFIG" ]; then
    echo -e "Run this to activate (one time only):"
    echo ""
    echo -e "  ${BOLD}source $SHELL_CONFIG${RESET}"
    echo ""
fi

echo -e "Then just type:"
echo ""
echo -e "  ${BOLD}${CYAN}llm-forge${RESET}"
echo ""
echo -e "${DIM}That's it. Go to any folder and type 'llm-forge' to start building your AI.${RESET}"
echo ""
