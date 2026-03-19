#!/bin/bash
# Safety check hook — blocks dangerous operations in Claude Code

INPUT="$TOOL_INPUT_command"

# Block force pushes to main (use --force-with-lease on feature branches only)
if echo "$INPUT" | grep -q "git push.*--force"; then
  echo "BLOCKED: Force push detected."
  echo "Use --force-with-lease on feature branches only."
  echo "Never force push to main."
  exit 1
fi

# Block recursive deletion of critical directories
if echo "$INPUT" | grep -qE "rm.*-rf.*(src|tests|configs|\.claude)"; then
  echo "BLOCKED: Recursive deletion of critical directory detected."
  echo "Directories src/, tests/, configs/, and .claude/ are protected."
  exit 1
fi

# Block accidental deletion of the database
if echo "$INPUT" | grep -qE "rm.*metrics\.db"; then
  echo "BLOCKED: Deletion of metrics database detected."
  echo "The metrics database contains historical training data."
  exit 1
fi

# Warn about pip install without --break-system-packages
if echo "$INPUT" | grep -q "pip install" && ! echo "$INPUT" | grep -q "break-system-packages"; then
  echo "WARNING: Consider using --break-system-packages flag with pip install."
  echo "Continuing anyway..."
fi

# All checks passed
exit 0
