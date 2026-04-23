#!/bin/bash
# Update the app repo to a chosen branch and optionally restart containers.

set -euo pipefail

DEFAULT_REMOTE="origin"
DEFAULT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
REMOTE="${DEFAULT_REMOTE}"
BRANCH="${DEFAULT_BRANCH}"
RESTART_APP="true"

usage() {
    cat <<EOF
Usage: ./update_app.sh [branch] [options]

Examples:
  ./update_app.sh
  ./update_app.sh adding-models
  ./update_app.sh adding-models --no-restart
  ./update_app.sh --branch adding-models --remote origin

Options:
  -b, --branch <name>     Branch to update to. Defaults to current branch.
  -r, --remote <name>     Git remote to use. Defaults to origin.
      --no-restart        Only update git state; do not restart Docker services.
  -h, --help              Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        --no-restart)
            RESTART_APP="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "❌ Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            BRANCH="$1"
            shift
            ;;
    esac
done

echo "📦 Updating ChemSpaceCopilot"
echo "   Remote : $REMOTE"
echo "   Branch : $BRANCH"
echo ""

if ! command -v git >/dev/null 2>&1; then
    echo "❌ git is required but not installed."
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "❌ This script must be run inside the ChemSpaceCopilot git repository."
    exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
DIRTY_STATUS="$(git status --porcelain)"

if [[ -n "$DIRTY_STATUS" ]]; then
    echo "❌ Working tree has uncommitted changes."
    echo "Commit or stash them before running update_app.sh."
    exit 1
fi

echo "🔄 Fetching latest refs from $REMOTE..."
git fetch "$REMOTE"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
        echo "🌿 Switching local branch to $BRANCH"
        git checkout "$BRANCH"
    fi
else
    echo "🌿 Creating local branch $BRANCH from $REMOTE/$BRANCH"
    git checkout -b "$BRANCH" "$REMOTE/$BRANCH"
fi

echo "⬇️ Pulling latest commits for $BRANCH"
git pull "$REMOTE" "$BRANCH"

NEW_HEAD="$(git rev-parse --short HEAD)"
echo ""
echo "✅ Repo updated to $BRANCH @ $NEW_HEAD"

if [[ "$RESTART_APP" != "true" ]]; then
    echo "ℹ️ Skipping app restart (--no-restart)."
    exit 0
fi

COMPOSE_CMD=""
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
fi

if [[ -z "$COMPOSE_CMD" ]]; then
    echo "⚠️ Docker Compose not found. Repo update succeeded, but app restart was skipped."
    exit 0
fi

echo ""
echo "🐳 Restarting containers with $COMPOSE_CMD"
$COMPOSE_CMD down --remove-orphans
$COMPOSE_CMD up -d --build

echo ""
echo "🚀 App update complete."
echo "   Branch : $BRANCH"
echo "   Commit : $NEW_HEAD"
