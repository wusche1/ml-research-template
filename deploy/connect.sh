#!/bin/bash
set -euo pipefail

INSTANCE_NAME="research-gpu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching SkyPilot cluster ${INSTANCE_NAME}..."
sky launch -c "$INSTANCE_NAME" "$SCRIPT_DIR/sky.yaml" || true

GITHUB_USER=$(gh api user --jq .login)
echo "Syncing dotfiles for ${GITHUB_USER}..."

DOTFILES_SETUP=$(cat <<'EOF'
git clone https://github.com/$GITHUB_USER/dotfiles.git ~/dotfiles 2>/dev/null || true
if [ -f ~/dotfiles/setup.sh ]; then
    cd ~/dotfiles
    ./setup.sh
fi
EOF
)

if ! sky exec "$INSTANCE_NAME" "GITHUB_USER='${GITHUB_USER}' bash -c '$DOTFILES_SETUP'"; then
    echo "Dotfiles setup failed; continuing without customization."
fi

echo "Opening remote session in Cursor..."
cursor --remote "ssh-remote+${INSTANCE_NAME}" /root/sky_workdir

