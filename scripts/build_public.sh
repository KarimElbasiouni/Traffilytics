#!/usr/bin/env bash
# Build the one-process public site: Python package, dashboard, weights, demo clip.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python -m pip install -e ".[api]"

if ! command -v npm >/dev/null 2>&1; then
  node_version="v22.13.1"
  node_dir="/tmp/node-${node_version}-linux-x64"
  if [[ ! -x "${node_dir}/bin/npm" ]]; then
    curl -fL "https://nodejs.org/dist/${node_version}/node-${node_version}-linux-x64.tar.gz" \
      | tar -xz -C /tmp
  fi
  export PATH="${node_dir}/bin:${PATH}"
fi

(cd frontend && npm ci && npm run build)
bash scripts/fetch_public_assets.sh
