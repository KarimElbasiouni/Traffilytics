#!/usr/bin/env bash
# Download the published weights and the finished demo clip when they are not
# already on disk. Safe to re-run: an existing database or weights file is left
# alone.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

weights_url="https://github.com/KarimElbasiouni/Traffilytics/releases/download/obb-v1/your_obb.pt"
demo_url="https://github.com/KarimElbasiouni/Traffilytics/releases/download/obb-v1/demo-test_video2.tar.gz"

mkdir -p models data

if [[ ! -f models/your_obb.pt ]]; then
  echo "Fetching weights"
  curl -fL --retry 3 -o models/your_obb.pt "$weights_url"
fi

if [[ ! -f data/traffilytics.db ]]; then
  echo "Fetching demo clip"
  tmp="$(mktemp)"
  curl -fL --retry 3 -o "$tmp" "$demo_url"
  tar -xzf "$tmp" -C "$root"
  rm -f "$tmp"
fi
