#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [additional memuaq arguments]" >&2
  exit 2
fi
CONFIG="$1"
shift
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m memuaq run --config "$CONFIG" "$@"
