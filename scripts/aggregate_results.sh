#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || "$1" != "--output" ]]; then
  echo "Usage: $0 --output SUMMARY.json METRICS.json [METRICS.json ...]" >&2
  exit 2
fi
OUTPUT="$2"
shift 2
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m memuaq aggregate --output "$OUTPUT" --metrics "$@"
