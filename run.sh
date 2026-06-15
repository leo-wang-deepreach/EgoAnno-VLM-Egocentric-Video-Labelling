#!/usr/bin/env bash
# facts-first pipeline runner — uses the parent egoanno venv (PIL + stdlib).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python          # -> ../.venv (egoanno), has all deps
exec "$PY" run.py "$@"
