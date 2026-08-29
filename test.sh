#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "Run ./install.sh first." >&2
    exit 1
fi

cd "$ROOT"
"$PY" -m compileall -q bytebanklm tests project_gui.py
"$PY" -m unittest discover -s tests -p 'test_*.py' -v

plan_output="$("$ROOT/cli.sh" examples/plan.json)"
"$PY" -c '
import json
import sys

payload = json.load(sys.stdin)
assert payload["usable_bytes"] > payload["admitted_bytes"] > 0, payload
assert [item["name"] for item in payload["decisions"]] == ["chat", "coder"], payload
assert all(item["accepted"] for item in payload["decisions"]), payload
' <<<"$plan_output"

echo "ByteBankLM tests and example plan passed."
