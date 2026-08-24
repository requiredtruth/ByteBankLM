from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from .planner import Job, plan_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Admit concurrent local-LLM jobs using byte-exact inputs.")
    parser.add_argument("spec", help="JSON planning specification")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        jobs = [Job(**item) for item in data["jobs"]]
        plan = plan_jobs(data["ram_bytes"], data.get("reserve_bytes", 0), jobs)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"bytebanklm: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 1 if args.fail_on_reject and any(not item.accepted for item in plan.decisions) else 0
