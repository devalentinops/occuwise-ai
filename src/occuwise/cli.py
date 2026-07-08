"""Thin console entrypoint (`occuwise <command>`).

Delegates to the module entrypoints so both `occuwise train ...` and
`py -m occuwise.train ...` work. Hydra-based commands receive their args verbatim.
"""

from __future__ import annotations

import runpy
import sys

COMMANDS = {"train", "evaluate", "compare", "export", "predict"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: occuwise <{'|'.join(sorted(COMMANDS))}> [args...]")
        raise SystemExit(1)
    cmd = sys.argv[1]
    sys.argv = [f"occuwise.{cmd}", *sys.argv[2:]]
    runpy.run_module(f"occuwise.{cmd}", run_name="__main__")


if __name__ == "__main__":
    main()
