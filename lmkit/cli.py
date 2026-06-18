"""lmkit command-line entry point."""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else ""
    if cmd == "quickstart":
        from . import quickstart
        return quickstart.main()
    print("usage: lmkit quickstart    # 60-second CPU demo of the pretrain loop")
    return 0 if cmd in ("", "-h", "--help", "help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
