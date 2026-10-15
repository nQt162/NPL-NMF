"""Start the backend without Click's command-line wrapper or auto-reload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=8001)
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent)
    print("Loading backend and NLP libraries; please wait for 'Application startup complete'.", flush=True)

    try:
        import uvicorn
    except ImportError:
        print('Uvicorn is missing. Install backend dependencies with: python -m pip install -e ".[dev]"', file=sys.stderr)
        return 1

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        reload=False,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
