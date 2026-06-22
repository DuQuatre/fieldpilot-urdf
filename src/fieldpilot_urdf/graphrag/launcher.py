"""Console-script entry point for the GraphRAG server (``fieldpilot-urdf-server``).

Thin wrapper that boots uvicorn on the FastAPI app. Gated behind the ``[server]``
extra; prints a helpful hint if uvicorn is missing.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fieldpilot-urdf-server", description="Run the fieldpilot-urdf GraphRAG HTTP API"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install the server extras: "
            "pip install 'fieldpilot-urdf[server]'",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "fieldpilot_urdf.graphrag.server:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
