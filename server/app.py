# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the OpenClean Environment.

Endpoints:
    - POST /reset
    - POST /step
    - GET /state
    - GET /schema
    - WS /ws
"""

# -------- IMPORTS --------
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install dependencies using pip.") from e

# Handle both local + docker execution
try:
    from ..models import KernelAction, KernelObservation
    from server.environment import KernelEnvironment
except ImportError:
    from models import KernelAction, KernelObservation
    from server.environment import KernelEnvironment


# -------- CREATE APP --------
app = create_app(
    KernelEnvironment,
    KernelAction,
    KernelObservation,
    env_name="opencleanenv",
    max_concurrent_envs=1,
)


# -------- MAIN ENTRYPOINT --------
def main() -> None:
    """Run the OpenEnv server.

    Supports:
        python -m server.app
        uv run server
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


# -------- CLI SUPPORT --------
if __name__ == "__main__":
    main()