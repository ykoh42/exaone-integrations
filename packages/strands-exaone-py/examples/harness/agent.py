"""Use EXAONE with the Strands Harness read tool and optional Exa search."""

from __future__ import annotations

import argparse
import os
from typing import cast

from strands import Agent
from strands.models import Model
from strands_harness import BuiltinToolsConfig, create_harness

from strands_exaone import ExaoneModel, ExaoneRetryStrategy


def make_agent(*, model: Model | None = None, allow_file_read: bool = False, web_search: bool = False) -> Agent:
    """Create a harness with explicitly enabled file reading or optional Exa search."""
    builtin_tools = cast(BuiltinToolsConfig, {"*": False})
    if allow_file_read:
        builtin_tools["read"] = {"media": False}
    if web_search:
        builtin_tools["web_search"] = "exa"
    if model is None:
        base_url = os.environ.get("EXAONE_BASE_URL")
        model = ExaoneModel(base_url=base_url) if base_url is not None else ExaoneModel()
    return create_harness(
        model=model,
        retry_strategy=ExaoneRetryStrategy(),
        builtin_tools=builtin_tools,
        builtin_plugins=[],
        session=False,
        memory=False,
        skills=False,
        caching=False,
        callback_handler=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Task for the agent")
    parser.add_argument(
        "--allow-file-read",
        action="store_true",
        help="Enable file reading; the read tool can access files permitted by the agent sandbox",
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable Exa search; search queries are sent to the third-party Exa service",
    )
    args = parser.parse_args()
    agent = make_agent(allow_file_read=args.allow_file_read, web_search=args.web_search)
    print(agent(args.prompt))


if __name__ == "__main__":
    main()
