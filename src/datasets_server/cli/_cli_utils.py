# Copyright 2024 Daniel van Strien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI utilities and shared type annotations for datasets-server CLI.

Follows patterns from huggingface_hub CLI for consistency.
"""

from enum import Enum
from typing import Annotated, Optional

import typer


class OutputFormat(str, Enum):
    """Output format options."""

    json = "json"
    table = "table"


def typer_factory(**kwargs: object) -> typer.Typer:
    """Create a Typer app with consistent settings for agent-friendly output.

    Disables Rich markup and pretty exceptions to ensure clean, parseable output
    that works well with agents and scripting.
    """
    return typer.Typer(
        rich_markup_mode=None,
        pretty_exceptions_enable=False,
        **kwargs,
    )


# Reusable type annotations for CLI options and arguments
# Following huggingface_hub patterns

DatasetArg = Annotated[
    str,
    typer.Argument(
        help="The dataset ID (e.g., 'squad', 'username/dataset-name').",
    ),
]

ConfigOpt = Annotated[
    Optional[str],
    typer.Option(
        "--config",
        "-c",
        help="The dataset configuration name.",
    ),
]

SplitOpt = Annotated[
    Optional[str],
    typer.Option(
        "--split",
        "-s",
        help="The dataset split (e.g., 'train', 'test', 'validation').",
    ),
]

TokenOpt = Annotated[
    Optional[str],
    typer.Option(
        "--token",
        help="HuggingFace token for accessing private/gated datasets.",
        envvar="HF_TOKEN",
    ),
]

FormatOpt = Annotated[
    OutputFormat,
    typer.Option(
        "--format",
        "-f",
        help="Output format.",
    ),
]

LimitOpt = Annotated[
    int,
    typer.Option(
        "--limit",
        "-l",
        help="Maximum number of results to return.",
    ),
]

OffsetOpt = Annotated[
    int,
    typer.Option(
        "--offset",
        "-o",
        help="Number of results to skip.",
    ),
]
