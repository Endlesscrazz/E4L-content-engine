"""Shared pytest configuration — custom options and fixtures."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live pipeline browser tests (~$0.02 in API cost)",
    )
