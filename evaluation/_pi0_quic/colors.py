"""ANSI color helpers for the eval CLI."""

from __future__ import annotations

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"


def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{RESET}"
