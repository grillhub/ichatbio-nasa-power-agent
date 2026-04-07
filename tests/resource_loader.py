"""Load JSON/text fixtures from ``tests/resources/`` for tests."""

from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent


def resource(filename: str) -> str:
    return (_TESTS_DIR / "resources" / filename).read_text(encoding="utf-8")
