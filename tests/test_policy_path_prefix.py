"""Regression coverage for normalised path-prefix policy matching."""

import pytest

from kinetic_sdk.security.policy import _path_matches_prefix


@pytest.mark.parametrize(
    ("path", "prefix", "expected"),
    [
        ("/etc/passwd", "/", True),
        ("/workspace/project/a.py", "/workspace", True),
        ("/workspace", "/workspace/", True),
        ("//workspace/project/a.py", "/workspace", True),
        ("a/b.py", ".", True),
        ("./a/b.py", ".", True),
        ("../a/b.py", ".", False),
        ("../../a/b.py", ".", False),
        ("/workspace-evil/a.py", "/workspace", False),
        ("/workspace/../etc/passwd", "/workspace", False),
        ("/etcetera/notes.txt", "/etc", False),
        ("a/b.py", "/", False),
        ("/a/b.py", ".", False),
        ("a/b.py", "/workspace", False),
        ("/workspace/a.py", "workspace", False),
    ],
)
def test_path_matches_prefix_normalised_boundaries(
    path: str, prefix: str, expected: bool
) -> None:
    assert _path_matches_prefix(path, prefix) is expected
