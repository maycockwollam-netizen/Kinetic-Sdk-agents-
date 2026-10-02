import pytest
from app import clamp

def test_edges_and_invalid_range():
    assert clamp(-1, 0, 3) == 0
    assert clamp(9, 0, 3) == 3
    with pytest.raises(ValueError):
        clamp(1, 3, 0)
