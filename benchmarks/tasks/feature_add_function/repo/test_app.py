from app import clamp

def test_inside_range():
    assert clamp(2, 0, 3) == 2
