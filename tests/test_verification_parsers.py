from kinetic_sdk.verify.parsers import parse_mypy, parse_pytest, parse_ruff


def test_parsers_return_stable_failure_summaries():
    pytest_summary = parse_pytest("FAILED tests/test_a.py::test_x - AssertionError: nope\n1 failed")
    ruff_summary = parse_ruff('[{"code":"F401","filename":"a.py","location":{"row":2,"column":1},"message":"unused"}]')
    mypy_summary = parse_mypy("a.py:3: error: bad type  [arg-type]\nFound 1 error")
    assert pytest_summary.failures[0].id == "tests/test_a.py::test_x"
    assert ruff_summary.failures[0].id == "F401"
    assert mypy_summary.failures[0].id == "arg-type"
    assert pytest_summary.signature


def test_parsers_never_crash_on_unrecognised_output():
    result = parse_ruff("not json at all")
    assert result.unparsed
    assert result.raw == "not json at all"
