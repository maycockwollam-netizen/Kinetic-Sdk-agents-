def test_rust_patch_engine_round_trips_hunks_and_preserves_crlf(tmp_path):
    from kinetic_sdk.files.patch import ApplyPatchTool, _FilePatch, _Hunk

    original = "first\r\nsecond\r\n"
    file_patch = _FilePatch(
        "sample.txt",
        (_Hunk(2, 1, 2, 1, ("-second\n", "+changed\n")),),
    )

    assert ApplyPatchTool._apply_file_patch(original, file_patch) == "first\r\nchanged\r\n"


def test_rust_patch_engine_supports_zero_count_insertion():
    from kinetic_sdk.files.patch import ApplyPatchTool

    parsed = ApplyPatchTool._parse_patch(
        "--- a/file.txt\n+++ b/file.txt\n@@ -0,0 +1 @@\n+first\n"
    )
    assert ApplyPatchTool._apply_file_patch("", parsed[0]) == "first\n"
