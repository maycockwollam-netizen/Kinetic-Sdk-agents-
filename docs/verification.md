# Verified coding loop

Kinetic chỉ xác nhận kết quả từ bằng chứng trong `RunTrace`, không tin văn bản
cuối cùng của agent. Tính năng hoàn toàn opt-in: truyền `require_verification=True`
vào `Agent` hoặc `AsyncAgent`. Kết quả nằm ở `agent.last_verification`; event
`verification.completed` phát status. Khi cờ này bật, câu trả lời cuối được thêm
`UNVERIFIED:` nếu không đạt hợp đồng. Kiểu trả về của `run()` không thay đổi.

## Manifest

Tạo `.kinetic/project.toml`; SDK không đoán lệnh khi file không tồn tại.

```toml
setup = "python -m pip install -e ."
test = "python -m pytest -q"
test_targeted = "python -m pytest -q {files}"
lint = "ruff check kinetic_sdk tests"
typecheck = "mypy"
protected_paths = [".github/", "pyproject.toml"]
command_timeout = 120
max_diff_lines = 500

[verification]
kind = "tests" # tests | lint | diff | custom
# command = "python -m compileall kinetic_sdk" # required for custom
```

Mọi key đều được kiểm tra nghiêm; key lạ là lỗi. `test_targeted` là template
khai báo, không phải lệnh SDK tự bịa. `custom` dùng `verification.command`.

## Quy tắc status

`VERIFIED` chỉ xuất hiện khi trace có **đúng lệnh** kiểm chứng khai trong
manifest, exit code bằng 0, và thời điểm chạy sau lần chỉnh sửa file cuối.
`FAILED` là lệnh kiểm chứng có exit code khác 0. `UNVERIFIED` là thiếu bằng
chứng, sửa sau lần chạy xanh, hoặc báo cáo JSON của agent không khớp trace.
`NOT_APPLICABLE` biểu thị không manifest (hoặc diff không có giới hạn).

`kind = "lint"` kiểm chứng bằng `lint`; `kind = "diff"` giới hạn số dòng của
output `git diff`; `kind = "custom"` so với `verification.command`. Parser
pytest, ruff và mypy tạo failure summary có signature ổn định, nhưng parser
không quyết định status: exit code và trace mới là bằng chứng gốc.
