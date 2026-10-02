# Roadmap progress

## 2026-10-02 — Nền tảng verified coding loop

- Đã thêm manifest nghiêm ngặt tại `kinetic_sdk/project/manifest.py`, hợp đồng
  trace-derived và parser kết quả công cụ tại `kinetic_sdk/verify/`.
- Agent sync/async có cờ opt-in `require_verification`, phát
  `verification.completed` và giữ `last_verification`; event tool nay mang
  arguments, output và metadata để có bằng chứng kiểm chứng.
- Lệnh dự kiến chạy: `python -m pytest -q tests/test_project_manifest.py
  tests/test_verification.py tests/test_verification_parsers.py`, `ruff check
  kinetic_sdk tests`, `mypy`, rồi toàn suite.
- Vấn đề còn mở: cần kiểm tra tương thích Python 3.10 cho TOML loader (bản đầu
  tiên hiện dựa vào `tomllib` của Python 3.11+); nên thay bằng parser stdlib
  tương thích trước khi phát hành 3.10.
