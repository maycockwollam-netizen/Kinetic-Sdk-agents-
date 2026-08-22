"""Tests for kinetic_sdk.mcp.protocol (JSON-RPC 2.0 message layer)."""

from __future__ import annotations

import threading

import pytest

from kinetic_sdk.mcp.protocol import (
    JSONRPC_VERSION,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPProtocolError,
    RequestIdGenerator,
    decode,
    encode,
)


class TestEncodeDecodeRoundTrip:
    def test_request_round_trip(self) -> None:
        msg = JsonRpcRequest(id=7, method="tools/call", params={"name": "search"})
        line = encode(msg)
        assert line.endswith("\n")
        assert line.count("\n") == 1  # exactly one line on the wire
        decoded = decode(line)
        assert isinstance(decoded, JsonRpcRequest)
        assert decoded.id == 7
        assert decoded.method == "tools/call"
        assert decoded.params == {"name": "search"}

    def test_request_without_params_omits_key(self) -> None:
        line = encode(JsonRpcRequest(id=1, method="ping"))
        assert '"params"' not in line
        decoded = decode(line)
        assert isinstance(decoded, JsonRpcRequest)
        assert decoded.params is None

    def test_notification_round_trip(self) -> None:
        msg = JsonRpcNotification(method="notifications/initialized")
        decoded = decode(encode(msg))
        assert isinstance(decoded, JsonRpcNotification)
        assert decoded.method == "notifications/initialized"
        assert decoded.params is None

    def test_result_response_round_trip(self) -> None:
        msg = JsonRpcResponse(id="abc", result={"tools": []})
        decoded = decode(encode(msg))
        assert isinstance(decoded, JsonRpcResponse)
        assert decoded.id == "abc"
        assert decoded.result == {"tools": []}
        assert not decoded.is_error

    def test_error_response_round_trip(self) -> None:
        msg = JsonRpcResponse.make_error(3, code=-32601, message="Method not found")
        decoded = decode(encode(msg))
        assert isinstance(decoded, JsonRpcResponse)
        assert decoded.is_error
        assert decoded.error == {"code": -32601, "message": "Method not found"}

    def test_response_with_null_result_is_valid(self) -> None:
        decoded = decode('{"jsonrpc": "2.0", "id": 1, "result": null}')
        assert isinstance(decoded, JsonRpcResponse)
        assert decoded.result is None
        assert not decoded.is_error

    def test_bytes_input_decoded_as_utf8(self) -> None:
        decoded = decode(b'{"jsonrpc": "2.0", "method": "ping", "id": 1}\n')
        assert isinstance(decoded, JsonRpcRequest)

    def test_non_ascii_payload_preserved(self) -> None:
        msg = JsonRpcNotification(method="log", params={"text": "Xin chào"})
        decoded = decode(encode(msg))
        assert isinstance(decoded, JsonRpcNotification)
        assert decoded.params == {"text": "Xin chào"}


class TestDecodeErrors:
    @pytest.mark.parametrize(
        "line",
        [
            "not json at all",
            '{"jsonrpc": "2.0", "method": "x"',  # truncated
            "[1, 2, 3]",  # not an object
            "   ",  # empty
        ],
    )
    def test_malformed_json_raises(self, line: str) -> None:
        with pytest.raises(MCPProtocolError):
            decode(line)

    def test_wrong_jsonrpc_version_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="jsonrpc"):
            decode('{"jsonrpc": "1.0", "id": 1, "method": "ping"}')

    def test_missing_jsonrpc_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="jsonrpc"):
            decode('{"id": 1, "method": "ping"}')

    def test_request_missing_method_and_id_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="method"):
            decode('{"jsonrpc": "2.0", "params": {}}')

    def test_response_with_both_result_and_error_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="exactly one"):
            decode('{"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": 1, "message": "x"}}')

    def test_response_with_neither_result_nor_error_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="exactly one"):
            decode('{"jsonrpc": "2.0", "id": 1}')

    def test_error_object_missing_code_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="code"):
            decode('{"jsonrpc": "2.0", "id": 1, "error": {"message": "x"}}')

    def test_bad_id_type_raises(self) -> None:
        with pytest.raises(MCPProtocolError, match="'id'"):
            decode('{"jsonrpc": "2.0", "id": [1], "method": "ping"}')

    def test_bool_id_rejected(self) -> None:
        with pytest.raises(MCPProtocolError, match="'id'"):
            decode('{"jsonrpc": "2.0", "id": true, "method": "ping"}')

    def test_invalid_utf8_bytes_raise(self) -> None:
        with pytest.raises(MCPProtocolError, match="UTF-8"):
            decode(b"\xff\xfe{")

    def test_response_cannot_hold_result_and_error(self) -> None:
        with pytest.raises(MCPProtocolError):
            JsonRpcResponse(id=1, result={}, error={"code": 1, "message": "x"})


class TestRequestIdGenerator:
    def test_ids_increase_monotonically(self) -> None:
        gen = RequestIdGenerator()
        ids = [gen.next() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_custom_start(self) -> None:
        gen = RequestIdGenerator(start=42)
        assert gen.next() == 42
        assert gen.next() == 43

    def test_thread_safe_no_duplicates(self) -> None:
        gen = RequestIdGenerator()
        collected: list[int] = []

        def worker() -> None:
            for _ in range(200):
                collected.append(gen.next())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(collected) == 800
        assert len(set(collected)) == 800  # no duplicates across threads
