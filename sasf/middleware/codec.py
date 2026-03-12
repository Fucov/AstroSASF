"""
AstroSASF · Middleware · SpaceMCPCodec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
静态字典压缩引擎 —— JSON MCP 工具调用 → 紧凑二进制帧。
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_FRAME_MAGIC: int = 0xA5

_STR_TO_TOKEN: dict[str, int] = {
    "set_temperature": 0x01, "move_robotic_arm": 0x02, "toggle_vacuum_pump": 0x03,
    "target": 0x10, "target_angle": 0x11, "activate": 0x12,
    "skill": 0x20, "status": 0x21, "detail": 0x22, "fsm_state": 0x23,
    "success": 0x30, "error": 0x31,
}
_TOKEN_TO_STR: dict[int, str] = {v: k for k, v in _STR_TO_TOKEN.items()}

_TYPE_FLOAT: int = 0x01
_TYPE_INT:   int = 0x02
_TYPE_BOOL:  int = 0x03
_TYPE_STR:   int = 0x04
_TYPE_RAW:   int = 0x05


@dataclass
class SpaceMCPCodec:
    """静态字典二进制编解码器。"""

    lab_id: str
    _encode_count: int = field(default=0, init=False)
    _decode_count: int = field(default=0, init=False)
    _total_json_bytes: int = field(default=0, init=False)
    _total_binary_bytes: int = field(default=0, init=False)

    def encode(self, request: dict[str, Any]) -> bytearray:
        buf = bytearray()
        buf.append(_FRAME_MAGIC)

        skill_name: str = request.get("skill", "")
        skill_token = _STR_TO_TOKEN.get(skill_name)
        if skill_token is None:
            raise ValueError(f"[{self.lab_id}] 未知 Skill 无法编码: {skill_name}")
        buf.append(skill_token)

        params: dict[str, Any] = request.get("params", {})
        buf.append(len(params))
        for key, value in params.items():
            key_token = _STR_TO_TOKEN.get(key)
            if key_token is None:
                raise ValueError(f"[{self.lab_id}] 未知参数键无法编码: {key}")
            buf.append(key_token)
            self._encode_value(buf, value)

        self._encode_count += 1
        json_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        self._total_json_bytes += json_bytes
        self._total_binary_bytes += len(buf)
        return buf

    def encode_response(self, response: dict[str, Any]) -> bytearray:
        buf = bytearray()
        buf.append(_FRAME_MAGIC)
        buf.append(len(response))
        for key, value in response.items():
            key_token = _STR_TO_TOKEN.get(key)
            if key_token is not None:
                buf.append(key_token)
            else:
                buf.append(0xFF)
                raw = key.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)
            self._encode_value(buf, value)
        return buf

    def decode(self, data: bytearray | bytes) -> dict[str, Any]:
        offset = 0
        if data[offset] != _FRAME_MAGIC:
            raise ValueError(f"[{self.lab_id}] 无效帧: Magic 字节错误")
        offset += 1
        skill_token = data[offset]; offset += 1
        skill_name = _TOKEN_TO_STR.get(skill_token, f"unknown_0x{skill_token:02X}")
        n_params = data[offset]; offset += 1
        params: dict[str, Any] = {}
        for _ in range(n_params):
            key_token = data[offset]; offset += 1
            key_name = _TOKEN_TO_STR.get(key_token, f"unknown_0x{key_token:02X}")
            value, offset = self._decode_value(data, offset)
            params[key_name] = value
        self._decode_count += 1
        return {"skill": skill_name, "params": params}

    def decode_response(self, data: bytearray | bytes) -> dict[str, Any]:
        offset = 0
        if data[offset] != _FRAME_MAGIC:
            raise ValueError(f"[{self.lab_id}] 无效响应帧: Magic 字节错误")
        offset += 1
        n_fields = data[offset]; offset += 1
        result: dict[str, Any] = {}
        for _ in range(n_fields):
            key_byte = data[offset]; offset += 1
            if key_byte == 0xFF:
                str_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
                key_name = data[offset:offset + str_len].decode("utf-8"); offset += str_len
            else:
                key_name = _TOKEN_TO_STR.get(key_byte, f"unknown_0x{key_byte:02X}")
            value, offset = self._decode_value(data, offset)
            result[key_name] = value
        return result

    @staticmethod
    def calculate_compression_ratio(json_size: int, binary_size: int) -> float:
        if json_size == 0:
            return 0.0
        return (1.0 - binary_size / json_size) * 100.0

    @property
    def stats(self) -> dict[str, Any]:
        ratio = self.calculate_compression_ratio(self._total_json_bytes, self._total_binary_bytes)
        return {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "total_json_bytes": self._total_json_bytes,
            "total_binary_bytes": self._total_binary_bytes,
            "overall_compression_ratio": f"{ratio:.1f}%",
        }

    @staticmethod
    def _encode_value(buf: bytearray, value: Any) -> None:
        if isinstance(value, bool):
            buf.append(_TYPE_BOOL); buf.append(0x01 if value else 0x00)
        elif isinstance(value, float):
            buf.append(_TYPE_FLOAT); buf.extend(struct.pack(">f", value))
        elif isinstance(value, int):
            buf.append(_TYPE_INT); buf.extend(struct.pack(">b", max(-128, min(127, value))))
        elif isinstance(value, str):
            token = _STR_TO_TOKEN.get(value)
            if token is not None:
                buf.append(_TYPE_STR); buf.append(token)
            else:
                buf.append(_TYPE_RAW); raw = value.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw))); buf.extend(raw)
        else:
            buf.append(_TYPE_RAW); raw = str(value).encode("utf-8")
            buf.extend(struct.pack(">H", len(raw))); buf.extend(raw)

    @staticmethod
    def _decode_value(data: bytes | bytearray, offset: int) -> tuple[Any, int]:
        type_tag = data[offset]; offset += 1
        if type_tag == _TYPE_FLOAT:
            return round(struct.unpack_from(">f", data, offset)[0], 4), offset + 4
        if type_tag == _TYPE_INT:
            return struct.unpack_from(">b", data, offset)[0], offset + 1
        if type_tag == _TYPE_BOOL:
            return data[offset] != 0x00, offset + 1
        if type_tag == _TYPE_STR:
            return _TOKEN_TO_STR.get(data[offset], f"unknown_0x{data[offset]:02X}"), offset + 1
        if type_tag == _TYPE_RAW:
            str_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
            return data[offset:offset + str_len].decode("utf-8"), offset + str_len
        raise ValueError(f"未知类型标签: 0x{type_tag:02X}")
