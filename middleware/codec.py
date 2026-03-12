"""
AstroSASF · Middleware · SpaceMCPCodec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
静态字典压缩引擎 —— 将冗长的 JSON MCP 工具调用压缩为紧凑的二进制帧。

核心思想：航天总线（SpaceWire / 1553B）带宽极低（< 1Mbps），大模型输出
的 JSON 文本充斥了大量重复的高频键名。本模块通过**预置静态字典**将这些
高频字符串映射为单字节 Token ID，配合 struct 打包数值参数，实现 > 85%
的压缩率。

帧格式 (Frame Layout)
~~~~~~~~~~~~~~~~~~~~~~
::

    ┌──────────┬───────────┬──────────┬──────────┬──────┐
    │ MAGIC(1) │ SKILL(1)  │ N_PARAMS │ PARAM_1  │ ...  │
    │  0xA5    │ token_id  │  (1B)    │ (var)    │      │
    └──────────┴───────────┴──────────┴──────────┴──────┘

    每个 PARAM 块：
    ┌───────────┬──────────┬───────────────┐
    │ KEY(1)    │ TYPE(1)  │ VALUE (1~4B)  │
    │ token_id  │ F/I/B    │               │
    └───────────┴──────────┴───────────────┘

    TYPE 字节：
    - 0x01 = float32 (4 bytes)
    - 0x02 = int8    (1 byte)
    - 0x03 = bool    (1 byte: 0x00/0x01)
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Magic Byte                                                                  #
# --------------------------------------------------------------------------- #

_FRAME_MAGIC: int = 0xA5  # 帧起始标识

# --------------------------------------------------------------------------- #
#  Static Dictionary — 高频字符串 ↔ 单字节 Token                               #
# --------------------------------------------------------------------------- #

# 正向映射：字符串 → Token ID
_STR_TO_TOKEN: dict[str, int] = {
    # ---- Skill 名称 ----
    "set_temperature":    0x01,
    "move_robotic_arm":   0x02,
    "toggle_vacuum_pump": 0x03,
    # ---- 参数键名 ----
    "target":             0x10,
    "target_angle":       0x11,
    "activate":           0x12,
    # ---- 状态/响应键名 ----
    "skill":              0x20,
    "status":             0x21,
    "detail":             0x22,
    "fsm_state":          0x23,
    # ---- 状态值 ----
    "success":            0x30,
    "error":              0x31,
}

# 反向映射：Token ID → 字符串
_TOKEN_TO_STR: dict[int, str] = {v: k for k, v in _STR_TO_TOKEN.items()}

# --------------------------------------------------------------------------- #
#  Value Type Tags                                                             #
# --------------------------------------------------------------------------- #

_TYPE_FLOAT: int = 0x01   # float32 → 4 bytes
_TYPE_INT:   int = 0x02   # int8    → 1 byte
_TYPE_BOOL:  int = 0x03   # bool    → 1 byte
_TYPE_STR:   int = 0x04   # token   → 1 byte (字典内字符串)
_TYPE_RAW:   int = 0x05   # raw UTF-8 字符串 → len(2B) + bytes


# --------------------------------------------------------------------------- #
#  SpaceMCPCodec                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SpaceMCPCodec:
    """静态字典二进制编解码器。

    将 MCP Skill 调用的 JSON 结构压缩为紧凑的 ``bytearray``，
    并支持从二进制帧还原回 JSON。
    """

    lab_id: str
    _encode_count: int = field(default=0, init=False)
    _decode_count: int = field(default=0, init=False)
    _total_json_bytes: int = field(default=0, init=False)
    _total_binary_bytes: int = field(default=0, init=False)

    # ==================================================================== #
    #  ENCODE: JSON dict → bytearray                                       #
    # ==================================================================== #

    def encode(self, request: dict[str, Any]) -> bytearray:
        """将 Skill 调用请求编码为二进制帧。

        Parameters
        ----------
        request : dict
            包含 ``skill`` 和 ``params`` 键的 JSON 请求。

        Returns
        -------
        bytearray
            紧凑的二进制帧。
        """
        buf = bytearray()

        # 1) Magic byte
        buf.append(_FRAME_MAGIC)

        # 2) Skill token
        skill_name: str = request.get("skill", "")
        skill_token = _STR_TO_TOKEN.get(skill_name)
        if skill_token is None:
            raise ValueError(f"[{self.lab_id}] 未知 Skill 无法编码: {skill_name}")
        buf.append(skill_token)

        # 3) Params
        params: dict[str, Any] = request.get("params", {})
        buf.append(len(params))  # param count

        for key, value in params.items():
            # Key token
            key_token = _STR_TO_TOKEN.get(key)
            if key_token is None:
                raise ValueError(f"[{self.lab_id}] 未知参数键无法编码: {key}")
            buf.append(key_token)

            # Value (type-tagged)
            self._encode_value(buf, value)

        self._encode_count += 1
        json_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        binary_bytes = len(buf)
        self._total_json_bytes += json_bytes
        self._total_binary_bytes += binary_bytes

        return buf

    def encode_response(self, response: dict[str, Any]) -> bytearray:
        """将 Skill 执行响应编码为二进制帧。

        Parameters
        ----------
        response : dict
            包含 ``skill``, ``status``, ``detail``, ``fsm_state`` 等键的响应。

        Returns
        -------
        bytearray
            紧凑的二进制帧。
        """
        buf = bytearray()
        buf.append(_FRAME_MAGIC)

        # 响应字段数
        buf.append(len(response))

        for key, value in response.items():
            key_token = _STR_TO_TOKEN.get(key)
            if key_token is not None:
                buf.append(key_token)
            else:
                # 未知键 → 原始编码
                buf.append(0xFF)
                raw = key.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)

            self._encode_value(buf, value)

        return buf

    # ==================================================================== #
    #  DECODE: bytearray → JSON dict                                       #
    # ==================================================================== #

    def decode(self, data: bytearray | bytes) -> dict[str, Any]:
        """将二进制请求帧解码为 JSON 请求。"""
        offset = 0

        # 1) Magic
        if data[offset] != _FRAME_MAGIC:
            raise ValueError(f"[{self.lab_id}] 无效帧: Magic 字节错误")
        offset += 1

        # 2) Skill token
        skill_token = data[offset]
        offset += 1
        skill_name = _TOKEN_TO_STR.get(skill_token, f"unknown_0x{skill_token:02X}")

        # 3) Param count
        n_params = data[offset]
        offset += 1

        params: dict[str, Any] = {}
        for _ in range(n_params):
            key_token = data[offset]
            offset += 1
            key_name = _TOKEN_TO_STR.get(key_token, f"unknown_0x{key_token:02X}")

            value, offset = self._decode_value(data, offset)
            params[key_name] = value

        self._decode_count += 1
        return {"skill": skill_name, "params": params}

    def decode_response(self, data: bytearray | bytes) -> dict[str, Any]:
        """将二进制响应帧解码为 JSON 响应。"""
        offset = 0

        if data[offset] != _FRAME_MAGIC:
            raise ValueError(f"[{self.lab_id}] 无效响应帧: Magic 字节错误")
        offset += 1

        n_fields = data[offset]
        offset += 1

        result: dict[str, Any] = {}
        for _ in range(n_fields):
            key_byte = data[offset]
            offset += 1
            if key_byte == 0xFF:
                # 原始字符串键
                str_len = struct.unpack_from(">H", data, offset)[0]
                offset += 2
                key_name = data[offset:offset + str_len].decode("utf-8")
                offset += str_len
            else:
                key_name = _TOKEN_TO_STR.get(key_byte, f"unknown_0x{key_byte:02X}")

            value, offset = self._decode_value(data, offset)
            result[key_name] = value

        return result

    # ==================================================================== #
    #  Statistics                                                           #
    # ==================================================================== #

    @staticmethod
    def calculate_compression_ratio(json_size: int, binary_size: int) -> float:
        """计算压缩率。

        Returns
        -------
        float
            压缩率百分比，例如 92.9 表示节省了 92.9% 的空间。
        """
        if json_size == 0:
            return 0.0
        return (1.0 - binary_size / json_size) * 100.0

    @property
    def stats(self) -> dict[str, Any]:
        """返回编解码统计信息。"""
        ratio = self.calculate_compression_ratio(
            self._total_json_bytes, self._total_binary_bytes,
        )
        return {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "total_json_bytes": self._total_json_bytes,
            "total_binary_bytes": self._total_binary_bytes,
            "overall_compression_ratio": f"{ratio:.1f}%",
        }

    # ==================================================================== #
    #  Internal helpers                                                     #
    # ==================================================================== #

    @staticmethod
    def _encode_value(buf: bytearray, value: Any) -> None:
        """将单个值编码并追加到缓冲区。"""
        if isinstance(value, bool):
            buf.append(_TYPE_BOOL)
            buf.append(0x01 if value else 0x00)
        elif isinstance(value, float):
            buf.append(_TYPE_FLOAT)
            buf.extend(struct.pack(">f", value))
        elif isinstance(value, int):
            buf.append(_TYPE_INT)
            buf.extend(struct.pack(">b", max(-128, min(127, value))))
        elif isinstance(value, str):
            token = _STR_TO_TOKEN.get(value)
            if token is not None:
                buf.append(_TYPE_STR)
                buf.append(token)
            else:
                buf.append(_TYPE_RAW)
                raw = value.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)
        else:
            # Fallback: JSON 字符串
            buf.append(_TYPE_RAW)
            raw = str(value).encode("utf-8")
            buf.extend(struct.pack(">H", len(raw)))
            buf.extend(raw)

    @staticmethod
    def _decode_value(data: bytes | bytearray, offset: int) -> tuple[Any, int]:
        """从缓冲区解码单个值，返回 (value, new_offset)。"""
        type_tag = data[offset]
        offset += 1

        if type_tag == _TYPE_FLOAT:
            value = struct.unpack_from(">f", data, offset)[0]
            value = round(value, 4)  # 规避浮点精度噪声
            return value, offset + 4

        if type_tag == _TYPE_INT:
            value = struct.unpack_from(">b", data, offset)[0]
            return value, offset + 1

        if type_tag == _TYPE_BOOL:
            value = data[offset] != 0x00
            return value, offset + 1

        if type_tag == _TYPE_STR:
            token = data[offset]
            return _TOKEN_TO_STR.get(token, f"unknown_0x{token:02X}"), offset + 1

        if type_tag == _TYPE_RAW:
            str_len = struct.unpack_from(">H", data, offset)[0]
            offset += 2
            raw = data[offset:offset + str_len].decode("utf-8")
            return raw, offset + str_len

        raise ValueError(f"未知类型标签: 0x{type_tag:02X}")
