"""
AstroSASF · Middleware · SpaceMCPCodec (V4.2 — 动态字典)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于启动期协商的动态字典压缩引擎 —— 彻底消除静态硬编码映射表。

核心变化（V4.2）：
**完全删除静态 ``_STR_TO_TOKEN`` 字典**。在系统启动时，``SpaceMCPCodec``
从 ``MCPToolRegistry.all_vocabulary()`` 接收完整词汇表，自动按字母序
分配 ``0x01~0xFF`` 的单字节 Token ID，在内存中构建正反映射表。

这意味着：新增一个 MCP Tool 时 **无需修改 Codec 任何代码**。
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Frame Constants                                                             #
# --------------------------------------------------------------------------- #

_FRAME_MAGIC: int = 0xA5

_TYPE_FLOAT: int = 0x01
_TYPE_INT:   int = 0x02
_TYPE_BOOL:  int = 0x03
_TYPE_STR:   int = 0x04       # Token 映射命中
_TYPE_RAW:   int = 0x05       # Token 未命中，原文传输


# --------------------------------------------------------------------------- #
#  SpaceMCPCodec                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SpaceMCPCodec:
    """基于动态字典协商的 Space-MCP 二进制编解码器。

    Parameters
    ----------
    lab_id : str
        实验柜标识。
    vocabulary : list[str]
        启动期协商的词汇表（由 ``MCPToolRegistry.all_vocabulary()`` 提供）。
        Codec 按此列表的排序自动分配 Token ID。
    """

    lab_id: str
    vocabulary: list[str] = field(default_factory=list)

    # ── 动态映射表（__post_init__ 中构建） ── #
    _str_to_token: dict[str, int] = field(default_factory=dict, init=False)
    _token_to_str: dict[int, str] = field(default_factory=dict, init=False)

    # ── 统计 ── #
    _encode_count: int = field(default=0, init=False)
    _decode_count: int = field(default=0, init=False)
    _total_json_bytes: int = field(default=0, init=False)
    _total_binary_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """根据词汇表动态分配 Token ID（0x01 起始）。"""
        self._str_to_token = {}
        self._token_to_str = {}

        for i, word in enumerate(self.vocabulary):
            token_id = i + 1      # 0x01 起始，0x00 保留
            if token_id > 0xFF:
                logger.warning(
                    "[%s] Codec: 词汇表超过 255 项，截断", self.lab_id,
                )
                break
            self._str_to_token[word] = token_id
            self._token_to_str[token_id] = word

        logger.info(
            "[%s] 📖 Codec 动态字典协商完成: %d 个词条已映射",
            self.lab_id, len(self._str_to_token),
        )
        for word, tid in self._str_to_token.items():
            logger.info(
                "[%s]    0x%02X ← '%s'", self.lab_id, tid, word,
            )

    # ================================================================== #
    #  编码                                                                #
    # ================================================================== #

    def encode(self, request: dict[str, Any]) -> bytearray:
        """将 MCP Tool 调用 JSON 编码为紧凑二进制帧。"""
        buf = bytearray()
        buf.append(_FRAME_MAGIC)

        skill_name: str = request.get("skill", "")
        skill_token = self._str_to_token.get(skill_name)
        if skill_token is None:
            raise ValueError(
                f"[{self.lab_id}] Codec: Tool '{skill_name}' 不在动态字典中"
            )
        buf.append(skill_token)

        params: dict[str, Any] = request.get("params", {})
        buf.append(len(params))
        for key, value in params.items():
            key_token = self._str_to_token.get(key)
            if key_token is None:
                # fallback: 原文传输
                buf.append(0xFF)
                raw = key.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)
            else:
                buf.append(key_token)
            self._encode_value(buf, value)

        self._encode_count += 1
        json_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        self._total_json_bytes += json_bytes
        self._total_binary_bytes += len(buf)
        return buf

    def encode_response(self, response: dict[str, Any]) -> bytearray:
        """将响应 dict 编码为二进制帧。"""
        buf = bytearray()
        buf.append(_FRAME_MAGIC)
        buf.append(len(response))
        for key, value in response.items():
            key_token = self._str_to_token.get(key)
            if key_token is not None:
                buf.append(key_token)
            else:
                buf.append(0xFF)
                raw = key.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)
            self._encode_value(buf, value)
        return buf

    # ================================================================== #
    #  解码                                                                #
    # ================================================================== #

    def decode(self, data: bytearray | bytes) -> dict[str, Any]:
        """将二进制帧解码为 MCP Tool 调用 JSON。"""
        offset = 0
        if data[offset] != _FRAME_MAGIC:
            raise ValueError(f"[{self.lab_id}] 无效帧: Magic 字节错误")
        offset += 1

        skill_token = data[offset]; offset += 1
        skill_name = self._token_to_str.get(
            skill_token, f"unknown_0x{skill_token:02X}",
        )

        n_params = data[offset]; offset += 1
        params: dict[str, Any] = {}
        for _ in range(n_params):
            key_byte = data[offset]; offset += 1
            if key_byte == 0xFF:
                str_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
                key_name = data[offset:offset + str_len].decode("utf-8"); offset += str_len
            else:
                key_name = self._token_to_str.get(
                    key_byte, f"unknown_0x{key_byte:02X}",
                )
            value, offset = self._decode_value_with_dict(data, offset)
            params[key_name] = value

        self._decode_count += 1
        return {"skill": skill_name, "params": params}

    def decode_response(self, data: bytearray | bytes) -> dict[str, Any]:
        """将二进制响应帧解码为 dict。"""
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
                key_name = self._token_to_str.get(
                    key_byte, f"unknown_0x{key_byte:02X}",
                )
            value, offset = self._decode_value_with_dict(data, offset)
            result[key_name] = value
        return result

    # ================================================================== #
    #  统计                                                                #
    # ================================================================== #

    @staticmethod
    def calculate_compression_ratio(json_size: int, binary_size: int) -> float:
        if json_size == 0:
            return 0.0
        return (1.0 - binary_size / json_size) * 100.0

    @property
    def stats(self) -> dict[str, Any]:
        ratio = self.calculate_compression_ratio(
            self._total_json_bytes, self._total_binary_bytes,
        )
        return {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "total_json_bytes": self._total_json_bytes,
            "total_binary_bytes": self._total_binary_bytes,
            "overall_compression_ratio": f"{ratio:.1f}%",
            "dictionary_size": len(self._str_to_token),
        }

    @property
    def dictionary_table(self) -> dict[str, int]:
        """返回当前动态字典映射表（供日志展示）。"""
        return dict(self._str_to_token)

    # ================================================================== #
    #  内部编解码工具                                                       #
    # ================================================================== #

    def _encode_value(self, buf: bytearray, value: Any) -> None:
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
            token = self._str_to_token.get(value)
            if token is not None:
                buf.append(_TYPE_STR)
                buf.append(token)
            else:
                buf.append(_TYPE_RAW)
                raw = value.encode("utf-8")
                buf.extend(struct.pack(">H", len(raw)))
                buf.extend(raw)
        else:
            buf.append(_TYPE_RAW)
            raw = str(value).encode("utf-8")
            buf.extend(struct.pack(">H", len(raw)))
            buf.extend(raw)

    def _decode_value_with_dict(
        self, data: bytes | bytearray, offset: int,
    ) -> tuple[Any, int]:
        """带字典反查的值解码。"""
        type_tag = data[offset]; offset += 1
        if type_tag == _TYPE_FLOAT:
            return round(struct.unpack_from(">f", data, offset)[0], 4), offset + 4
        if type_tag == _TYPE_INT:
            return struct.unpack_from(">b", data, offset)[0], offset + 1
        if type_tag == _TYPE_BOOL:
            return data[offset] != 0x00, offset + 1
        if type_tag == _TYPE_STR:
            token_id = data[offset]; offset += 1
            return self._token_to_str.get(token_id, f"0x{token_id:02X}"), offset
        if type_tag == _TYPE_RAW:
            str_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
            return data[offset:offset + str_len].decode("utf-8"), offset + str_len
        raise ValueError(f"未知类型标签: 0x{type_tag:02X}")
