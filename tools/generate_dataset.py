#!/usr/bin/env python3
"""
AstroSASF · Benchmark Dataset Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
生成标准化测试数据集 `astro_bench.jsonl`，用于评估操作系统的并发调度能力
和灾难逃生能力。

Usage:
    python tools/generate_dataset.py

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
#  Pydantic Schema (数据集结构定义)                                           #
# --------------------------------------------------------------------------- #

class ChaosEvent(BaseModel):
    """混沌事件：模拟硬件延迟或遥测报警。"""
    trigger_time_sec: float = Field(
        ...,
        ge=0.0,
        description="触发时间（秒，从 episode 开始计时）"
    )
    type: str = Field(
        ...,
        description="事件类型：'hardware_delay' 或 'telemetry_alarm'"
    )
    target_tool: str | None = Field(
        default=None,
        description="目标工具名称（如 vacuum_pump, robotic_arm）"
    )
    delay_multiplier: float | None = Field(
        default=None,
        ge=0.1,
        le=10.0,
        description="延迟倍数（如 3.0 表示慢 3 倍）"
    )
    telemetry_key: str | None = Field(
        default=None,
        description="遥测键名（如 temperature, smoke_level）"
    )
    override_value: float | None = Field(
        default=None,
        description="遥测覆盖值（如 85.0 触发温度 Guard）"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"hardware_delay", "telemetry_alarm"}
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}, got: {v}")
        return v


class BenchmarkEpisode(BaseModel):
    """单个评测 Episode。"""
    episode_id: str = Field(
        ...,
        description="唯一标识符（格式：bench-YYYYMMDD-XXXXXX）"
    )
    difficulty: str = Field(
        ...,
        description="难度等级：Easy / Medium / Hard"
    )
    description: str = Field(
        ...,
        description="场景描述（人类可读）"
    )
    astronaut_prompts: list[str] = Field(
        ...,
        min_length=1,
        description="宇航员并发下发的自然语言指令列表"
    )
    initial_telemetry: dict[str, float] = Field(
        ...,
        description="初始遥测状态"
    )
    chaos_events: list[ChaosEvent] = Field(
        default_factory=list,
        description="混沌事件列表"
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"Easy", "Medium", "Hard"}
        if v not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}, got: {v}")
        return v


# --------------------------------------------------------------------------- #
#  Prompt 模板库（多样化语义表达）                                             #
# --------------------------------------------------------------------------- #

class PromptLibrary:
    """宇航员 Prompt 模板库（支持多样化语义表达）。"""

    # 流体实验相关
    FLUID_PROMPTS = [
        "帮我把流体柜抽个真空准备一下",
        "启动微重力流体测试流程",
        "执行流体样品的真空脱气操作",
        "准备流体实验舱的真空环境",
        "开启微重力流体行为观测实验",
        "对流体样本进行抽真空预处理",
        "启动流体物理特性测试",
        "帮我把流体实验装置调到真空模式",
    ]

    # 生物培养相关
    BIO_PROMPTS = [
        "开始太空生物细胞培养流程",
        "启动细胞培养舱的温控程序",
        "执行微重力环境下的细胞培养",
        "准备生物培养舱的 37 度环境",
        "开启新一轮细胞生长实验",
        "帮我设置生物实验柜的培养温度",
        "启动生物样本的培养基注入",
        "执行太空生物实验前的准备工作",
    ]

    # 材料合成相关
    MATERIAL_PROMPTS = [
        "启动材料合成实验流程",
        "帮我把合成舱加热到 500 度",
        "执行晶体生长实验",
        "开启高温材料制备流程",
        "准备材料合成的热处理环境",
        "启动太空材料实验柜",
        "执行金属合金的微重力合成",
        "帮我开启材料热处理程序",
    ]

    # 火情应急相关
    FIRE_PROMPTS = [
        "启动火情应急响应程序",
        "帮我开启烟雾传感器的监控",
        "执行舱段火情检测流程",
        "准备消防系统的预启动",
        "开启紧急报警测试",
        "帮我检查各舱段的火灾隐患",
        "执行太空舱消防演练流程",
        "启动火情预警机制",
    ]

    # 植物生长相关
    PLANT_PROMPTS = [
        "启动植物生长监测程序",
        "帮我开启植物舱的补光灯",
        "执行植物灌溉流程",
        "准备植物培养舱的环境参数",
        "开启太空农业实验监测",
        "帮我设置植物舱的光照周期",
        "执行植物生长数据采集",
        "启动植物舱的温湿度调控",
    ]

    # 通用实验相关
    GENERIC_PROMPTS = [
        "帮我准备实验环境",
        "启动标准操作流程",
        "执行实验前的设备检查",
        "开启常规实验准备流程",
        "帮我初始化实验舱",
        "执行系统预热流程",
        "启动实验前的校准程序",
        "帮我检查实验柜状态",
    ]

    ALL_PROMPTS = {
        "fluid": FLUID_PROMPTS,
        "bio": BIO_PROMPTS,
        "material": MATERIAL_PROMPTS,
        "fire": FIRE_PROMPTS,
        "plant": PLANT_PROMPTS,
        "generic": GENERIC_PROMPTS,
    }


# --------------------------------------------------------------------------- #
#  初始遥测状态模板                                                           #
# --------------------------------------------------------------------------- #

DEFAULT_TELEMETRY = {
    "temperature": 25.0,       # 摄氏度
    "humidity": 50.0,          # 百分比
    "pressure": 101.325,       # kPa (标准大气压)
    "co2_level": 400.0,        # ppm
    "smoke_level": 0.0,        # 0.0 ~ 1.0
    "flame_detected": 0.0,     # 0.0 ~ 1.0
    "vacuum_level": 101.325,   # kPa (真空度，101.325 = 常压)
    "light_intensity": 0.0,     # μmol/m²/s
    "soil_moisture": 40.0,     # 百分比
    "nutrient_level": 80.0,    # 百分比
}


# --------------------------------------------------------------------------- #
#  数据集生成器                                                               #
# --------------------------------------------------------------------------- #

class DatasetGenerator:
    """生成 astro_bench.jsonl 数据集。"""

    def __init__(
        self,
        output_path: str = "datasets/astro_bench.jsonl",
        seed: int | None = 42,
    ):
        self.output_path = Path(output_path)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    # ── 难度分布 ── #
    DIFFICULTY_COUNTS = {
        "Easy": 15,
        "Medium": 20,
        "Hard": 15,
    }

    # ── 冲突资源池 ── #
    CONFLICT_TOOLS = [
        "vacuum_pump",
        "robotic_arm",
        "heater",
        "water_pump",
        "co2_controller",
    ]

    def _generate_episode_id(self) -> str:
        """生成唯一 Episode ID。"""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        suffix = uuid.uuid4().hex[:6].upper()
        return f"bench-{today}-{suffix}"

    def _pick_prompts(self, count: int, domains: list[str]) -> list[str]:
        """从指定领域随机选择 Prompt。"""
        result: list[str] = []
        for _ in range(count):
            domain = random.choice(domains)
            prompts = PromptLibrary.ALL_PROMPTS.get(domain, PromptLibrary.GENERIC_PROMPTS)
            result.append(random.choice(prompts))
        return result

    def _gen_telemetry_for_prompts(self, prompts: list[str]) -> dict[str, float]:
        """根据 Prompt 内容推断初始遥测状态。"""
        telemetry = dict(DEFAULT_TELEMETRY)

        if any(kw in " ".join(prompts) for kw in ["流体", "真空", "抽真空"]):
            telemetry["vacuum_level"] = 101.325  # 从常压开始抽真空

        if any(kw in " ".join(prompts) for kw in ["细胞", "培养", "生物", "37"]):
            telemetry["temperature"] = 25.0  # 接近培养温度
            telemetry["humidity"] = 70.0

        if any(kw in " ".join(prompts) for kw in ["合成", "加热", "500", "热处理", "高温"]):
            telemetry["temperature"] = 20.0  # 准备加热

        if any(kw in " ".join(prompts) for kw in ["植物", "生长", "光照", "灌溉"]):
            telemetry["light_intensity"] = 200.0
            telemetry["temperature"] = 22.0
            telemetry["humidity"] = 75.0
            telemetry["soil_moisture"] = 50.0

        if any(kw in " ".join(prompts) for kw in ["火情", "消防", "烟雾", "报警"]):
            telemetry["smoke_level"] = 0.0  # 正常状态
            telemetry["temperature"] = 25.0

        return telemetry

    # ── Easy: 2-3 个无冲突 Prompt，无 Chaos ── #
    def _generate_easy_episode(self) -> BenchmarkEpisode:
        domains_easy = [["fluid", "bio"], ["bio", "generic"], ["fluid", "material"]]
        domains = random.choice(domains_easy)
        prompt_count = random.randint(2, 3)

        prompts = self._pick_prompts(prompt_count, domains)
        description = f"Easy 并发实验：{' + '.join(domains)}，无资源冲突"

        return BenchmarkEpisode(
            episode_id=self._generate_episode_id(),
            difficulty="Easy",
            description=description,
            astronaut_prompts=prompts,
            initial_telemetry=self._gen_telemetry_for_prompts(prompts),
            chaos_events=[],
        )

    # ── Medium: 3-4 个强冲突 Prompt + hardware_delay ── #
    def _generate_medium_episode(self) -> BenchmarkEpisode:
        conflict_tool = random.choice(self.CONFLICT_TOOLS)
        prompt_count = random.randint(3, 4)

        # 所有 Prompt 都涉及同一个冲突工具
        if conflict_tool == "vacuum_pump":
            domains = ["fluid"] * prompt_count
        elif conflict_tool == "robotic_arm":
            domains = ["bio"] * prompt_count  # 生物实验常用机械臂
        elif conflict_tool == "heater":
            domains = ["material"] * prompt_count
        elif conflict_tool == "water_pump":
            domains = ["plant"] * prompt_count
        else:
            domains = ["generic"] * prompt_count

        prompts = self._pick_prompts(prompt_count, domains)
        description = (
            f"Medium 资源冲突：{prompt_count} 个 Prompt 竞争 {conflict_tool}，"
            f"注入 hardware_delay（延迟 3 倍）"
        )

        # hardware_delay 事件
        delay_multiplier = random.uniform(2.5, 4.0)
        chaos = ChaosEvent(
            trigger_time_sec=0.5,
            type="hardware_delay",
            target_tool=conflict_tool,
            delay_multiplier=delay_multiplier,
        )

        return BenchmarkEpisode(
            episode_id=self._generate_episode_id(),
            difficulty="Medium",
            description=description,
            astronaut_prompts=prompts,
            initial_telemetry=self._gen_telemetry_for_prompts(prompts),
            chaos_events=[chaos],
        )

    # ── Hard: 4-5 个大并发 + 冲突 + telemetry_alarm ── #
    def _generate_hard_episode(self) -> BenchmarkEpisode:
        conflict_tool = random.choice(self.CONFLICT_TOOLS)
        prompt_count = random.randint(4, 5)

        if conflict_tool == "vacuum_pump":
            domains = ["fluid"] * prompt_count
        elif conflict_tool == "robotic_arm":
            domains = ["bio"] * prompt_count
        elif conflict_tool == "heater":
            domains = ["material"] * prompt_count
        elif conflict_tool == "water_pump":
            domains = ["plant"] * prompt_count
        else:
            domains = ["generic"] * prompt_count

        prompts = self._pick_prompts(prompt_count, domains)

        # 在第 [5.0, 8.0] 秒区间注入 telemetry_alarm
        alarm_time = random.uniform(5.0, 8.0)

        # 随机选择报警类型
        alarm_types = [
            {"telemetry_key": "temperature", "override_value": 85.0, "desc": "温度骤升"},
            {"telemetry_key": "smoke_level", "override_value": 0.9, "desc": "烟雾浓度超限"},
            {"telemetry_key": "pressure", "override_value": 150.0, "desc": "压力超限"},
            {"telemetry_key": "co2_level", "override_value": 2000.0, "desc": "CO2 浓度过高"},
        ]
        alarm_cfg = random.choice(alarm_types)

        description = (
            f"Hard 极限并发：{prompt_count} 个 Prompt 竞争 {conflict_tool}，"
            f"在 {alarm_time:.1f}s 触发 {alarm_cfg['desc']}（{alarm_cfg['telemetry_key']}={alarm_cfg['override_value']}），"
            f"测试硬件抢占和灾难逃生"
        )

        # hardware_delay + telemetry_alarm
        chaos_events = [
            ChaosEvent(
                trigger_time_sec=0.5,
                type="hardware_delay",
                target_tool=conflict_tool,
                delay_multiplier=random.uniform(2.0, 3.0),
            ),
            ChaosEvent(
                trigger_time_sec=alarm_time,
                type="telemetry_alarm",
                telemetry_key=alarm_cfg["telemetry_key"],
                override_value=alarm_cfg["override_value"],
            ),
        ]

        # Hard 场景的遥测状态稍微调整，便于触发报警
        telemetry = self._gen_telemetry_for_prompts(prompts)
        if alarm_cfg["telemetry_key"] == "temperature":
            telemetry["temperature"] = 75.0  # 接近阈值
        elif alarm_cfg["telemetry_key"] == "smoke_level":
            telemetry["smoke_level"] = 0.7  # 接近阈值
        elif alarm_cfg["telemetry_key"] == "pressure":
            telemetry["pressure"] = 130.0  # 接近阈值

        return BenchmarkEpisode(
            episode_id=self._generate_episode_id(),
            difficulty="Hard",
            description=description,
            astronaut_prompts=prompts,
            initial_telemetry=telemetry,
            chaos_events=chaos_events,
        )

    def generate(self) -> list[BenchmarkEpisode]:
        """生成全部 50 条评测数据。"""
        episodes: list[BenchmarkEpisode] = []

        for difficulty, count in self.DIFFICULTY_COUNTS.items():
            for _ in range(count):
                if difficulty == "Easy":
                    ep = self._generate_easy_episode()
                elif difficulty == "Medium":
                    ep = self._generate_medium_episode()
                else:  # Hard
                    ep = self._generate_hard_episode()

                episodes.append(ep)

        # 打乱顺序（混合难度）
        random.shuffle(episodes)

        # 重新编号（保持 ID 唯一）
        for i, ep in enumerate(episodes, start=1):
            base_id = ep.episode_id
            # 保留日期和随机后缀，只在日志中标注序号
            logger_ep = BenchmarkEpisode(
                episode_id=base_id,
                difficulty=ep.difficulty,
                description=f"[#{i:02d}] {ep.description}",
                astronaut_prompts=ep.astronaut_prompts,
                initial_telemetry=ep.initial_telemetry,
                chaos_events=ep.chaos_events,
            )
            episodes[i - 1] = logger_ep

        return episodes

    def save(self, episodes: list[BenchmarkEpisode]) -> None:
        """追加写入 JSONL 文件。"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.output_path.exists() else "w"
        with open(self.output_path, mode, encoding="utf-8") as f:
            for ep in episodes:
                f.write(ep.model_dump_json(indent=None) + "\n")

        print(f"✅ 写入 {len(episodes)} 条评测数据 → {self.output_path}")
        print(f"   Easy: {sum(1 for e in episodes if e.difficulty == 'Easy')} 条")
        print(f"   Medium: {sum(1 for e in episodes if e.difficulty == 'Medium')} 条")
        print(f"   Hard: {sum(1 for e in episodes if e.difficulty == 'Hard')} 条")

    def validate(self, path: str | Path) -> dict[str, Any]:
        """验证已生成的数据集。"""
        path = Path(path)
        if not path.exists():
            return {"valid": False, "error": "文件不存在"}

        records: list[dict[str, Any]] = []
        errors: list[str] = []

        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Pydantic 验证
                    BenchmarkEpisode.model_validate(record)
                    records.append(record)
                except Exception as exc:
                    errors.append(f"第 {line_no} 行: {exc}")

        difficulty_counts = {}
        for r in records:
            d = r["difficulty"]
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

        return {
            "valid": len(errors) == 0,
            "total_records": len(records),
            "difficulty_counts": difficulty_counts,
            "errors": errors,
        }


# --------------------------------------------------------------------------- #
#  Main                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("╔" + "═" * 60 + "╗")
    print("║  AstroSASF Benchmark Dataset Generator (V7.1)                  ║")
    print("╚" + "═" * 60 + "╝")

    generator = DatasetGenerator(
        output_path="datasets/astro_bench.jsonl",
        seed=42,  # 固定种子保证可复现
    )

    print("\n📦 正在生成 50 条评测数据...")
    episodes = generator.generate()

    print("\n💾 正在写入 datasets/astro_bench.jsonl ...")
    generator.save(episodes)

    print("\n🔍 正在验证数据集...")
    validation = generator.validate("datasets/astro_bench.jsonl")
    if validation["valid"]:
        print("✅ 数据集验证通过!")
        print(f"   总记录数: {validation['total_records']}")
        print(f"   难度分布: {validation['difficulty_counts']}")
    else:
        print("❌ 数据集验证失败!")
        for err in validation["errors"]:
            print(f"   - {err}")

    print("\n🎯 生成完成!")


if __name__ == "__main__":
    main()
