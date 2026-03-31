"""
AstroSASF V7.1 · Demo · Integration Example
============================================
演示「外部团队的多智能体框架」如何通过 AstroOSFacade 接入 AstroSASF。

场景：
1. 外部 Agent 初始化 Facade
2. 获取实验舱工具列表，构建 Agent 的 Tool Registry
3. 调用底层 LLM 进行自然语言理解
4. 提交 Tool Call 到调度器
5. 订阅硬件报警，接收抢占通知

Author: AstroSASF Team
Version: 7.1
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sasf.core.config_loader import load_config
from sasf.core.models import TaskPriority
from sasf.core.environment import LaboratoryEnvironment
from sasf.core.os_gateway import AstroOSFacade, ExecutionStatus, ToolCallResult
from sasf.physics.interlock_engine import InterlockEngine, InterlockRule
from sasf.middleware.mcp_registry import MCPToolRegistry, MCPToolContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  示例：业务层 Tool 注册函数（外部团队需要实现的回调）                          #
# --------------------------------------------------------------------------- #

def register_bio_lab_tools(
    registry: MCPToolRegistry,
    engine: InterlockEngine,
    bus: Any,
) -> None:
    """生物实验舱工具注册示例。

    外部团队需要提供此函数，用于注册业务相关的 MCP Tools。
    AstroSASF 框架会自动调用此函数注册工具。
    """

    @registry.mcp_tool(
        forbid_states={"heater": "ACTIVE"},
        telemetry_rules=["oxygen_level > 10"],
    )
    async def read_sensor(ctx: MCPToolContext, channel: int) -> dict:
        """读取传感器通道值。"""
        # 从遥测总线读取实际数据
        telemetry = await ctx.bus.snapshot()
        key = f"sensor_channel_{channel}"
        if key in telemetry:
            value = telemetry[key]
        else:
            # 如果没有预设值，返回模拟数据
            value = 20.0 + channel * 0.5
        return {"status": "success", "channel": channel, "value": value}

    @registry.mcp_tool(
        require_states={"vacuum": "IDLE"},
    )
    async def control_heater(ctx: MCPToolContext, temperature: float, duration: int) -> dict:
        """控制加热器设置温度和持续时间。"""
        await ctx.engine.set_subsystem_state("heater", "ACTIVE")
        await asyncio.sleep(duration / 1000.0)  # 模拟加热
        await ctx.engine.set_subsystem_state("heater", "IDLE")
        return {"status": "success", "temperature": temperature, "duration": duration}

    @registry.mcp_tool(
        forbid_states={"arm": "MOVING"},
    )
    async def move_robotic_arm(ctx: MCPToolContext, position: str) -> dict:
        """移动机械臂到指定位置。"""
        await ctx.engine.set_subsystem_state("arm", "MOVING")
        await asyncio.sleep(0.5)  # 模拟移动
        await ctx.engine.set_subsystem_state("arm", "IDLE")
        return {"status": "success", "position": position}

    logger.info("[注册函数] 生物实验舱工具已注册")


# --------------------------------------------------------------------------- #
#  示例：外部 Agent 实现                                                        #
# --------------------------------------------------------------------------- #

class ExternalAgent:
    """外部多智能体框架中的单个 Agent 实现示例。

    该 Agent 通过 AstroOSFacade 与 AstroSASF 内核交互。
    """

    def __init__(
        self,
        agent_id: str,
        facade: AstroOSFacade,
        lab_id: str,
    ):
        self.agent_id = agent_id
        self.facade = facade
        self.lab_id = lab_id
        self.tool_schemas: list[dict] = []

    async def initialize(self) -> None:
        """初始化 Agent —— 获取工具列表，构建本地 Tool Registry。"""
        logger.info("[%s] Agent 初始化中...", self.agent_id)

        # 获取实验舱的工具列表
        self.tool_schemas = await self.facade.get_lab_tools(self.lab_id)

        logger.info("[%s] 发现 %d 个工具:", self.agent_id, len(self.tool_schemas))
        for tool in self.tool_schemas:
            logger.info(
                "  - %s: %s %s",
                tool["name"],
                tool["description"][:50] + "..." if len(tool["description"]) > 50 else tool["description"],
                "[MACRO]" if tool.get("is_macro") else "",
            )

    async def think(self, user_message: str) -> dict:
        """调用底层 LLM 进行思考/推理。

        外部 Agent 的自然语言理解模块通过此接口使用 AstroSASF 的 LLM 算力。
        """
        logger.info("[%s] 请求 LLM 推理: %s", self.agent_id, user_message[:50] + "...")

        messages = [
            {"role": "system", "content": "你是一个太空实验助手。请简洁地回答问题。"},
            {"role": "user", "content": user_message},
        ]

        response = await self.facade.chat_completion(
            messages=messages,
            agent_id=self.agent_id,
            temperature=0.7,
        )

        return response

    async def execute_tool(
        self,
        tool_name: str,
        params: dict,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> ToolCallResult:
        """执行工具调用。

        外部 Agent 的执行器通过此接口提交原子操作到调度器。
        """
        logger.info(
            "[%s] 提交 Tool Call: %s(%s) [priority=%s]",
            self.agent_id, tool_name, params, priority.name,
        )

        result = await self.facade.execute_tool_call(
            lab_id=self.lab_id,
            tool_name=tool_name,
            params=params,
            agent_id=self.agent_id,
            priority=priority,
            timeout=30.0,
        )

        # 记录执行结果
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(
                "[%s] ✅ Tool 执行成功: %s (%.1fms)",
                self.agent_id, tool_name, result.execution_time_ms,
            )
        else:
            logger.warning(
                "[%s] ⚠️  Tool 执行失败: %s [%s] - %s",
                self.agent_id, tool_name, result.status.value, result.detail,
            )

        return result


# --------------------------------------------------------------------------- #
#  示例：硬件报警处理器                                                         #
# --------------------------------------------------------------------------- #

async def on_hardware_alert(interrupt: Any) -> None:
    """硬件报警回调 —— 外部多智能体框架需要实现此逻辑。

    当 AstroSASF 底层发生火灾等紧急情况时，此回调会被触发。
    外部框架应该：
    1. 中断当前所有正在进行的任务
    2. 通知所有相关 Agent
    3. 等待逃生任务完成
    4. 恢复或重新规划任务
    """
    logger.critical("🚨" * 20)
    logger.critical("🚨 [外部框架] 收到硬件报警!")
    logger.critical("🚨 中断 ID: %s", interrupt.interrupt_id)
    logger.critical("🚨 描述: %s", interrupt.description)
    logger.critical("🚨 逃生动作: %s(%s)", interrupt.action_skill, interrupt.action_params)
    logger.critical("🚨 目标实验舱: %s", interrupt.lab_id)
    logger.critical("🚨" * 20)

    # 外部框架的实际处理逻辑：
    # 1. 向所有 Agent 广播 "紧急中断" 消息
    # 2. 取消所有等待中的 Tool Calls
    # 3. 等待 AstroSASF 完成逃生任务
    # 4. 重新评估并规划后续任务


# --------------------------------------------------------------------------- #
#  主演示流程                                                                  #
# --------------------------------------------------------------------------- #

async def demo_integration_flow():
    """完整的集成演示流程。"""

    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  AstroSASF V7.1 北向接口集成演示                                 ║")
    logger.info("║  模拟外部多智能体框架接入 AstroSASF 内核                          ║")
    logger.info("╚" + "═" * 70 + "╝")
    logger.info("")

    # ── Step 1: 加载配置并创建 Facade ── #
    config = load_config("config.yaml")
    facade = AstroOSFacade.from_config_obj(config)

    # ── Step 2: 初始化 Facade（启动 Orchestrator 和 LLM Manager） ── #
    await facade.initialize()

    # ── Step 3: 创建并注册实验舱 ── #
    # 创建 InterlockEngine（定义子系统和联锁规则）
    engine = InterlockEngine(
        lab_id="bio_lab_01",
        subsystems={
            "heater": ["IDLE", "ACTIVE", "COOLING"],
            "vacuum": ["IDLE", "ACTIVE"],
            "arm": ["IDLE", "MOVING"],
            "fan": ["OFF", "LOW", "HIGH"],
        },
        initial_states={
            "heater": "IDLE",
            "vacuum": "IDLE",
            "arm": "IDLE",
            "fan": "OFF",
        },
        interlocks=[
            # 联锁规则：真空激活时禁止移动机械臂
            InterlockRule(
                condition="vacuum == 'ACTIVE' and arm == 'MOVING'",
                message="真空激活时禁止移动机械臂",
            ),
        ],
    )

    # 创建实验舱环境
    # 初始化遥测数据（包含 Guard 规则需要的变量）
    initial_telemetry = {
        "oxygen_level": 21.0,  # Guard: oxygen_level > 10
        "temperature": 25.0,
        "humidity": 50.0,
    }

    lab = LaboratoryEnvironment(
        lab_id="bio_lab_01",
        config=config,
        engine=engine,
        tool_registrar=register_bio_lab_tools,
        initial_telemetry=initial_telemetry,
    )

    # 注册实验舱到 Facade
    await facade.register_lab(lab)

    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  实验舱已注册，开始演示流程                                        ║")
    logger.info("╚" + "═" * 70 + "╝")
    logger.info("")

    # ── Step 4: 创建外部 Agent ── #
    agent = ExternalAgent(
        agent_id="agent_bio_01",
        facade=facade,
        lab_id="bio_lab_01",
    )
    await agent.initialize()

    # ── Step 5: 注册硬件报警回调 ── #
    sub_id = facade.register_hardware_alert_callback(on_hardware_alert)
    logger.info("[主流程] 已注册硬件报警回调 (subscription_id=%s)", sub_id)

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 1: 获取工具列表")
    logger.info("─" * 70)

    # 演示获取工具列表（已在上方 initialize 中完成）
    logger.info("可用工具数量: %d", len(agent.tool_schemas))

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 2: 调用 LLM 推理")
    logger.info("─" * 70)

    # 演示调用 LLM
    llm_response = await agent.think("我需要在生物实验舱中培养细胞，需要设置什么温度？")
    logger.info("[LLM 响应] %s", llm_response.get("content", "N/A")[:200])

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 3: 成功执行 Tool Call")
    logger.info("─" * 70)

    # 演示成功执行工具
    result1 = await agent.execute_tool("read_sensor", {"channel": 1})
    logger.info("[结果] %s", result1.to_dict())

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 4: FSM 拦截（联锁规则触发）")
    logger.info("─" * 70)

    # 先激活真空泵（这会触发联锁规则）
    await engine.set_subsystem_state("vacuum", "ACTIVE")
    logger.info("[FSM] 已激活真空泵")

    # 尝试在真空激活时移动机械臂（应该被 FSM 拦截）
    result2 = await agent.execute_tool("move_robotic_arm", {"position": "home"})
    logger.info("[结果] FSM 拦截演示: %s - %s", result2.status.value, result2.detail)

    # 重置状态
    await engine.set_subsystem_state("vacuum", "IDLE")

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 5: Guard 拦截（前置条件不满足）")
    logger.info("─" * 70)

    # 激活加热器
    await engine.set_subsystem_state("heater", "ACTIVE")
    logger.info("[FSM] 已激活加热器")

    # 尝试在加热器激活时读取传感器（应该被 Guard 拦截）
    result3 = await agent.execute_tool("read_sensor", {"channel": 1})
    logger.info("[结果] Guard 拦截演示: %s - %s", result3.status.value, result3.detail)

    # 重置状态
    await engine.set_subsystem_state("heater", "IDLE")

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 6: 查看系统状态")
    logger.info("─" * 70)

    # 查看 Facade 状态摘要
    status = facade.status_summary
    logger.info("[Facade 状态]")
    for key, value in status.items():
        if key != "orchestrator":  # 避免打印过多嵌套数据
            logger.info("  %s: %s", key, value)

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示 7: 取消硬件报警订阅")
    logger.info("─" * 70)

    # 取消订阅
    success = facade.unregister_hardware_alert_callback(sub_id)
    logger.info("[订阅管理] 取消订阅结果: %s", success)

    logger.info("")
    logger.info("─" * 70)
    logger.info("演示完成")
    logger.info("─" * 70)

    # ── Step 8: 优雅关闭 ── #
    logger.info("")
    await facade.shutdown()

    logger.info("")
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  演示完成 ✓                                                        ║")
    logger.info("╚" + "═" * 70 + "╝")


# --------------------------------------------------------------------------- #
#  入口点                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        asyncio.run(demo_integration_flow())
    except KeyboardInterrupt:
        logger.info("演示被用户中断")
    except Exception as exc:
        logger.exception("演示异常: %s", exc)
        sys.exit(1)
