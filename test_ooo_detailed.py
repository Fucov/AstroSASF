#!/usr/bin/env python
"""
测试 OoO 越级调度的核心场景：
1. 创建 DAG：L0 两个节点并行（heater:10s, arm:1s）
2. L1 两个节点分别依赖 L0 的两个节点
3. 验证：当 arm(1s) 完成后，OoO 能立即越级执行依赖 arm 的 L1 节点
"""

import asyncio
import sys
sys.path.insert(0, '/root/AstroSASF')

from scheduler.core import DAGNode, DAGOrchestrator, DAGTaskGraph, NodeStatus, TaskPriority
from scheduler.device_runtime import DeviceRuntime, ChaosEngine
from scheduler.device_model import DEFAULT_DEVICE_REGISTRY
from scheduler.metrics_collector import MetricsCollector

class TestLab:
    def __init__(self, lab_id, runtime):
        self.lab_id = lab_id
        self.runtime = runtime
        self.execution_log = []
        
    async def run_single_task(self, task_id, skill_name, params, required_devices, task_priority, blocking=True):
        self.execution_log.append(f"START: {task_id} ({skill_name})")
        result = await self.runtime.invoke(
            device_id=required_devices[0] if required_devices else 'heater_bio',
            action=skill_name,
            params=params,
            task_id=task_id,
            lab_id=self.lab_id,
            cabin_id=self.lab_id,
            telemetry_snapshot={},
            blocking=blocking,
        )
        self.execution_log.append(f"END: {task_id}")
        return result

async def test_ooo_scenario():
    print("=" * 60)
    print("测试 OoO 越级调度场景")
    print("=" * 60)
    
    # 创建 DeviceRuntime（使用较长延迟）
    runtime = DeviceRuntime(
        device_registry=dict(DEFAULT_DEVICE_REGISTRY),
        chaos=ChaosEngine(seed=42),
        seed=42,
        physical_delay_scale=1.0,
    )
    
    lab = TestLab('DemoBio', runtime)
    
    # 创建 DAG
    # L0: heater(10s) + arm(1s) 并行
    # L1: task_depends_heater(依赖heater) + task_depends_arm(依赖arm)
    dag = DAGTaskGraph(graph_id='test', name='Test OoO')
    
    heater = DAGNode(
        node_id='heater',
        skill_name='control_heater',
        params={'temperature': 50.0, 'duration': 10.0},
        dependencies=[],
        priority=TaskPriority.NORMAL,
        description='Heater 10s',
        lab_id='DemoBio',
    )
    
    arm = DAGNode(
        node_id='arm',
        skill_name='move_robotic_arm',
        params={'target_position': 'HOME'},
        dependencies=[],
        priority=TaskPriority.NORMAL,
        description='Arm 1s',
        lab_id='DemoBio',
    )
    
    task_heater = DAGNode(
        node_id='task_after_heater',
        skill_name='toggle_vacuum_pump',
        params={'activate': True},
        dependencies=['heater'],
        priority=TaskPriority.NORMAL,
        description='After heater',
        lab_id='DemoBio',
    )
    
    task_arm = DAGNode(
        node_id='task_after_arm',
        skill_name='toggle_vacuum_pump',
        params={'activate': True},
        dependencies=['arm'],
        priority=TaskPriority.NORMAL,
        description='After arm',
        lab_id='DemoBio',
    )
    
    for node in [heater, arm, task_heater, task_arm]:
        dag.add_node(node)
    
    dag.add_edge('heater', 'task_after_heater')
    dag.add_edge('arm', 'task_after_arm')
    
    print("\nDAG 结构:")
    print("  L0: heater(10s) || arm(1s)")
    print("  L1: task_heater(依赖heater) || task_arm(依赖arm)")
    print()
    
    # 执行
    orch = DAGOrchestrator(max_workers=3)
    orch.register_lab(lab)
    
    print("开始执行...")
    await orch.start()
    await orch.submit_dag(dag)
    
    try:
        await asyncio.wait_for(orch._dag_complete_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("超时!")
    
    await orch.shutdown()
    
    print("\n执行日志:")
    for log in lab.execution_log:
        print(f"  {log}")
    
    print(f"\n总时间: {dag.get_completed_count()} 节点完成")
    print(f"OoO 越级次数: {orch._ooo_execution_count}")
    
    # 分析
    print("\n" + "=" * 60)
    print("分析:")
    print("=" * 60)
    
    # 检查 task_after_arm 是否在 heater 之前完成
    arm_end_idx = None
    heater_end_idx = None
    task_arm_start_idx = None
    
    for i, log in enumerate(lab.execution_log):
        if 'END: arm' in log:
            arm_end_idx = i
        if 'END: heater' in log:
            heater_end_idx = i
        if 'START: task_after_arm' in log:
            task_arm_start_idx = i
    
    if arm_end_idx and task_arm_start_idx and arm_end_idx < task_arm_start_idx:
        print(f"✅ task_after_arm 在 arm 完成后立即开始（越级成功）")
    else:
        print(f"❌ task_after_arm 没有越级执行")
    
    print(f"\narm 结束索引: {arm_end_idx}")
    print(f"task_after_arm 开始索引: {task_arm_start_idx}")

if __name__ == '__main__':
    asyncio.run(test_ooo_scenario())
