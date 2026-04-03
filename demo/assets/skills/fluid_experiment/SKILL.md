---
name: fluid_experiment
description: 微重力流体实验标准操作程序 (适用于空间站流体物理实验柜)
---

# 微重力流体实验 SOP

## When to use

当用户要求执行以下类型任务时，应参考此 SOP：
- 微重力环境下的流体动力学实验
- 涉及温度梯度的对流实验
- 需要同时控制温度和机械臂的复合实验

## Workflow

### Step 1: 环境准备
1. 确认实验柜当前状态为 `IDLE`
2. 调用 `set_temperature` 将舱温设置到实验初始温度（通常 25℃）
3. 等待温度稳定（约 30 秒）

### Step 2: 样品装载
1. 调用 `move_robotic_arm` 将机械臂移动到样品仓位置（角度 90°）
2. 确认机械臂到位后，调用 `toggle_vacuum_pump(activate=true)` 启动真空环境
3. 等待真空度达标

### Step 3: 实验执行
1. 调用 `set_temperature` 逐步升温至目标温度（如 50℃）
2. 调用 `move_robotic_arm` 将机械臂移至观测位置（角度 45°）
3. 记录遥测数据

### Step 4: 实验清理
1. 调用 `set_temperature` 降温回 25℃
2. 调用 `toggle_vacuum_pump(activate=false)` 关闭真空泵
3. 调用 `move_robotic_arm` 将机械臂归位（角度 0°）

## Safety Notes
- 温度不得超过 75℃（FSM 安全约束会在 80℃ 时拦截加热）
- 低压环境下（< 50kPa）禁止移动机械臂
- 每步操作后应检查 FSM 状态是否回到 IDLE

## References
- ESA Fluid Science Laboratory User Guide v3.2
- NASA ISS Experiment Operations Manual
