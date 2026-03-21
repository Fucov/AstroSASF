---
name: material_synthesis
description: 微重力环境下合金材料合成标准操作程序
---

# 微重力合金材料合成 SOP

## Workflow

### Step 1: 真空环境构建
1. 调用 `toggle_vacuum_pump` 激活真空泵（合金合成需要真空环境）
2. 等待舱内达到低压状态

### Step 2: 加热至合成温度
1. 调用 `set_temperature` 设温到 70℃（合金熔融预热温度）
2. 等待温度稳定

### Step 3: 激光烧结启动
1. 调用 `turn_on_laser` 开启激光烧结设备
2. 调用 `set_temperature` 精确控制至 65℃ 维持合成温度

### Step 4: 样品操作
1. 调用 `move_robotic_arm` 将机械臂移至观测位（45°）进行原位观测
2. 调用 `move_robotic_arm` 将机械臂归零

### Step 5: 降温与环境恢复
1. 调用 `turn_on_laser` 关闭激光设备
2. 调用 `set_temperature` 降温到 25℃
3. 调用 `toggle_vacuum_pump` 关闭真空泵恢复常压
