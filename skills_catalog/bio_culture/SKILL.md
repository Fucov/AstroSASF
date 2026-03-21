---
name: bio_culture
description: 太空微重力环境下生物细胞培养标准操作程序
---

# 太空生物细胞培养 SOP

## Workflow

### Step 1: 培养舱环境准备
1. 调用 `set_temperature` 设温到 37℃（细胞最佳培养温度）
2. 等待温度稳定

### Step 2: 培养舱密封
1. 调用 `toggle_vacuum_pump` 关闭真空泵（确保常压环境）
2. 确认舱内压力正常

### Step 3: 营养液注入
1. 调用 `inject_nutrient` 注入培养基营养液
2. 记录注入量

### Step 4: 机械臂装载样品
1. 调用 `move_robotic_arm` 将机械臂移至对接位（90°）装载培养皿
2. 调用 `move_robotic_arm` 将机械臂归零

### Step 5: 启动培养监控
1. 调用 `set_temperature` 维持 37℃ 恒温
2. 周期性记录遥测数据
