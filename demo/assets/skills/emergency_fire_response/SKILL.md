---
name: emergency_fire_response
description: 太空舱火情应急响应标准操作程序（包含切断电源、启动真空泵排烟、机械臂隔离危险源等原子操作）
---

# 太空舱火情应急响应 SOP

## 紧急程度
**CRITICAL** - 本 SOP 触发硬件级抢占，绕过 LLM 规划直接执行。

## 触发条件
- 烟雾传感器报警：`smoke_level > 0.8`
- 温度异常：`temperature > 150℃`
- 火焰检测：`flame_detected == true`

## 依赖的 MCP Tools（白名单）
- `cut_power`: 切断指定舱段电源
- `toggle_vacuum_pump`: 启动/关闭真空泵（排烟）
- `move_robotic_arm`: 控制机械臂移动
- `trigger_fire_extinguisher`: 启动灭火器释放
- `seal_compartment`: 密封舱段
- `activate_alarm`: 启动声光报警
- `broadcast_emergency`: 广播紧急通知
- `activate_oxygen_mask`: 启动氧气面罩

## Workflow

### Phase 1: 感知与告警（立即执行）

#### Step 1: 启动声光报警
1. 调用 `activate_alarm` 启动紧急报警
2. 确认报警状态

#### Step 2: 广播紧急撤离通知
1. 调用 `broadcast_emergency` 广播火情警报
2. 确认广播状态

### Phase 2: 电源切断（火灾响应关键）

#### Step 3: 切断火情舱段电源
1. 调用 `cut_power` 切断火情舱段电源（`section: fire_zone`）
2. 确认电源已切断
3. 记录切断时间戳

### Phase 3: 排烟通风（防止烟雾扩散）

#### Step 4: 启动真空泵排烟
1. 调用 `toggle_vacuum_pump` 启动真空泵（`action: "on"`, `mode: "exhaust"`）
2. 等待排烟完成（烟雾浓度 < 0.1）
3. 确认排烟效果

### Phase 4: 灭火处置

#### Step 5: 启动灭火器
1. 调用 `trigger_fire_extinguisher` 释放灭火剂（`agent: "foam"`, `zone: fire_zone`）
2. 确认灭火剂已释放
3. 等待 30 秒观察火情

### Phase 5: 机械臂隔离（危险源处置）

#### Step 6: 机械臂转移危险物品
1. 调用 `move_robotic_arm` 将机械臂移至火源位置（`target_position: "fire_source"`, `grip: true`）
2. 调用 `move_robotic_arm` 将危险物品移至隔离舱（`target_position: "quarantine_bay"`）
3. 调用 `move_robotic_arm` 将机械臂归零（`target_position: "home"`）

### Phase 6: 舱段隔离（防止蔓延）

#### Step 7: 密封火情舱段
1. 调用 `seal_compartment` 密封火情舱段（`section: fire_zone`, `mode: "hermetic"`）
2. 确认密封状态

### Phase 7: 人员防护

#### Step 8: 启动氧气面罩
1. 调用 `activate_oxygen_mask` 在相邻舱段启动氧气面罩（`sections: ["adjacent_zones"]`）
2. 确认氧气供应正常

### Phase 8: 状态上报

#### Step 9: 上报应急响应状态
1. 汇总各步骤执行结果
2. 记录总响应时间
3. 生成应急响应报告

## 依赖关系图

```
alarm → broadcast → cut_power
                      ↓
              vacuum_pump (排烟)
                      ↓
            fire_extinguisher (灭火)
                      ↓
            robotic_arm (隔离危险源)
                      ↓
            seal_compartment (密封)
                      ↓
            oxygen_mask (人员防护)
```

## 执行约束
- **最大重试次数**：3 次（关键安全操作不轻易放弃）
- **超时时间**：各步骤独立超时，整体响应需在 120 秒内完成
- **并发策略**：Phase 1 必须串行执行，Phase 2-4 可部分并行

## 注意事项
1. 切断电源前需确认无人员在危险区域
2. 启动真空泵前需确认舱门已关闭
3. 机械臂操作需避开明火区域
4. 灭火器释放后需通风检测空气质量
