---
name: plant_growth_monitor
description: 太空植物生长周期监测标准操作程序（包含开灯、注水、调节温湿度、拍照等周期性操作）
---

# 太空植物生长周期监测 SOP

## 适用场景
- 太空农业实验舱
- 长期任务食品补给培养
- 植物生长周期研究（发芽→幼苗→成熟→收获）

## 依赖的 MCP Tools（白名单）
- `set_light_intensity`: 调节光照强度
- `set_temperature`: 设置培养舱温度
- `set_humidity`: 调节空气湿度
- `toggle_vacuum_pump`: 控制通风/排水
- `inject_nutrient`: 注入营养液
- `activate_camera`: 激活摄像头拍摄
- `trigger_water_pump`: 触发水泵灌溉
- `set_co2_level`: 设置二氧化碳浓度

## 生长阶段定义

| 阶段 | 天数 | 光照强度 | 温度 | 湿度 | 营养液 |
|------|------|----------|------|------|--------|
| 发芽期 | Day 0-7 | 200 μmol/m²/s | 22℃ | 80% | 每3天 50ml |
| 幼苗期 | Day 8-21 | 350 μmol/m²/s | 24℃ | 70% | 每2天 80ml |
| 生长期 | Day 22-45 | 500 μmol/m²/s | 25℃ | 60% | 每天 100ml |
| 成熟期 | Day 46-60 | 400 μmol/m²/s | 23℃ | 55% | 每2天 120ml |

## Workflow

### Step 1: 环境参数初始化
1. 调用 `set_light_intensity` 设置当日光照强度
2. 调用 `set_temperature` 设置培养温度
3. 调用 `set_humidity` 调节空气湿度

### Step 2: 灌溉管理
1. 检查土壤湿度传感器
2. 如果湿度 < 40%：
   - 调用 `trigger_water_pump` 进行灌溉（`volume: 100ml`）
3. 如果湿度 > 80%：
   - 调用 `toggle_vacuum_pump` 启动通风（`action: "on"`, `mode: "vent"`）

### Step 3: 营养液补充
1. 检查营养液剩余量
2. 如果液位 < 20%：
   - 调用 `inject_nutrient` 补充营养液（`volume: 200ml`）

### Step 4: CO2 浓度调节
1. 检查 CO2 浓度
2. 如果 CO2 < 400ppm：
   - 调用 `set_co2_level` 增加 CO2（`level: 800ppm`）
3. 如果 CO2 > 1000ppm：
   - 调用 `toggle_vacuum_pump` 通风换气

### Step 5: 生长状态拍照记录
1. 调用 `activate_camera` 拍摄植物照片
   - `angle: "top_down"` - 俯视图
   - `angle: "side_view"` - 侧视图
2. 记录拍照时间戳

### Step 6: 遥测数据采集
1. 采集温度、湿度、光照、CO2 数据
2. 记录土壤湿度
3. 记录营养液液位
4. 生成每日生长报告

## 周期性任务调度

### 每日任务（每日 08:00 执行）
1. 环境参数初始化（Step 1）
2. 灌溉检查（Step 2）
3. 营养液检查（Step 3）
4. 拍照记录（Step 5）

### 每周任务（每周一执行）
1. CO2 浓度调节（Step 4）
2. 全面环境评估
3. 病虫害检查

### 每月任务（每月1日执行）
1. 全面环境参数校准
2. 营养液更换
3. 过滤器清洁

## 依赖关系

```
light_init → temperature_init → humidity_init
                    ↓
            water_check → water_pump (if needed)
                    ↓
            nutrient_check → inject_nutrient (if needed)
                    ↓
            co2_check → co2_adjust (if needed)
                    ↓
            camera_photo
                    ↓
            telemetry_report
```

## 执行约束
- **最大重试次数**：2 次
- **超时时间**：单次操作 30 秒
- **并发策略**：Step 1 内部可并行，Step 2-4 需顺序执行

## 注意事项
1. 光照强度调节需渐进式变化，避免光强骤变
2. 灌溉后需等待 30 分钟再拍照，确保水分分布均匀
3. 营养液温度需与舱内温度一致后再注入
4. 拍照时光源设置需标准化（使用补光灯）
