> 1. 明确问题边界，针对海缆故障现状、进度及统计分析类的问题进行回答，隔离复杂关联，趋势预测问题
> 2. 海缆数据结构先工程化，明确最小字段集，优化海缆故障数据库，通过实时清洗入库的方式统一命名，稳定字段，针对字段实际类型进行格式统一
> 3. 基于当前 Text2SQL 路径，新增关键词/条件查询路径，优化关键词命中路径，提高准确率

1. # 数据现状

> 故障数据为人工录入

1. ## 字段说明

| 字段                              | 含义                                                         | 保留 | 样例                                                         | 备注                                                         |
| --------------------------------- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fault_id                          | 故障ID                                                       | 否   |                                                              |                                                              |
| affect_business                   | 是否影响业务                                                 | 是   |                                                              |                                                              |
| affect_info                       | 影响的中继、电路、带宽                                       | 是   | {"relayNum":16,"affectNum":31,"rate":"506.85200000000003G"}  |                                                              |
| business_break_time               | 业务中断时间                                                 | 否   |                                                              |                                                              |
| create_time                       | 数据记录创建时间                                             | 是   |                                                              |                                                              |
| is_delete                         | 是否删除标识                                                 | 否   | 保留 is_delete=0                                             |                                                              |
| other_info                        | 故障历时、影响方向、处理过程                                 | 是   | {"faultDuration":"30天17小时28分钟","sectionDir":"East Coast-Shima ;East Coast-Kuantan ;East Coast-Busan ;East Coast-Busan ;East Coast-NanHui ;East Coast-Chongming ;East Coast-Tseung Kwan ;East Coast-Danang ;East Coast-Shinmaruyama ;East Coast-Songkhla ;East Coast-Toucheng ;","opretionProcess":"暂无"} | 影响方向为系统关联，后续通过另外的表维护，不需要此字段       |
| pop_fault_reason                  | 故障原因代码: 0-未知、1-人为（如打鱼）、2-设备、3-天气       | 否   |                                                              |                                                              |
| pop_fault_reason_name             | 故障原因翻译值                                               | 否   |                                                              |                                                              |
| pop_fault_seg                     | 故障光缆段编号                                               | 是   |                                                              | 后续通过静态关系表维护                                       |
| pop_fault_seg_detail              | 故障光缆段名称                                               | 是   |                                                              | 后续通过静态关系表维护                                       |
| pop_fault_time                    | 故障发生/发现时间                                            | 是   |                                                              | 故障开始时间/中断时间等唯一口径                              |
| pop_repair_charge_man             | 维修负责人                                                   | 是   |                                                              |                                                              |
| pop_repair_plane                  | 维修计划                                                     | 否   |                                                              | 24年3月之后无新数据                                          |
| pop_sys                           | 海缆名称                                                     | 是   |                                                              |                                                              |
| pop_type                          | 海缆类型                                                     | 是   |                                                              |                                                              |
| repair_status                     | 维修状态代码                                                 | 是   |                                                              | 与状态名保留一个                                             |
| repair_status_name                | 维修状态名称                                                 | 是   |                                                              |                                                              |
| repair_type                       | 维修类型                                                     | 否   |                                                              |                                                              |
| reserved_field                    | 通报时间                                                     | 否   |                                                              |                                                              |
| seg_data                          | 海缆列表                                                     | 否   |                                                              |                                                              |
| send_email_type                   | 发送邮件标识 ： 1-未发送 2-已发送新增邮件 3-已发送修复邮件 4-已发送一周总结 | 否   |                                                              |                                                              |
| update_time                       | 数据更新时间                                                 | 否   |                                                              |                                                              |
| estimated_repair_completion_time  | 预计维修完成时间                                             | 否   |                                                              |                                                              |
| estimated_ship_departure_time     | 预计维修船出发时间                                           | 否   |                                                              |                                                              |
| repair_progress                   | 维修进展                                                     | 是   |                                                              | 保留，这个是最新的修复进展，故障修复前可以用，但是当故障修复完成后这个字段不对再更新了就不要再看这个，最新进展就是已修复。 |
| actual_ship_departure_time        | 实际船只出发时间                                             | 否   |                                                              | 25年1月后无数据                                              |
| affect_direction                  | 影响方向（从other字段拆解出来的）                            | 是   |                                                              | 后续通过静态关系表维护                                       |
| arrival_service_area_time         | 到达维修区域时间                                             | 否   |                                                              |                                                              |
| business_recovery_reason          | 业务恢复说明                                                 | 否   |                                                              |                                                              |
| business_recovery_time            | 业务恢复时间                                                 | 否   |                                                              |                                                              |
| confirm_repair_plan_time          | 确认修复计划时间                                             | 否   |                                                              |                                                              |
| confirm_repair_ship_time          | 确认修复船只时间                                             | 否   |                                                              |                                                              |
| estimated_confirm_plan_time       | 预计确认修复计划时间                                         | 否   |                                                              |                                                              |
| estimated_maintenance_permit_time | 预计获得维修许可时间                                         | 否   |                                                              |                                                              |
| latitude                          | 纬度                                                         | 否   |                                                              |                                                              |
| longitude                         | 经度                                                         | 否   |                                                              |                                                              |
| obtain_repair_permit_time         | 获得维修许可时间                                             | 否   |                                                              |                                                              |
| pop_repair_boat                   | POP联系组织                                                  | 是   |                                                              |                                                              |
| pop_repair_dyn_info               | 船只动态信息                                                 | 否   |                                                              |                                                              |
| pop_repair_remark                 | 故障备注或说明                                               | 是   |                                                              | 保留，备注字段，主要记录影响业务情况                         |
| repair_boat                       | 维修船名称                                                   | 是   |                                                              |                                                              |
| repair_done_time                  | 故障修复完成时间                                             | 是   |                                                              |                                                              |
| request_maintenance_permit_time   | 申请维修许可时间                                             | 否   |                                                              |                                                              |
| start_maintenance_operations_time | 开始维修作业时间                                             | 否   |                                                              |                                                              |
| timestamp                         | 修复计划时间                                                 | 否   |                                                              |                                                              |

1. ## 关键数据（MVP）

| 字段                  | 含义                              | 样例                                                         | 备注                                                         |
| --------------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| affect_business       | 是否影响业务                      |                                                              |                                                              |
| affect_info           | 影响的中继、电路、带宽            | {"relayNum":16,"affectNum":31,"rate":"506.85200000000003G"}  |                                                              |
| create_time           | 数据记录创建时间                  |                                                              |                                                              |
| other_info            | 故障历时、影响方向、处理过程      | {"faultDuration":"30天17小时28分钟","sectionDir":"East Coast-Shima ;East Coast-Kuantan ;East Coast-Busan ;East Coast-Busan ;East Coast-NanHui ;East Coast-Chongming ;East Coast-Tseung Kwan ;East Coast-Danang ;East Coast-Shinmaruyama ;East Coast-Songkhla ;East Coast-Toucheng ;","opretionProcess":"暂无"} | 影响方向为系统关联，后续通过另外的表维护，不需要此字段       |
| pop_fault_seg         | 故障光缆段编号                    |                                                              | 后续通过静态关系表维护                                       |
| pop_fault_seg_detail  | 故障光缆段名称                    |                                                              | 后续通过静态关系表维护                                       |
| pop_fault_time        | 故障发生/发现时间                 |                                                              | 故障开始时间/中断时间等唯一口径                              |
| pop_repair_charge_man | 维修负责人                        |                                                              |                                                              |
| pop_sys               | 海缆名称                          |                                                              |                                                              |
| pop_type              | 海缆类型                          |                                                              |                                                              |
| repair_status         | 维修状态代码                      |                                                              | 与状态名保留一个                                             |
| repair_status_name    | 维修状态名称                      |                                                              |                                                              |
| repair_progress       | 维修进展                          |                                                              | 保留，这个是最新的修复进展，故障修复前可以用，但是当故障修复完成后这个字段不对再更新了就不要再看这个，最新进展就是已修复。 |
| affect_direction      | 影响方向（从other字段拆解出来的） |                                                              | 后续通过静态关系表维护                                       |
| pop_repair_boat       | POP联系组织                       |                                                              |                                                              |
| pop_repair_remark     | 故障备注或说明                    |                                                              | 保留，备注字段，主要记录影响业务情况                         |
| repair_boat           | 维修船名称                        |                                                              |                                                              |
| repair_done_time      | 故障修复完成时间                  |                                                              |                                                              |

1. ## 海缆关系表

1. # 问题边界明确

1. ## 整体问题域定义

**问题域：** 👉 *海底通信光缆（海缆）故障的运行态势、影响评估与统计分析*

**核心对象：**

- 海缆系统（如 AAE-1、APG、APCN2、TPE、SMW4/5 等）
- 故障事件（发生、维修、恢复、影响）

**核心目标：**

- 判断**是否故障**
- 判断**是否影响业务**
- 判断**是否已修复 / 何时修复**
- 对**时间、次数、范围**进行统计与汇总
- 输出**标准化故障报告**

1. ## 问题边界拆分

### ① 故障状态类（What is happening）

**关注点：**

- 是否存在故障
- 当前是否仍在故障中

**典型问题边界：**

- 某条海缆目前有什么故障？
- 现在有多少条海缆故障？
- 当前海缆故障清单？

✅ **只关心“有没有故障”，不涉及影响与维修细节**

### ② 业务影响类（Impact）

**关注点：**

- 是否影响业务
- 是否造成中断
- 是否仍在影响中

**典型问题边界：**

- 某段故障是否影响业务？
- 目前影响业务的海缆故障有多少？
- 影响业务且未修复的故障有多少？
- 是否影响某条具体业务路径（如 香港–新加坡）

✅ **边界清晰：只讨论“业务是否受影响”，不要求技术原因**

### ③ 故障时间类（When）

**关注点：**

- 发生时间
- 中断时间
- 时间区间内的统计

**典型问题边界：**

- 最新故障发生时间
- 最新业务中断时间
- 昨天 / 上周 / 指定日期内发生了几次故障
- 某时间段内的故障清单 / 数量

✅ **时间是主维度，海缆是次维度**

### ④ 故障段落 / 区段类（Where）

**关注点：**

- 故障发生在哪一段（S1.8、S1.9、S3、S1S 等）

**典型问题边界：**

- 最新故障段落
- 指定段落的故障情况
- 某段落的维修状态与计划

✅ **精确到“海缆 + 段号”，不泛化到整条海缆**

### ⑤ 维修与恢复类（Fix status）

**关注点：**

- 是否在维修
- 维修进展
- 预计修复时间 / 计划

**典型问题边界：**

- 故障维修状态
- 最新维修计划
- 什么时候修复？

✅ **关注“处置状态”，不关注历史统计**

### ⑥ 报告与统计分析类（Analysis & Output）

**关注点：**

- 汇总
- 对比
- 输出标准化结果

**典型问题边界：**

- 生成某条海缆故障报告
- 今年故障次数最多的海缆
- 某时间段内的故障数量统计

✅ **这是“二次加工问题”，依赖前五类数据**

1. # 海缆数据清洗

1. ## 数据筛选

- 保留 2025 年之后的数据
- 拓宽原表，增加地址映射台账

1. # 技术方案

1. ## 关键词

- 海缆名称（AAE1 / AAE-1 / AAE_1）
- 缆段名称（S1-8 / S1.8 / Segment 1.8）
- 城市 / 站点（香港 / HK / HongKong）

1. ## 整体架构

```Plain
用户问题
   ↓
关键词识别 & 实体抽取
   ↓
实体纠错 / 规范化（重点）
   ↓
标准化问题
   ↓
问题边界识别
   ↓
Text2SQL
   ↓
执行 & 返回
```

**关键词模块在最前面，且不直接生成 SQL**

1. ## 纠错模块设计

1. ### 关键词类型划分

| 实体类型    | 示例               |
| ----------- | ------------------ |
| 海缆名称    | AAE-1、APG、SMW4   |
| 缆段名称    | S1.8、S3、S1S      |
| 地点 / 城市 | 香港、新加坡、东京 |

1. ### 标准词典

```JSON
{
  "type": "cable_name",
  "standard": "AAE-1",
  "aliases": ["AAE1", "AAE_1", "Asia Africa Europe-1"]
}
{
  "type": "segment",
  "standard": "S1.8",
  "aliases": ["S1-8", "Segment1.8", "S1_8"]
}
{
  "type": "city",
  "standard": "Hong Kong",
  "aliases": ["香港", "HK", "HongKong"]
}
```

1. ### 实体抽取策略

- 推荐组合：
  - **正则 / 模式优先** S\d+(.\d+)?
- 常见海缆缩写规则
  - **词典匹配（模糊）**alias → standard
  - **LLM 补充**  仅用于兜底，不用于最终裁决

1. ### 纠错流程示例

- 用户输入

> “AAE1 在 S1-8 段现在有故障吗？”

- 纠错后

```JSON
{
  "cable_name": "AAE-1",
  "segment": "S1.8"
}
```

1. ## 关键词纠错 vs Text2SQL 的职责边界

| 模块       | 负责什么           | 不负责什么   |
| ---------- | ------------------ | ------------ |
| 关键词纠错 | 实体识别、名称统一 | 判断是否故障 |
| Text2SQL   | 条件、时间、统计   | 名称猜测     |
| SQL        | 精确查询           | 语义推理     |

1. ## 纠错结果

返回**可解释结构**：

```JSON
{
  "normalized_query": "AAE-1 在 S1.8 段现在有故障吗？",
  "entities": {
    "cable_name": {
      "raw": "AAE1",
      "standard": "AAE-1",
      "confidence": 0.92
    },
    "segment": {
      "raw": "S1-8",
      "standard": "S1.8",
      "confidence": 0.97
    }
  }
}
```