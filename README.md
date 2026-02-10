# 海缆故障 NL2SQL 问答系统

基于 Qwen3-32B + LangChain 的海底光缆故障自然语言查询系统。用户用自然语言提问，系统自动识别意图、纠正实体、生成 SQL、查询数据库并返回中文回答。

## 系统架构

```
用户问题
  ↓
意图路由 (Qwen3-32B thinking)       ← 判断问题是否在 6 类边界内
  ↓
实体抽取与纠错 (规则引擎 + NER + LLM) ← AAE1 → AAE-1, S1-8 → S1.8
  ↓
Text2SQL (Qwen3-32B no_think)        ← 生成 SELECT 语句
  ↓
SQL 校验 & 执行                       ← 查询 SQLite
  ↓
结果格式化 (Qwen3-32B no_think)      ← 转为自然语言回答
```

## 前置条件

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器
- Qwen3-32B 推理服务（OpenAI 兼容 API，默认 `http://10.120.84.7:8001/v1`）

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

如需训练 NER 小模型（需要 GPU）：

```bash
uv sync --extra train
```

### 2. 数据初始化

首次使用需要清洗 CSV 并导入 SQLite，同时构建实体别名词典：

```bash
# 清洗故障数据 + 段落关系表，写入 data/sea_cable.db
uv run python etl/clean_fault_data.py

# 从关系表自动生成别名词典 (海缆/缆段/城市)
uv run python etl/build_dictionary.py
```

执行后会在 `data/` 下生成：

| 文件 | 说明 |
|------|------|
| `sea_cable.db` | SQLite 数据库（268 条故障 + 463 条段落关系） |
| `dictionaries/cable_aliases.json` | 8 个海缆标准名 + 29 个别名 |
| `dictionaries/segment_aliases.json` | 78 个缆段标准名 + 465 个别名 |
| `dictionaries/city_aliases.json` | 64 个城市标准名 + 141 个别名 |

### 3. 启动服务

```bash
uv run python server.py
```

服务启动后监听 `http://0.0.0.0:8080`，API 文档访问 `http://localhost:8080/docs`。

## API 接口

### 问答接口

```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "APG 海缆现在有什么故障？"}'
```

响应示例：

```json
{
  "query": "APG 海缆现在有什么故障？",
  "answer": "APG 海缆目前有 1 起未修复的故障...",
  "intent": {"intent": "fault_status", "confidence": 0.95},
  "entities": {"cable_names": [{"raw": "APG", "standard": "APG"}]},
  "sql": "SELECT * FROM sea_cable_fault WHERE pop_sys = 'APG' AND repair_status_name != '已结束'",
  "elapsed_ms": 3200
}
```

跳过意图路由（调试用，直接进入 NL2SQL 流程）：

```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "当前未修复的故障有哪些？", "skip_intent": true}'
```

### 实体抽取接口

纯规则引擎，不依赖 LLM，可用于调试纠错效果：

```bash
curl -X POST http://localhost:8080/api/extract_entities \
  -H "Content-Type: application/json" \
  -d '{"text": "AAE1 在 S1-8 段有故障吗"}'
```

响应：

```json
{
  "cable_names": [{"raw": "AAE1", "standard": "AAE-1", "confidence": 0.95}],
  "segments": [{"raw": "S1-8", "standard": "S1.8", "confidence": 0.92}],
  "cities": [],
  "normalized_query": "AAE-1 在 S1.8 段有故障吗"
}
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/tables` | 查看数据库表结构 |
| GET | `/docs` | Swagger UI 交互文档 |

## 环境变量

通过环境变量覆盖默认配置（定义在 `config.py`）。  
支持本地 `.env.local` 文件（仅本地使用，不提交）：

```bash
cp .env.local.example .env.local
```

`.env.local` 会在启动时被自动读取（不会覆盖已存在的系统环境变量）。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `openai_compat` | 大模型提供方：`openai_compat`/`openai`/`azure` |
| `LLM_BASE_URL` | `http://10.120.84.7:8001/v1` | Qwen3-32B API 地址 |
| `LLM_MODEL_NAME` | `Qwen/Qwen3-32B` | 模型名称 |
| `LLM_API_KEY` | `test` | API Key |
| `LLM_SUPPORTS_THINKING` | `true` | 是否支持 thinking 模式（不支持可设为 `false`） |
| `AZURE_OPENAI_ENDPOINT` | 空 | Azure OpenAI 端点 |
| `AZURE_OPENAI_API_KEY` | 空 | Azure OpenAI Key |
| `AZURE_OPENAI_API_VERSION` | 空 | Azure OpenAI API 版本 |
| `AZURE_OPENAI_DEPLOYMENT` | 空 | Azure OpenAI 部署名 |

示例（OpenAI 兼容接口）：

```bash
LLM_BASE_URL=http://your-server:8001/v1 uv run python server.py
```

示例（OpenAI 官方）：

```bash
LLM_PROVIDER=openai LLM_MODEL_NAME=gpt-4o-mini LLM_API_KEY=your-key uv run python server.py
```

示例（Azure OpenAI）：

```bash
LLM_PROVIDER=azure \
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com \
AZURE_OPENAI_API_KEY=your-key \
AZURE_OPENAI_API_VERSION=2024-02-15-preview \
AZURE_OPENAI_DEPLOYMENT=your-deployment \
uv run python server.py
```

## 支持的问题类型

| 类型 | 示例 |
|------|------|
| 故障状态 | "APG 海缆现在有什么故障？"、"当前海缆故障清单？" |
| 业务影响 | "目前影响业务的故障有多少？"、"AAE-1 故障影响了香港吗？" |
| 故障时间 | "最近一周有几起海缆故障？"、"今年 APG 故障了几次？" |
| 故障段落 | "AAE-1 的 S1.8 段有故障吗？"、"S3 段修复了吗？" |
| 维修状态 | "APG 的故障修了吗？"、"最新的维修进展是什么？" |
| 统计分析 | "今年故障次数最多的海缆是哪个？"、"生成 APG 故障报告" |

## NER 小模型训练（可选）

系统默认使用规则引擎进行实体纠错，效果已经可用。如需进一步提升纠错能力，可训练 BERT NER 模型。

### 方案 A：BERT-CRF NER（推荐）

```bash
# 安装训练依赖
uv sync --extra train

# 生成 5000 条 BIO 标注数据
uv run python models/ner/generate_training_data.py

# 训练 (需要 GPU，约 30 分钟)
uv run python models/ner/train.py
```

训练完成后模型保存在 `models/ner/saved_model/`，系统会自动加载。

### 方案 B：Qwen3-0.6B LoRA 微调（备选）

```bash
# 安装 LoRA 依赖
uv sync --extra lora

# 生成纠错训练数据
uv run python models/correction/finetune.py --action generate

# 微调 (需要 GPU)
uv run python models/correction/finetune.py --action train
```

## 项目结构

```
nl2sql/
├── config.py                   # 全局配置
├── server.py                   # FastAPI 服务入口
├── pyproject.toml              # uv 项目配置与依赖
├── chains/                     # LangChain 链模块
│   ├── intent_router.py        #   意图路由 (thinking 模式)
│   ├── entity_extractor.py     #   三级实体纠错
│   ├── sql_generator.py        #   Text2SQL (no_think 模式)
│   ├── answer_formatter.py     #   SQL 执行 + 结果格式化
│   └── pipeline.py             #   完整 Pipeline 编排
├── prompts/                    # Prompt 模板
│   ├── intent.py               #   意图分类
│   ├── text2sql.py             #   SQL 生成
│   └── answer.py               #   回答格式化
├── etl/                        # 数据工程
│   ├── clean_fault_data.py     #   CSV 清洗入库
│   └── build_dictionary.py     #   别名词典构建
├── models/                     # 小模型
│   ├── ner/                    #   BERT NER (方案A)
│   └── correction/             #   Qwen3-0.6B LoRA (方案B)
└── data/
    ├── sea_cable.db            # SQLite 数据库
    └── dictionaries/           # 别名词典 JSON
```
