"""
FastAPI 服务入口
提供 NL2SQL 问答 API
"""
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from chains.pipeline import NL2SQLPipeline
from chains.entity_extractor import rule_based_extract

# ─── FastAPI App ───
app = FastAPI(
    title="海缆故障 NL2SQL 问答系统",
    description="基于 Qwen3-32B + LangChain 的海缆故障自然语言查询系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 全局 Pipeline ───
pipeline = NL2SQLPipeline()
pipeline_skip_intent = NL2SQLPipeline(skip_intent=True)


# ─── 请求/响应模型 ───

class QueryRequest(BaseModel):
    query: str = Field(..., description="用户自然语言问题", min_length=1, max_length=500)
    skip_intent: bool = Field(False, description="是否跳过意图路由(调试用)")


class EntityInfo(BaseModel):
    raw: str
    standard: str
    entity_type: str
    confidence: float
    source: str


class QueryResponse(BaseModel):
    query: str = Field(..., description="原始问题")
    answer: str = Field(..., description="自然语言回答")
    intent: dict | None = Field(None, description="意图分类结果")
    entities: dict | None = Field(None, description="实体抽取结果")
    normalized_query: str = Field("", description="标准化后的问题")
    sql: str | None = Field(None, description="生成的 SQL")
    raw_result: str | None = Field(None, description="SQL 查询原始结果")
    elapsed_ms: int = Field(0, description="总耗时(毫秒)")
    error: str | None = Field(None, description="错误信息")


class EntityExtractionRequest(BaseModel):
    text: str = Field(..., description="待抽取实体的文本")


class EntityExtractionResponse(BaseModel):
    cable_names: list[dict]
    segments: list[dict]
    cities: list[dict]
    normalized_query: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── API 端点 ───

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    NL2SQL 问答接口

    完整流程: 意图路由 → 实体纠错 → Text2SQL → 执行 → 格式化回答
    """
    try:
        p = pipeline_skip_intent if request.skip_intent else pipeline
        result = p.run(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract_entities", response_model=EntityExtractionResponse)
async def extract_entities_api(request: EntityExtractionRequest):
    """
    实体抽取接口 (仅规则引擎，无需 LLM)
    用于调试和测试纠错模块
    """
    result = rule_based_extract(request.text)
    return EntityExtractionResponse(
        cable_names=[{"raw": e.raw, "standard": e.standard, "confidence": e.confidence} for e in result.cable_names],
        segments=[{"raw": e.raw, "standard": e.standard, "confidence": e.confidence} for e in result.segments],
        cities=[{"raw": e.raw, "standard": e.standard, "confidence": e.confidence} for e in result.cities],
        normalized_query=result.normalized_query,
    )


@app.get("/api/tables")
async def list_tables():
    """列出数据库表结构(调试用)"""
    from sqlalchemy import create_engine, text, inspect
    import config

    engine = create_engine(config.DB_URI)
    inspector = inspect(engine)
    tables = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        tables[table_name] = [{"name": c["name"], "type": str(c["type"])} for c in columns]
    return {"tables": tables}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
