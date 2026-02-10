"""
Text2SQL 生成 Chain
使用 Qwen3-32B no_think 模式生成 SQL
"""
import os
import re
import sys
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from prompts.text2sql import TEXT2SQL_PROMPT
from chains.entity_extractor import ExtractionResult

# ─── 不启用 thinking 的 LLM (追求速度) ───
llm_no_think = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    model=config.LLM_MODEL_NAME,
    api_key=config.LLM_API_KEY,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)


def _clean_sql(text: str) -> str:
    """清理 LLM 输出中的 SQL"""
    text = text.strip()
    # 移除 markdown 代码块
    if "```sql" in text:
        text = text.split("```sql")[-1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # 移除末尾分号
    text = text.rstrip(";").strip()

    # 基本安全校验
    text_upper = text.upper()
    for kw in config.FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", text_upper):
            raise ValueError(f"SQL 包含禁止的关键词: {kw}")

    # 验证是 SELECT 语句
    if not text_upper.lstrip().startswith("SELECT"):
        raise ValueError(f"只允许 SELECT 语句，收到: {text[:50]}")

    return text


def _prepare_sql_input(data: dict) -> dict:
    """准备 Text2SQL 的输入"""
    entities_result: ExtractionResult = data.get("entities_result")
    query = data.get("query", "")

    entities_str = "无特定实体"
    normalized_query = query

    if entities_result:
        entities_str = entities_result.entities_summary()
        normalized_query = entities_result.normalized_query or query

    return {
        "normalized_query": normalized_query,
        "entities": entities_str,
        "today": date.today().isoformat(),
    }


# ─── Text2SQL Chain ───
sql_generation_chain = (
    RunnableLambda(_prepare_sql_input)
    | TEXT2SQL_PROMPT
    | llm_no_think
    | StrOutputParser()
    | RunnableLambda(_clean_sql)
)


def generate_sql(query: str, entities_result: ExtractionResult = None) -> str:
    """便捷调用函数"""
    return sql_generation_chain.invoke({
        "query": query,
        "entities_result": entities_result,
    })


if __name__ == "__main__":
    from chains.entity_extractor import rule_based_extract

    test_queries = [
        "APG 海缆现在有什么故障？",
        "当前影响业务的故障有多少？",
        "AAE-1 的 S1.8 段修了吗？",
    ]
    for q in test_queries:
        entities = rule_based_extract(q)
        try:
            sql = generate_sql(q, entities)
            print(f"Q: {q}")
            print(f"SQL: {sql}\n")
        except Exception as e:
            print(f"Q: {q}")
            print(f"Error: {e}\n")
