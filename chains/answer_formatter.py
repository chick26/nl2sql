"""
结果格式化 Chain + SQL 校验与执行
"""
import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from llm_factory import create_chat_llm
from prompts.answer import ANSWER_PROMPT, REJECT_PROMPT

# ─── LLM (no_think 模式) ───
llm_no_think = create_chat_llm(thinking=False, temperature=0.1)

# ─── 数据库引擎 ───
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URI, echo=False)
    return _engine


# ──────────────────────── SQL 执行 ────────────────────────

def execute_sql(sql: str) -> str:
    """
    执行 SQL 并返回格式化结果
    """
    engine = _get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()

        if not rows:
            return "查询结果为空，未找到匹配的记录。"

        # 格式化为表格字符串
        lines = []
        # 表头
        lines.append(" | ".join(str(c) for c in columns))
        lines.append("-" * len(lines[0]))
        # 数据行 (最多显示 50 行)
        for row in rows[:50]:
            lines.append(" | ".join(str(v) if v is not None else "N/A" for v in row))

        if len(rows) > 50:
            lines.append(f"... 共 {len(rows)} 条记录 (仅显示前50条)")
        else:
            lines.append(f"共 {len(rows)} 条记录")

        return "\n".join(lines)

    except Exception as e:
        return f"SQL 执行错误: {str(e)}"


def validate_and_execute(data: dict) -> dict:
    """校验并执行 SQL，返回结果"""
    sql = data.get("sql", "")
    query = data.get("query", "")

    if not sql:
        return {**data, "result": "未能生成有效的 SQL 查询。", "sql_error": True}

    # 安全校验
    sql_upper = sql.upper()
    for kw in config.FORBIDDEN_SQL_KEYWORDS:
        if kw in sql_upper:
            return {**data, "result": f"SQL 安全校验失败: 包含禁止的操作 {kw}", "sql_error": True}

    # 执行
    result = execute_sql(sql)
    return {**data, "result": result, "sql_error": "执行错误" in result}


# ──────────────────────── 回答格式化 Chain ────────────────────────

def _prepare_answer_input(data: dict) -> dict:
    return {
        "query": data.get("query", ""),
        "sql": data.get("sql", ""),
        "result": data.get("result", ""),
    }


answer_formatting_chain = (
    RunnableLambda(_prepare_answer_input)
    | ANSWER_PROMPT
    | llm_no_think
    | StrOutputParser()
)

# ─── 拒绝回答 Chain ───
reject_chain = (
    REJECT_PROMPT
    | llm_no_think
    | StrOutputParser()
)


def format_answer(query: str, sql: str, result: str) -> str:
    """便捷调用函数"""
    return answer_formatting_chain.invoke({
        "query": query,
        "sql": sql,
        "result": result,
    })


if __name__ == "__main__":
    # 测试 SQL 执行
    test_sql = f"SELECT pop_sys, pop_fault_seg, repair_status_name, pop_fault_time FROM {config.FAULT_TABLE} WHERE repair_status_name != '已结束' LIMIT 5"
    print(f"测试 SQL: {test_sql}")
    result = execute_sql(test_sql)
    print(f"结果:\n{result}\n")

    test_sql2 = f"SELECT pop_sys, COUNT(*) as cnt FROM {config.FAULT_TABLE} GROUP BY pop_sys ORDER BY cnt DESC LIMIT 10"
    print(f"统计 SQL: {test_sql2}")
    result2 = execute_sql(test_sql2)
    print(f"结果:\n{result2}")
