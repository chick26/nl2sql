"""
完整 NL2SQL Pipeline
使用 LangChain LCEL 编排:
  用户问题 → 意图路由 → 实体纠错 → Text2SQL → SQL执行 → 回答格式化
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chains.intent_router import classify_intent
from chains.entity_extractor import extract_entities, ExtractionResult
from chains.sql_generator import generate_sql
from chains.answer_formatter import validate_and_execute, format_answer, reject_chain


class NL2SQLPipeline:
    """
    海缆故障 NL2SQL 问答 Pipeline

    流程:
    1. 意图路由 (Qwen3-32B thinking) → 判断问题边界
    2. 实体抽取与纠错 (规则 + NER + LLM 三级)
    3. Text2SQL (Qwen3-32B no_think) → 生成 SQL
    4. SQL 校验 & 执行 → 查询数据库
    5. 结果格式化 (Qwen3-32B no_think) → 自然语言回答
    """

    def __init__(self, skip_intent: bool = False):
        """
        Args:
            skip_intent: 跳过意图路由(调试用)，直接进入 NL2SQL 流程
        """
        self.skip_intent = skip_intent

    def run(self, query: str) -> dict:
        """
        执行完整 NL2SQL 流程

        Args:
            query: 用户自然语言问题

        Returns:
            {
                "query": 原始问题,
                "intent": 意图分类结果,
                "entities": 实体抽取结果,
                "normalized_query": 标准化后的问题,
                "sql": 生成的 SQL,
                "raw_result": SQL 查询原始结果,
                "answer": 自然语言回答,
                "elapsed_ms": 耗时(毫秒),
                "error": 错误信息(如有),
            }
        """
        start_time = time.time()
        result = {
            "query": query,
            "intent": None,
            "entities": None,
            "normalized_query": query,
            "sql": None,
            "raw_result": None,
            "answer": None,
            "elapsed_ms": 0,
            "error": None,
        }

        try:
            # ─── Step 1: 意图路由 ───
            if not self.skip_intent:
                intent_result = classify_intent(query)
                result["intent"] = intent_result

                if intent_result["intent"] == "out_of_scope":
                    result["answer"] = reject_chain.invoke({"query": query})
                    result["elapsed_ms"] = int((time.time() - start_time) * 1000)
                    return result
            else:
                result["intent"] = {"intent": "skipped", "confidence": 1.0, "reasoning": "跳过意图路由"}

            # ─── Step 2: 实体抽取与纠错 ───
            entities_result: ExtractionResult = extract_entities(query)
            result["entities"] = entities_result.to_dict()
            result["normalized_query"] = entities_result.normalized_query

            # ─── Step 3: Text2SQL ───
            sql = generate_sql(query, entities_result)
            result["sql"] = sql

            # ─── Step 4: SQL 执行 ───
            exec_result = validate_and_execute({
                "query": query,
                "sql": sql,
            })
            result["raw_result"] = exec_result.get("result", "")

            # ─── Step 5: 结果格式化 ───
            if exec_result.get("sql_error"):
                result["answer"] = f"查询执行出现问题: {exec_result.get('result', '未知错误')}"
            else:
                answer = format_answer(query, sql, exec_result.get("result", ""))
                result["answer"] = answer

        except Exception as e:
            result["error"] = str(e)
            result["answer"] = f"处理过程中发生错误: {str(e)}"
            traceback.print_exc()

        result["elapsed_ms"] = int((time.time() - start_time) * 1000)
        return result


# ─── 全局 pipeline 实例 ───
pipeline = NL2SQLPipeline()


def ask(query: str, skip_intent: bool = False) -> str:
    """
    简单接口: 输入问题，返回回答
    """
    p = NL2SQLPipeline(skip_intent=skip_intent)
    result = p.run(query)
    return result["answer"]


if __name__ == "__main__":
    import json

    test_queries = [
        "APG 海缆现在有什么故障？",
        "当前未修复的故障有哪些？",
        "AAE-1 今年有几次故障？",
        "今天天气怎么样？",
    ]

    for q in test_queries:
        print(f"{'='*60}")
        print(f"问题: {q}")
        result = pipeline.run(q)
        print(f"意图: {result['intent']}")
        print(f"SQL: {result['sql']}")
        print(f"回答: {result['answer']}")
        print(f"耗时: {result['elapsed_ms']}ms")
        if result["error"]:
            print(f"错误: {result['error']}")
        print()
