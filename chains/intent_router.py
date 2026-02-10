"""
意图路由 Chain
使用 Qwen3-32B thinking 模式进行意图分类
"""
import json
import sys
import os

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from prompts.intent import INTENT_PROMPT

# ─── 支持 thinking 模式的 LLM ───
llm_think = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    model=config.LLM_MODEL_NAME,
    api_key=config.LLM_API_KEY,
    temperature=0.1,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

VALID_INTENTS = {
    "fault_status", "business_impact", "fault_time",
    "fault_segment", "repair_status", "report_analysis",
    "out_of_scope",
}


def _parse_intent_output(text: str) -> dict:
    """解析 LLM 的意图分类输出"""
    # 尝试提取 JSON
    text = text.strip()
    # 移除可能的 markdown 代码块标记
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        intent = result.get("intent", "out_of_scope")
        if intent not in VALID_INTENTS:
            intent = "out_of_scope"
        return {
            "intent": intent,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, TypeError):
        # 尝试从文本中提取意图关键词
        for intent_name in VALID_INTENTS:
            if intent_name in text:
                return {"intent": intent_name, "confidence": 0.6, "reasoning": text[:100]}
        return {"intent": "out_of_scope", "confidence": 0.3, "reasoning": "无法解析意图"}


# ─── 意图识别 Chain ───
intent_chain = (
    INTENT_PROMPT
    | llm_think
    | StrOutputParser()
    | RunnableLambda(_parse_intent_output)
)


def classify_intent(query: str) -> dict:
    """便捷调用函数"""
    return intent_chain.invoke({"query": query})


if __name__ == "__main__":
    # 测试
    test_queries = [
        "APG 海缆现在有什么故障？",
        "目前影响业务的故障有多少？",
        "今天天气怎么样？",
        "AAE-1 的 S1.8 段修了吗？",
        "今年故障次数最多的海缆是哪条？",
    ]
    for q in test_queries:
        result = classify_intent(q)
        print(f"Q: {q}")
        print(f"A: {result}\n")
