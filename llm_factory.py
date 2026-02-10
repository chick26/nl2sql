"""
LLM 工厂: 统一创建 Chat LLM 实例
支持 openai_compat / openai / azure
"""
from __future__ import annotations

import config


def _build_extra_body(thinking: bool) -> dict | None:
    if not config.LLM_SUPPORTS_THINKING:
        return None
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def create_chat_llm(*, thinking: bool, temperature: float):
    provider = (config.LLM_PROVIDER or "openai_compat").lower()

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_DEPLOYMENT):
            raise ValueError("Azure OpenAI 配置不完整，请设置 AZURE_OPENAI_ENDPOINT/API_KEY/DEPLOYMENT")

        return AzureChatOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION or None,
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            temperature=temperature,
        )

    # 默认走 OpenAI 兼容接口
    from langchain_openai import ChatOpenAI

    extra_body = _build_extra_body(thinking)
    kwargs = {
        "model": config.LLM_MODEL_NAME,
        "api_key": config.LLM_API_KEY,
        "temperature": temperature,
    }

    # openai_compat 使用自定义 base_url；openai 用默认地址
    if provider != "openai":
        kwargs["base_url"] = config.LLM_BASE_URL

    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)
