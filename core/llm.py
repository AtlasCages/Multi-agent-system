"""
LLM 统一入口 — 懒加载，导入时不触发 API 调用
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agent_team.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT, LLM_MAX_RETRIES,
)

_llm = None


def get_llm():
    """懒加载 LLM 实例"""
    global _llm
    if _llm is None:
        if not LLM_API_KEY:
            raise ValueError(
                "API Key 未设置！请在 .env 中配置 TEAM_API_KEY 或 SILICONFLOW_API_KEY"
            )
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )
    return _llm


def get_llm_with_tools(tools: list):
    """获取绑定了工具的 LLM 实例"""
    llm = get_llm()
    return llm.bind_tools(tools)


def test_connection():
    """测试 API 连接（供入口调用）"""
    try:
        llm = get_llm()
        resp = llm.invoke([HumanMessage(content="回复 OK 即可")])
        print(f"✅ API 连接成功！模型: {LLM_MODEL}")
        return True
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        return False
