"""
保洁 Agent — LLM 驱动的内容清洗
相比原版 regex 删除，改用 LLM 进行语义级清理
"""

import re
from typing import Callable, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from agent_team.core.state import AgentState
from agent_team.core.llm import get_llm
from agent_team.agents.prompts import CLEANER_PROMPT


def create_cleaner_node(log_callback: Callable = None):
    """
    保洁 Agent 节点。
    先用 LLM 清洗，再用 regex 兜底。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLEANER_PROMPT),
        ("human", "请清理以下文本中的工具调用残留和格式污染:\n\n{text}"),
    ])
    chain = prompt | get_llm()

    def node(state: AgentState):
        _log("\n🧹 保洁 Agent 开始语义清洗...")
        new_state = dict(state)

        fields_to_clean = ["project_plan", "requirements_doc", "architecture_doc", "code"]

        for field in fields_to_clean:
            original = state.get(field, "")
            if not original or len(original) < 50:
                continue

            try:
                result = chain.invoke({"text": original[:4000]})
                cleaned = result.content.strip()

                # regex 兜底清理
                cleaned = re.sub(r'```mermaid\s*.*?\s*```', '', cleaned, flags=re.DOTALL)
                cleaned = re.sub(r'\[\s*\{[^}]*"name"\s*:[^}]*\}\s*\]', '', cleaned, flags=re.DOTALL)
                cleaned = re.sub(r'\n{4,}', '\n\n', cleaned).strip()

                if cleaned and len(cleaned) > 50:
                    new_state[field] = cleaned
                    _log(f"  ✅ {field}: {len(original)} → {len(cleaned)} 字符")
                else:
                    _log(f"  ⚠️ {field}: 清洗后过短，保留原文")
            except Exception as e:
                _log(f"  ⚠️ {field} 清洗失败: {e}")

        new_state["messages"] = [HumanMessage(content="保洁Agent工作完成")]
        new_state["current_step"] = "上下文清洗"
        _log("🧹 保洁完成")
        return new_state

    def _log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    return node
