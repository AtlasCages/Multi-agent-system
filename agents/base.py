"""
Agent 节点工厂
支持两种模式：
1. 工具模式 — LLM 可通过 bind_tools 调用工具
2. 纯文本模式 — LLM 直接输出文本
"""

import os
import re
import json
import time
import inspect
from typing import Callable, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as langchain_tool

from agent_team.core.state import AgentState
from agent_team.core.llm import get_llm, get_llm_with_tools
from agent_team.tools.registry import _registry, get_tool_descriptions, execute_tool


def create_agent_node(
    name: str,
    system_prompt: str,
    output_key: str,
    allowed_tools: list = None,
    max_tool_rounds: int = 3,
    log_callback: Callable = None,
    extract_code: bool = False,
) -> Callable:
    """
    创建 Agent LangGraph 节点。

    参数:
        name: Agent 显示名称
        system_prompt: 系统提示词
        output_key: 输出写入 state 的键
        allowed_tools: 允许使用的工具名列表(None = 纯文本)
        max_tool_rounds: 最大工具调用轮数
        log_callback: 日志回调函数
        extract_code: 是否从回复中提取 ```python 代码块
    """
    full_prompt = system_prompt + "\n\n" + get_tool_descriptions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "当前任务：完成你的 {agent_name} 工作。\n\n上下文:\n{context}\n\n工具返回: {tool_result}"),
    ])
    prompt = prompt.partial(agent_name=name, system_prompt=full_prompt)

    # 工具模式：使用 bind_tools
    if allowed_tools:
        registered_tools = []
        for t_name in allowed_tools:
            t_info = _registry.get(t_name)
            if t_info:
                registered_tools.append(_build_langchain_tool(t_info))
        if registered_tools:
            chain = prompt | get_llm_with_tools(registered_tools)
        else:
            chain = prompt | get_llm()
    else:
        chain = prompt | get_llm()

    def node(state: AgentState):
        _log(f"\n===== {name} 开始 =====")

        new_state = dict(state)
        context = _build_context(state)
        tool_result = "无"

        for i in range(max_tool_rounds):
            _log(f"[{name}] 第 {i+1} 轮...")

            try:
                result = chain.invoke({
                    "context": context,
                    "tool_result": tool_result,
                })
            except Exception as e:
                _log(f"❌ [{name}] LLM 调用失败: {e}")
                new_state[output_key] = f"执行失败: {e}"
                new_state["error"] = str(e)
                new_state["current_step"] = name
                new_state["fix_attempts"] = state.get("fix_attempts", 0) + 1
                new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
                return new_state

            # ── 处理 tool calls ──
            if hasattr(result, "tool_calls") and result.tool_calls:
                for tc in result.tool_calls:
                    _log(f"🔧 [{name}] 调用: {tc['name']}")
                    tool_result = execute_tool(tc["name"], tc.get("args", {}))
                    _log(f"   结果: {tool_result[:100]}...")
                continue

            # ── 纯文本输出 ──
            text = result.content if isinstance(result, AIMessage) else str(result)

            # 代码提取模式
            if extract_code:
                code = _extract_code(text)
                if code:
                    new_state["code"] = code
                    _log(f"📄 提取代码 {len(code)} 字符")
                    # 自动保存到文件
                    output_dir = _get_output_dir()
                    if output_dir:
                        try:
                            code_path = os.path.join(output_dir, "todo_manager.py")
                            with open(code_path, "w", encoding="utf-8") as f:
                                f.write(code)
                            _log(f"💾 代码已保存: {code_path}")
                        except Exception as e:
                            _log(f"⚠️ 保存代码失败: {e}")
                    new_state[output_key] = code
                    break
                # 没提取到代码就存原始文本
                new_state[output_key] = text
            else:
                new_state[output_key] = text
            break

        new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
        new_state["current_step"] = name
        new_state["fix_attempts"] = state.get("fix_attempts", 0) + 1
        _log(f"✅ {name} 完成")
        return new_state

    def _log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    return node


# ─── 辅助函数 ──────────────────────────────────


def _build_context(state: AgentState) -> str:
    """从 state 中提取上下文"""
    return "\n\n".join([
        f"用户需求: {state.get('user_requirement', '')}",
        f"项目计划: {state.get('project_plan', '')}",
        f"需求文档: {state.get('requirements_doc', '')}",
        f"架构设计: {state.get('architecture_doc', '')}",
        f"实现代码: {state.get('code', '')}",
        f"代码审查意见: {state.get('code_review', '')}",
    ])


def _extract_code(text: str) -> str:
    """从文本中提取 ```python 代码块"""
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).replace('\\n', '\n').replace('\\t', '\t')
    return ""


def _build_langchain_tool(t_info: dict):
    """将插件注册表里的工具转为 LangChain Tool 对象"""
    func = t_info["func"]
    tool_name = t_info["name"]
    tool_desc = t_info["description"]

    @langchain_tool(description=tool_desc)
    def wrapper(**kwargs) -> str:
        """Execute a registered tool plugin"""
        return execute_tool(tool_name, kwargs)

    return wrapper


def _get_output_dir() -> str:
    """获取当前输出目录"""
    from agent_team.tools.plugins import get_output_dir
    return get_output_dir()
