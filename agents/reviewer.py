"""
代码审查 Agent
使用 with_structured_output 实现结构化审查结果
"""

import os
import re
from typing import Callable
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from agent_team.core.state import AgentState
from agent_team.core.llm import get_llm
from agent_team.agents.base import _get_output_dir
from agent_team.agents.prompts import REVIEWER_PROMPT


class ReviewOutput(BaseModel):
    """结构化的审查结果"""
    passed: bool = Field(description="代码是否通过审查")
    issues: list[str] = Field(description="发现的问题列表，通过时为[]")
    suggestions: list[str] = Field(description="修改建议列表，通过时为[]")
    summary: str = Field(description="审查总结（2-3句话）")


def should_fix_code(state: AgentState) -> str:
    """
    条件判断函数 — 基于结构化 review_decision 决定下一步。
    返回: "code_expert" 或 "paper_writer"
    """
    decision = state.get("review_decision", {})
    passed = decision.get("passed", False)
    fix_attempts = state.get("fix_attempts", 0)

    if fix_attempts >= 3:
        print(f"⚠️ 已达最大修复次数({fix_attempts})，进入论文阶段")
        return "paper_writer"

    if passed:
        print("✅ 审查通过，进入论文撰写")
        return "paper_writer"

    issues = decision.get("issues", [])
    print(f"🔁 审查不通过，发现 {len(issues)} 个问题，进入修复 (第 {fix_attempts + 1} 次)")
    return "code_expert"


def create_reviewer_node(log_callback: Callable = None):
    """
    代码审查 Agent 节点
    使用 with_structured_output 确保输出格式稳定
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWER_PROMPT),
        ("human", "请审查以下代码:\n\n```python\n{code}\n```"),
    ])

    llm = get_llm()
    structured_llm = llm.with_structured_output(ReviewOutput)
    chain = prompt | structured_llm

    def node(state: AgentState):
        _log("\n===== 代码审查专家 开始 =====")

        new_state = dict(state)

        # 先尝试读文件
        code = state.get("code", "")
        if not code or len(code.strip()) == 0:
            _log("⚠️ 代码为空，审查自动不通过")
            new_state["code_review"] = "审查结果：不通过\n问题列表：\n1. 代码文件为空\n修改建议：请重新生成完整代码。"
            new_state["review_decision"] = {
                "passed": False,
                "issues": ["代码为空"],
                "suggestions": ["重新生成代码"],
                "summary": "代码文件为空，无法审查。",
            }
            new_state["current_step"] = "代码审查专家"
            new_state["messages"] = [HumanMessage(content="代码审查专家工作完成")]
            return new_state

        _log(f"📄 审查代码 ({len(code)} 字符)...")

        try:
            result = chain.invoke({"code": code[:6000]})

            # 生成文本版审查报告
            status = "通过" if result.passed else "不通过"
            report = f"审查结果：{status}\n\n"
            if result.issues:
                report += "问题列表:\n" + "\n".join(f"{i+1}. {iss}" for i, iss in enumerate(result.issues)) + "\n\n"
            if result.suggestions:
                report += "修改建议:\n" + "\n".join(f"{i+1}. {sug}" for i, sug in enumerate(result.suggestions)) + "\n\n"
            report += f"总结: {result.summary}"

            new_state["code_review"] = report
            new_state["review_decision"] = result.model_dump()
            new_state["current_step"] = "代码审查专家"

            _log(f"📋 审查: {status} — {result.summary[:80]}")

            # 保存报告
            output_dir = _get_output_dir()
            if output_dir:
                try:
                    path = os.path.join(output_dir, "code_review.md")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(report)
                    _log(f"💾 审查报告已保存: {path}")
                except Exception as e:
                    _log(f"⚠️ 保存审查报告失败: {e}")

        except Exception as e:
            _log(f"❌ 审查失败: {e}")
            new_state["code_review"] = f"审查执行失败: {e}"
            new_state["review_decision"] = {"passed": False, "issues": [str(e)], "suggestions": [], "summary": "审查异常"}

        new_state["messages"] = [HumanMessage(content="代码审查专家工作完成")]
        new_state["fix_attempts"] = state.get("fix_attempts", 0) + 1
        _log("✅ 代码审查专家 完成")
        return new_state

    def _log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    return node
