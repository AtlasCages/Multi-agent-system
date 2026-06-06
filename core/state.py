"""
AgentState 定义
"""

from typing import TypedDict, Annotated, Sequence, List
from operator import add
from langchain_core.messages import BaseMessage


class ReviewDecision(TypedDict, total=False):
    """代码审查的标准化结果"""
    passed: bool
    issues: List[str]
    suggestions: List[str]
    summary: str


class AgentState(TypedDict, total=False):
    """多 Agent 系统的完整状态"""
    messages: Annotated[Sequence[BaseMessage], add]
    user_requirement: str
    project_plan: str
    requirements_doc: str
    architecture_doc: str
    code: str
    paper: str
    final_report: str
    current_step: str

    # 审查闭环
    code_review: str           # 原始审查文本
    review_decision: ReviewDecision  # 结构化审查结果
    fix_attempts: int
    error: str

    # 历史与状态
    chat_history: list
    is_completed: bool
    last_checkpoint: str


def create_initial_state(user_requirement: str, chat_history: list = None) -> AgentState:
    """创建初始状态"""
    return {
        "user_requirement": user_requirement,
        "messages": [],
        "current_step": "开始",
        "fix_attempts": 0,
        "project_plan": "",
        "requirements_doc": "",
        "architecture_doc": "",
        "code": "",
        "paper": "",
        "code_review": "",
        "review_decision": {},
        "final_report": "",
        "error": "",
        "chat_history": chat_history or [],
        "is_completed": False,
        "last_checkpoint": "start",
    }
