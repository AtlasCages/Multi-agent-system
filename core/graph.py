"""
LangGraph 认知架构图
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_team.core.state import AgentState
from agent_team.agents.base import create_agent_node
from agent_team.agents.prompts import (
    MANAGER_PROMPT, ANALYST_PROMPT, ARCHITECT_PROMPT,
    CODER_PROMPT, WRITER_PROMPT,
)
from agent_team.agents.cleaner import create_cleaner_node
from agent_team.agents.reviewer import create_reviewer_node, should_fix_code
from agent_team.pipeline.acceptance import create_acceptance_node


def build_graph(log_callback=None):
    """构建多 Agent 工作流图"""
    workflow = StateGraph(AgentState)

    # ── 节点 ──
    workflow.add_node("project_manager", create_agent_node(
        name="项目管理QA",
        system_prompt=MANAGER_PROMPT,
        output_key="project_plan",
        allowed_tools=["write_file", "generate_mermaid"],
        log_callback=log_callback,
    ))
    workflow.add_node("requirements_analyst", create_agent_node(
        name="需求分析师",
        system_prompt=ANALYST_PROMPT,
        output_key="requirements_doc",
        allowed_tools=["write_file"],
        log_callback=log_callback,
    ))
    workflow.add_node("architect", create_agent_node(
        name="架构设计师",
        system_prompt=ARCHITECT_PROMPT,
        output_key="architecture_doc",
        allowed_tools=["write_file", "generate_mermaid"],
        log_callback=log_callback,
    ))
    workflow.add_node("cleaner", create_cleaner_node(log_callback=log_callback))
    workflow.add_node("code_expert", create_agent_node(
        name="代码实现专家",
        system_prompt=CODER_PROMPT,
        output_key="code",
        allowed_tools=["write_file"],
        log_callback=log_callback,
        extract_code=True,
    ))
    workflow.add_node("code_reviewer", create_reviewer_node(log_callback=log_callback))
    workflow.add_node("paper_writer", create_agent_node(
        name="论文写作助手",
        system_prompt=WRITER_PROMPT,
        output_key="paper",
        allowed_tools=["write_file"],
        log_callback=log_callback,
    ))
    workflow.add_node("final_acceptance", create_acceptance_node(log_callback=log_callback))

    # ── 边 ──
    workflow.set_entry_point("project_manager")
    workflow.add_edge("project_manager", "requirements_analyst")
    workflow.add_edge("requirements_analyst", "architect")
    workflow.add_edge("architect", "cleaner")
    workflow.add_edge("cleaner", "code_expert")
    workflow.add_edge("code_expert", "code_reviewer")

    workflow.add_conditional_edges(
        "code_reviewer",
        should_fix_code,
        {"code_expert": "code_expert", "paper_writer": "paper_writer"},
    )
    workflow.add_edge("paper_writer", "final_acceptance")
    workflow.add_edge("final_acceptance", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
