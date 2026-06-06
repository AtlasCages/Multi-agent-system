"""
最终验收节点
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from agent_team.core.state import AgentState
from agent_team.core.llm import get_llm


def create_acceptance_node(log_callback=None):
    """最终验收报告生成节点"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位高级项目验收专家，请根据以下项目产出，生成一份完整的最终验收报告。"),
        ("human", """
用户需求: {user_requirement}
项目计划: {project_plan}
需求文档: {requirements_doc}
架构设计: {architecture_doc}
代码实现: {code}
技术论文: {paper}
代码审查: {code_review}

请基于以上信息生成最终验收报告。
要求:
- 格式清晰，使用标题分段
- 忽略残留的 JSON 格式数据
- 对每个部分给出评价和建议
- 直接输出纯文本，不要输出代码块或 JSON
"""),
    ])
    chain = prompt | get_llm()

    def node(state: AgentState):
        _log("\n===== 最终验收 开始 =====")
        new_state = dict(state)

        try:
            result = chain.invoke({
                "user_requirement": state.get("user_requirement", ""),
                "project_plan": state.get("project_plan", ""),
                "requirements_doc": state.get("requirements_doc", ""),
                "architecture_doc": state.get("architecture_doc", ""),
                "code": state.get("code", ""),
                "paper": state.get("paper", ""),
                "code_review": state.get("code_review", ""),
            })
            new_state["final_report"] = result.content
            new_state["current_step"] = "最终验收"
            _log("✅ 最终验收报告生成完毕")
        except Exception as e:
            _log(f"❌ 验收失败: {e}")
            new_state["final_report"] = f"验收报告生成失败: {e}"
            new_state["error"] = str(e)

        new_state["messages"] = [HumanMessage(content="最终验收工作完成")]
        return new_state

    def _log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    return node
