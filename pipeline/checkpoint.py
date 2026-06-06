"""
检查点管理 — 断点续跑支持
"""

import os
import json
from datetime import datetime

from agent_team.config import SAVED_STATE_FILENAME, OUTPUT_DIR_PREFIX
from agent_team.core.state import AgentState


def create_output_dir(base_dir: str) -> str:
    """创建新的输出目录"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{OUTPUT_DIR_PREFIX}{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def find_latest_checkpoint(base_dir: str):
    """从输出目录中查找最新的检查点"""
    dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith(OUTPUT_DIR_PREFIX)],
        reverse=True,
    )
    for d in dirs:
        state_file = os.path.join(base_dir, d, SAVED_STATE_FILENAME)
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f), os.path.join(base_dir, d)
    return None, None


def save_checkpoint(state: AgentState, output_dir: str):
    """保存当前状态到检查点 JSON"""
    try:
        saveable = {
            "user_requirement": state.get("user_requirement", ""),
            "chat_history": state.get("chat_history", []),
            "is_completed": state.get("is_completed", False),
            "last_checkpoint": state.get("last_checkpoint", "start"),
            "fix_attempts": state.get("fix_attempts", 0),
            "project_plan": state.get("project_plan", ""),
            "requirements_doc": state.get("requirements_doc", ""),
            "architecture_doc": state.get("architecture_doc", ""),
            "code": state.get("code", ""),
            "paper": state.get("paper", ""),
            "code_review": state.get("code_review", ""),
            "review_decision": state.get("review_decision", {}),
            "final_report": state.get("final_report", ""),
        }
        path = os.path.join(output_dir, SAVED_STATE_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ 保存检查点失败: {e}")
        return False
