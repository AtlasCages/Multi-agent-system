"""
AgentTeam CLI 入口
命令：直接运行 python main.py
"""

import os
import sys

# 确保包可导入
_parent = os.path.dirname(os.path.abspath(__file__))
_parent_parent = os.path.dirname(_parent)
sys.path.insert(0, _parent_parent)

from agent_team.config import OUTPUT_DIR_PREFIX
from agent_team.core.llm import test_connection
from agent_team.core.state import create_initial_state
from agent_team.core.graph import build_graph
from agent_team.tools.plugins import set_output_dir
from agent_team.pipeline.checkpoint import create_output_dir, find_latest_checkpoint, save_checkpoint
from langchain_core.messages import HumanMessage


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = create_output_dir(base_dir)
    set_output_dir(output_dir)

    print(f"📁 输出目录: {output_dir}")

    if not test_connection():
        print("❌ 请联系管理员检查 API 配置")
        sys.exit(1)

    print("\n===== 🤖 AgentTeam 多 Agent 协作系统 =====")
    print("输入 'exit' 退出，输入 'continue' 断点续跑\n")

    app = build_graph()
    chat_history = []
    current_output_dir = output_dir

    while True:
        try:
            user_input = input("请输入项目需求: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 再见！")
            break

        # ── 断点续跑 ──
        if user_input.lower() == "continue":
            saved_state, saved_dir = find_latest_checkpoint(base_dir)
            if not saved_state:
                print("❌ 未找到可恢复的检查点")
                continue

            confirm = input("是否恢复上次任务？(y/n): ").strip().lower()
            if confirm != "y":
                print("开始新任务...")
                continue

            clean = input("是否清洗旧状态？(y/n，默认 y): ").strip().lower()
            if clean != "n":
                for field in ["project_plan", "requirements_doc", "architecture_doc",
                              "code", "paper", "code_review", "final_report"]:
                    saved_state.pop(field, None)
                saved_state["fix_attempts"] = 0
                print("🧹 旧状态已清洗")

            chat_history = saved_state.get("chat_history", [])
            current_output_dir = saved_dir
            set_output_dir(saved_dir)
            print(f"✅ 恢复对话，输出目录: {saved_dir}")
            continue

        # ── 运行任务 ──
        print(f"\n📋 收到需求: {user_input}")
        print("🤖 团队开始工作...\n")

        initial_state = create_initial_state(user_input, chat_history)
        config = {"configurable": {"thread_id": f"chat_{len(chat_history)}"}}

        try:
            for step in app.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, output in step.items():
                    print(f"\n{'='*50}")
                    print(f"节点 [{node_name}] 完成")
                    print(f"输出: {list(output.keys())}")
                    if "current_step" in output:
                        print(f"📍 {output['current_step']}")
                    print(f"{'='*50}")

                    current_state = app.get_state(config).values
                    save_checkpoint(current_state, current_output_dir)

            final_state = app.get_state(config).values
            report = final_state.get("final_report", "")

            print("\n" + "=" * 50)
            if report:
                print(report[:500])
            else:
                for key in ["project_plan", "requirements_doc", "architecture_doc", "code"]:
                    val = final_state.get(key, "")
                    if val:
                        print(f"\n📄 {key}: {val[:200]}...")
                        break

            # 保存完整报告
            report_path = os.path.join(current_output_dir, "project_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                parts = []
                for key in ["user_requirement", "project_plan", "requirements_doc",
                            "architecture_doc", "code", "paper", "final_report"]:
                    val = final_state.get(key, "")
                    if val:
                        parts.append(f"# {key}\n{val}")
                f.write("\n\n".join(parts))
            print(f"\n📁 完整报告: {report_path}")

            chat_history.append({
                "task": user_input,
                "output_dir": current_output_dir,
                "summary": (report or final_state.get("code", ""))[:200],
            })

            final_state["is_completed"] = True
            save_checkpoint(final_state, current_output_dir)

        except KeyboardInterrupt:
            print("\n⏸️ 任务暂停，进度已保存。下次输入 continue 可恢复。")
            try:
                current_state = app.get_state(config).values
                save_checkpoint(current_state, current_output_dir)
            except Exception:
                pass
        except Exception as e:
            print(f"\n❌ 任务异常: {e}")
            print("输入 continue 可尝试恢复")
            try:
                current_state = app.get_state(config).values
                save_checkpoint(current_state, current_output_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
