"""
AgentTeam Web UI — Gradio 界面
"""

import os
import sys
import datetime

_parent = os.path.dirname(os.path.abspath(__file__))
_parent_parent = os.path.dirname(_parent)
sys.path.insert(0, _parent_parent)

import gradio as gr

from agent_team.config import OUTPUT_DIR_PREFIX
from agent_team.core.state import create_initial_state
from agent_team.core.graph import build_graph
from agent_team.tools.plugins import set_output_dir
from agent_team.pipeline.checkpoint import create_output_dir, save_checkpoint


def run_agent_stream(user_requirement: str):
    """
    流式执行 Agent 任务，实时输出日志。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = create_output_dir(base_dir)
    set_output_dir(output_dir)

    log_store = []
    collected_files = []

    def on_log(msg):
        log_store.append(msg)

    app = build_graph(log_callback=on_log)
    initial_state = create_initial_state(user_requirement)

    thread_id = f"web_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    status_log = ""
    yield "⏳ 团队初始化中...", [], "准备中..."

    try:
        for step in app.stream(initial_state, config=config, stream_mode="updates"):
            # 刷新日志
            while log_store:
                status_log += log_store.pop(0) + "\n"

            # 更新文件列表
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname)
                if fname.endswith((".md", ".py", ".mermaid")) and fpath not in collected_files:
                    collected_files.append(fpath)

            yield status_log, collected_files, "任务进行中..."

        # 最后刷新
        while log_store:
            status_log += log_store.pop(0) + "\n"
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if fname.endswith((".md", ".py", ".mermaid")) and fpath not in collected_files:
                collected_files.append(fpath)

        final_state = app.get_state(config).values
        save_checkpoint(final_state, output_dir)

        summary = (final_state.get("final_report", "") or
                   final_state.get("project_plan", "") or
                   "任务完成！")[:500] or "任务完成，请查看输出文件。"

        yield status_log + "\n🎉 全部任务完成！", collected_files, summary

    except Exception as e:
        yield status_log + f"\n❌ 任务异常: {str(e)}", collected_files, f"执行失败: {e}"


# ─── Gradio UI ──────────────────────────────

with gr.Blocks(title="AgentTeam — 多Agent协作系统") as demo:
    gr.Markdown("""
    # 🤖 AgentTeam — 多 Agent 协作系统

    输入你的项目需求，AI 团队将自动完成：
    **项目计划 → 需求分析 → 架构设计 → 代码实现 → 代码审查 → 论文撰写 → 验收报告**
    """)

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="项目需求",
                placeholder="例如：开发一个命令行的待办事项管理工具",
                lines=3,
            )
            submit_btn = gr.Button("🚀 启动", variant="primary")

        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    "开发一个命令行的待办事项管理工具",
                    "开发一个 Markdown 笔记管理系统",
                    "开发一个 CSV 数据分析报告生成器",
                ],
                inputs=user_input,
            )

    with gr.Accordion("📋 实时执行进度", open=True):
        status_output = gr.Textbox(
            label="执行日志",
            lines=12,
            max_lines=40,
            interactive=False,
        )

    with gr.Row():
        file_output = gr.File(
            label="生成的文件（可下载）",
            file_count="multiple",
            interactive=False,
        )

    with gr.Accordion("📝 最终验收摘要", open=False):
        summary_output = gr.Textbox(
            label="报告摘要",
            lines=8,
            interactive=False,
        )

    submit_btn.click(
        fn=run_agent_stream,
        inputs=user_input,
        outputs=[status_output, file_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, inbrowser=True)
