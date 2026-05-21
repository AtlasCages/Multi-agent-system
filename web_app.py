import os
import sys
import datetime
import gradio as gr
import aiagent
#创建目录
web_output_dir = os.path.join(
    .BASE_DIR,
    f"project_output_web_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(web_output_dir, exist_ok=True)
aiagent.PROJECT_OUTPUT_DIR = web_output_dir
from aiagent import (
    build_agent_graph,
    AgentState,
    HumanMessage,
    save_checkpoint,
)
def run_agent_task_stream(user_requirement):
    """
    流式执行 Agent 任务，利用回调实时输出日志。
    """
    log_store = []      
    def on_log(msg):
        log_store.append(msg)  
    app = build_agent_graph(log_callback=on_log)
    initial_state = {
        'user_requirement': user_requirement,
        'messages': [HumanMessage(content=user_requirement)],
        'current_step': '开始',
        'fix_attempts': 0,
        'project_plan': '',
        'requirements_doc': '',
        'architecture_doc': '',
        'code': '',
        'paper': '',
        'code_review': '',
        'final_report': '',
        'error': '',
        'chat_history': [],
        'is_completed': False,
        'last_checkpoint': 'start'
    }
    thread_id = f"web_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {'configurable': {'thread_id': thread_id}}
    status_log = ""
    collected_files = []
    yield "⏳ 正在初始化团队，请稍候...", [], "准备中..."
    try:
        for step in app.stream(initial_state, config=config, stream_mode="updates"):
            while log_store:
                status_log += log_store.pop(0) + "\n"
            for fname in os.listdir(Day20.PROJECT_OUTPUT_DIR):
                fpath = os.path.join(Day20.PROJECT_OUTPUT_DIR, fname)
                if fname.endswith(('.md', '.py', '.mermaid')) and fpath not in collected_files:
                    collected_files.append(fpath)
            yield status_log, collected_files, "任务进行中..."
        while log_store:
            status_log += log_store.pop(0) + "\n"
        final_state = app.get_state(config).values
        summary = final_state.get('final_report', '')[:500] or "任务完成，请查看输出文件。"
        yield status_log + "\n🎉 全部任务完成！", collected_files, summary
    except Exception as e:
        yield status_log + f"\n❌ 任务异常: {str(e)}", collected_files, f"执行失败：{str(e)}"
#界面
with gr.Blocks(title="多Agent协作系统") as demo:
    gr.Markdown("""
    # 🤖 多Agent协作系统
    输入你的项目需求，AI团队将自动完成：**项目计划 → 需求分析 → 架构设计 → 代码实现 → 代码审查 → 论文撰写 → 验收报告**
    """)
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="请输入你的项目需求",
                placeholder="例如：开发一个命令行的待办事项管理工具",
                lines=3
            )
            submit_btn = gr.Button("🚀 启动", variant="primary")
        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    "开发一个命令行的待办事项管理工具",
                    "开发一个Markdown笔记管理系统",
                    "开发一个CSV数据分析报告生成器",
                ],
                inputs=user_input
            )
    with gr.Accordion("实时执行进度", open=True):
        status_output = gr.Textbox(
            label="执行日志（流式更新）",
            lines=10,
            max_lines=30,
            interactive=False,
            autoscroll=True
        )
    with gr.Row():
        with gr.Column():
            file_output = gr.File(
                label="生成的文件（实时更新，可多选下载）",
                file_count="multiple",
                interactive=False
            )
    with gr.Accordion("最终验收报告摘要", open=False):
        summary_output = gr.Textbox(
            label="报告摘要",
            lines=6,
            interactive=False
        )
    submit_btn.click(
        fn=run_agent_task_stream,
        inputs=user_input,
        outputs=[status_output, file_output, summary_output]
    )
#启动
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        inbrowser=True
    )