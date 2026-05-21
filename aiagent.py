import os
import sys
import subprocess
import io
import json, re
import inspect
import datetime
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import torch
from typing import TypedDict, Annotated, Sequence, Optional, Any
from operator import add
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.llms import LLM
from pydantic import Field
from langchain_openai import ChatOpenAI
import ast
import re
import json
# JSON解析
def safe_parse_tool_call(text: str):
    """增强版 JSON 解析，处理 Qwen3-8B 的不规范输出"""
    if not text:
        return None
    def try_parse(json_str):
        """尝试解析 JSON,成功返回对象,失败返回 None"""
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "name" in obj:
                return obj
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                return obj[0]
        except:
            pass
        return None
    result = try_parse(text)
    if result:
        return result
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        result = try_parse(cleaned)
        if result:
            return result
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if not match:
        match = re.search(
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}\s*\}',
            text, re.DOTALL
        )
    if not match:
        match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r'(?<!\\)\n', r'\\n', candidate)
        candidate = re.sub(r'(?<!\\)\r', r'\\r', candidate)
        candidate = re.sub(r'(?<!\\)\t', r'\\t', candidate)
        candidate = re.sub(
            r'("content"\s*:\s*")(.*?)("(?=\s*[,}]))',
            lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3),
            candidate, flags=re.DOTALL
        )
        result = try_parse(candidate)
        if result:
            return result
    return None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_OUTPUT_DIR = None 
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
MODEL_NAME = "Qwen/Qwen3-8B"
BASE_URL = "https://api.siliconflow.cn/v1"
TIMEOUT = 500 
MAX_RETRIES = 3
try:
    if not SILICONFLOW_API_KEY:
        raise ValueError("未找到SILICONFLOW_API_KEY环境变量,请先设置环境变量再运行.")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=SILICONFLOW_API_KEY,
        base_url=BASE_URL,
        temperature=0.1,
        max_tokens=8192,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
        streaming=True
    )
    print("正在测试硅基流动API连接...")
    test_response = llm.invoke([HumanMessage(content="你好")])
    print(f"API连接成功!当前使用模型:{MODEL_NAME}")
except Exception as e:
    print(f"\n初始化失败!错误详情:{str(e)}")
    print("\n解决方法:")
    print("1.确认你已经设置了正确的SILICONFLOW_API_KEY环境变量")
    print("2.确认密钥没有被删除或禁用")
    print("3.确认你已经完成了硅基流动的实名认证")
    print("4.如果还是不行,换成免费模型Qwen/Qwen3-8B测试")
    sys.exit(1)
#插件系统
_PLUGIN_REGISTRY = {}
def tool_plugin(name: str, description: str):
    """
    工具插件装饰器
    用法：@tool_plugin("工具名", "工具描述")
    """
    def decorator(func):
        _PLUGIN_REGISTRY[name] = {
            "name": name,
            "description": description,
            "func": func
        }
        return func
    return decorator
def get_tool_description() -> str:
    """动态生成工具描述（基于已注册的插件）"""
    desc = '你可以使用以下工具来完成任务:\n'
    for tool in _PLUGIN_REGISTRY.values():
        desc += f'- {tool["name"]}: {tool["description"]}\n'
    desc += """
工具调用规则：
1. 只输出纯JSON数组,不要任何```json、```、解释、说明、markdown格式
2. 正确格式：[{"name":"工具名","parameters":{"参数名":"参数值"}}]
3. 不要在JSON前后加任何文字
"""
    return desc
tool_registry = _PLUGIN_REGISTRY
#工具函数
@tool_plugin("run_code", "运行 Python 代码,参数:code (str) - 要运行的 Python 代码")
def run_code(code: str) -> str:
    try:
        exec(code, globals())
        return "代码执行成功"
    except Exception as e:
        return f"代码执行异常:{str(e)}"
@tool_plugin("write_file", "将内容写入本地文件,参数:file_path (str) - 文件路径, content (str) - 文件内容")
def write_file(file_path: str, content: str, **kwargs) -> str:
    try:
        full_path = os.path.join(PROJECT_OUTPUT_DIR, os.path.basename(file_path.lstrip("/\\")))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已保存到{full_path}"
    except Exception as e:
        return f"文件写入失败:{str(e)}"
@tool_plugin("generate_mermaid", "生成 Mermaid 图表,参数:chart_code (str) - 图表代码, file_name (str, 可选) - 保存文件名")
def generate_mermaid(chart_code: str, file_name: str = "architecture.md") -> str:
    try:
        full_path = os.path.join(PROJECT_OUTPUT_DIR, os.path.basename(file_name.lstrip("/\\")))
        chart_code = chart_code.strip().replace("\\n", "\n").replace("\\t", "\t")
        content = "```mermaid\n" + chart_code + "\n```"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"架构图已生成并保存在{full_path}"
    except Exception as e:
        return f"架构图生成失败:{str(e)}"
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    user_requirement: str
    project_plan: str
    requirements_doc: str
    architecture_doc: str
    code: str
    paper: str
    final_report: str
    current_step: str
    code_review: str
    fix_attempts: int
    error: str
    chat_history: list
    is_completed: bool                    
    last_checkpoint: str
#工具调用兜底
def execute_tool(tool_dict: dict) -> str:
    if not isinstance(tool_dict, dict):
        return "错误:工具调用必须是一个字典,包含 name 和 parameters"
    tool_name = tool_dict.get("name")
    parameters = tool_dict.get("parameters", {})
    if not tool_name:
        return "错误:工具名称缺失"
    for key in ["file_path", "file_name"]:
        if key in parameters:
            parameters[key] = os.path.basename(parameters[key])
    tool = tool_registry.get(tool_name)
    if not tool:
        return f"错误:未知工具'{tool_name}',可用工具:{list(tool_registry.keys())}"
    tool_func = tool["func"]
    valid_params = inspect.signature(tool_func).parameters.keys()
    filtered_params = {k: v for k, v in parameters.items() if k in valid_params}
    try:
        result = str(tool_func(**filtered_params))
        if tool_name == "write_file":
            content = tool_dict.get("parameters", {}).get("content", "")
            return content 
        return result
    except Exception as e:
        return f"工具执行失败:{str(e)}"
#agent节点
def create_tool_agent_node(name, system_prompt, output_key, allowed_tools=None, log_callback=None):
    """创建支持工具调用的Agent节点（支持回调输出）"""
    base_prompt = system_prompt + "\n\n" + get_tool_description()
    from langgraph.config import get_stream_writer
    base_prompt += """
核心任务指令：
1. 你的唯一任务是完成当前Agent的工作,不要执行其他Agent的任务。
2. 如果上下文信息中包含了与你无关的内容,如其他Agent的输出、历史任务残留等,请完全忽略它们。
3. 严格按照你的角色定义和输出格式进行工作。
"""
    if allowed_tools:
        base_prompt += f"\n你只能使用以下工具:{', '.join(allowed_tools)}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{base_prompt}"),
        ("human", "当前任务:完成你的{agent_name}工作\n\n上下文信息:\n{context}\n\n工具返回结果:{tool_result}")
    ])
    prompt = prompt.partial(agent_name=name, base_prompt=base_prompt)
    chain = prompt | llm
# 构建上下文
    def node(state: AgentState):
        new_state = state.copy()
        print(f"\n====={name}开始工作=====")
        context = "\n\n".join([
            f"用户需求:{state['user_requirement']}",
            f"项目计划:{state.get('project_plan', '')}",
            f"需求文档:{state.get('requirements_doc', '')}",
            f"架构设计:{state.get('architecture_doc', '')}",
            f"实现代码:{state.get('code', '')}",
            f"代码审查意见:{state.get('code_review', '')}"
        ])
        tool_result = "无"
        max_tool_calls = 3
        result_text = ""
        try:
            for i in range(max_tool_calls):
                msg = f"[{name}] 第{i+1}轮思考..."
                if log_callback: log_callback(msg)
                print(msg)

                result = chain.invoke({
                    "task": f"完成你的{name}工作",
                    "context": context,
                    "tool_result": tool_result
                })
                result_text = result.content if isinstance(result, AIMessage) else str(result)
                
                msg = f"[{name}] LLM返回内容：{result_text[:150]}..."
                if log_callback: log_callback(msg)
                print(msg)

                if allowed_tools:
                    tool_info = safe_parse_tool_call(result_text)
                    if tool_info:
                        msg = f"[{name}] 正在调用工具：{tool_info['name']}"
                        if log_callback: log_callback(msg)
                        print(msg)
                        
                        tool_result = execute_tool(tool_info)
                        msg = f"工具执行结果：{tool_result[:100]}..."
                        if log_callback: log_callback(msg)
                        print(msg)
                    else:
                        if i < max_tool_calls - 1:
                            tool_result = (
                                "工具调用格式错误!请严格遵守以下规则:\n"
                                "1.只输出纯JSON数组,不要加任何```json、解释、说明、markdown格式\n"
                                "2.正确格式：[{\"name\":\"工具名\",\"parameters\":{\"参数名\":\"参数值\"}}]"
                            )
                        else:
                            msg = f"[{name}] 已达到最大重试次数"
                            if log_callback: log_callback(msg)
                            print(msg)
                else:
                    break
#代码专家兜底
            if name == "代码实现专家":
                msg = "代码实现专家兜底：检查工具调用结果..."
                if log_callback: log_callback(msg)
                print(msg)
                if isinstance(tool_result, str) and len(tool_result) > 200:
                    full_code = tool_result
                else:
                    code_match = re.search(r'```python\s*(.*?)\s*```', result_text, re.DOTALL)
                    if not code_match:
                        code_match = re.search(r'"content"\s*:\s*"(.*?)"', result_text, re.DOTALL)
                        full_code = ""
                    if code_match:
                        full_code = code_match.group(1).replace('\\n', '\n').replace('\\t', '\t')
                try:
                    with open(os.path.join(PROJECT_OUTPUT_DIR, "todo_manager.py"), "w", encoding="utf-8") as f:
                        f.write(full_code)
                    print(f"代码已自动保存到{os.path.join(PROJECT_OUTPUT_DIR, 'todo_manager.py')}")
                except Exception as e:
                    print(f"保存代码失败:{str(e)}")
#返回完整代码
                new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
                new_state[output_key] = full_code
                new_state["current_step"] = name
                new_state["fix_attempts"] = state.get('fix_attempts', 0) + 1
                msg = f"✅ {name} 工作完成"
                if log_callback: log_callback(msg)
                print(msg)
                return new_state
#审查专家兜底
            if name == "代码审查专家":
                msg = "代码审查专家兜底:自动读取代码文件进行审查"
                if log_callback: log_callback(msg)
                print(msg)
                result_text = ""
                try:
                    code_path = os.path.join(PROJECT_OUTPUT_DIR, "todo_manager.py")
                    if os.path.exists(code_path):
                        with open(code_path, "r", encoding="utf-8") as f:
                            code_content = f.read()
                        if len(code_content.strip()) == 0:
                            print("代码文件为空,审查自动判定为不通过")
                            result_text = (
                                "审查结果:不通过\n"
                                "问题列表:\n"
                                "1.代码文件为空,无法运行或执行任何功能\n"
                                "修改建议:请代码实现专家重新生成完整代码。"
                            )
                        else:
                            tool_result = (
                                f"已自动读取到代码文件,内容如下：\n"
                                f"```python\n{code_content[:3000]}\n```\n"
                                "请基于提供的代码进行全面审查"
                            )
                            result = chain.invoke({
                                "task": "基于提供的代码进行全面审查",
                                "context": context,
                                "tool_result": tool_result
                            })
                            if isinstance(result, AIMessage):
                                result_text = result.content
                            else:
                                result_text = str(result)
                    else:
                        print("未找到代码文件,审查无法进行")
                        result_text = (
                            "审查结果:不通过\n"
                            "问题列表:\n"
                            "1.未找到todo_manager.py文件\n"
                            "修改建议:请代码实现专家先生成代码。"
                        )
                except Exception as e:
                    print(f"自动审查失败：{str(e)}")
                    result_text = f"代码审查执行失败:{str(e)}\n审查结果:不通过"
#保存审查报告到文件
                try:
                    review_path = os.path.join(PROJECT_OUTPUT_DIR, "code_review.md")
                    with open(review_path, "w", encoding="utf-8") as f:
                        f.write(f"#代码审查报告\n\n{result_text}")
                    print(f"代码审查报告已自动保存到 {review_path}")
                except Exception as e:
                    print(f"审查报告保存失败:{str(e)}")

                new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
                if '审查结果：不通过' in result_text:
                    new_state[output_key] = result_text
                else:
                    new_state[output_key] = result_text
                new_state["current_step"] = name
                new_state["fix_attempts"] = state.get('fix_attempts', 0) + 1
                msg = f"✅ {name} 工作完成"
                if log_callback: log_callback(msg)
                print(msg)
                return new_state
#通用节点处理
            if allowed_tools and tool_result and tool_result != "无" \
                    and not tool_result.startswith("错误") \
                    and not tool_result.startswith("工具执行失败") \
                    and not tool_result.startswith("架构图已生成"):
                result_text = tool_result
            new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
            new_state[output_key] = result_text
            new_state["current_step"] = name
            new_state["fix_attempts"] = state.get('fix_attempts', 0) + 1
            return new_state
        except Exception as e:
            msg = f"❌ {name} 节点异常：{str(e)}"
            if log_callback: log_callback(msg)
            print(msg)
            new_state["messages"] = [HumanMessage(content=f"{name}工作完成")]
            new_state[output_key] = f"执行失败：{str(e)}"
            new_state["current_step"] = name
            new_state["fix_attempts"] = state.get('fix_attempts', 0) + 1
            new_state["error"] = str(e)
            return new_state
    return node
#复审
def should_fix_code(state: AgentState):
    """条件判断函数：决定是否需要修复代码，同时传递审查意见"""
    review_result = state.get('code_review', '')
    fix_attempts = state.get('fix_attempts', 0)
    
    if fix_attempts >= 3:
        print(f'已达到最大修复次数({fix_attempts}次)，继续下一步')
        return 'paper_writer'
    
    if '审查结果：通过' in review_result or '审核结果：通过' in review_result:
        print('代码审核通过，进入论文撰写阶段')
        return 'paper_writer'
    if review_result:
        problems = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)', review_result, re.DOTALL)
        if problems:
            print(f'代码审核不通过，发现 {len(problems)} 个问题')
            for p in problems[:3]:
                print(f'   - {p.strip()[:80]}...')
            return 'code_expert'
    print('代码审核不通过，开始修复')
    return 'code_expert'
#保洁
def cleaner_agent_node(state: AgentState, log_callback=None):
    msg = "\n===== 保洁Agent 开始精准清理 ====="
    if log_callback: log_callback(msg)
    print(msg)
    new_state = state.copy()
    def clean_text(text: str) -> str:
        if not text:
            return text
        text = re.sub(r'```mermaid\s*.*?\s*```', '', text, flags=re.DOTALL)
        text = re.sub(r'\[\s*\{[^}]*"name"\s*:[^}]*}\s*\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"parameters"\s*:\s*\{[^}]*\}\s*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    for field in ['project_plan', 'requirements_doc', 'architecture_doc', 'code']:
        original = state.get(field, '')
        if original:
            cleaned = clean_text(original)
            if cleaned and len(cleaned) > 50:
                new_state[field] = cleaned
                if cleaned != original:
                    print(f"已清理 [{field}] 字段，原始长度 {len(original)} -> 清洗后长度 {len(cleaned)}")
            elif cleaned:
                print(f"[{field}] 清洗后内容过短，保留原始内容")
    new_state["messages"] = [HumanMessage(content="保洁Agent工作完成")]
    new_state["current_step"] = "上下文清理"
    msg = "🧹 ===== 保洁Agent 完成 ====="
    if log_callback: log_callback(msg)
    print(msg)
    return new_state
#最终报告
def final_acceptance_node(state: AgentState, log_callback=None):
    msg = '\n正在生成最终验收报告...'
    if log_callback: log_callback(msg)
    print(msg)
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一位高级项目验收专家,请根据以下项目产出，生成一份完整的最终验收报告.'),
        ('human', """
用户需求：{user_requirement}
项目计划：{project_plan}
需求文档：{requirements_doc}
架构设计：{architecture_doc}
代码实现：{code}
技术论文：{paper}
请基于以上信息生成最终验收报告。
注意:项目计划、需求文档、架构设计等内容中可能包含原始的JSON数据.
请忽略所有被 [ 和 ] 包裹的原始JSON,只提取其中的核心文本内容进行总结和汇报.
请直接输出一份完整的、格式化的纯文本验收报告,不要输出任何JSON或代码块。
""")
    ])
    chain = prompt | llm
    final_report = chain.invoke({
    'user_requirement': state.get('user_requirement', ''),
    'project_plan': state.get('project_plan', ''),
    'requirements_doc': state.get('requirements_doc', ''),
    'architecture_doc': state.get('architecture_doc', ''),
    'code': state.get('code', ''),
    'paper': state.get('paper', '')
}).content
    msg = '项目验收完成'
    if log_callback: log_callback(msg)
    print(msg)
    return { 'final_report': final_report, 'current_step': '最终验收' }
#工作流
def build_agent_graph(log_callback=None):
    """带代码审查闭环和保洁的工作流图"""
    project_manager_node = create_tool_agent_node(
        name='项目管理QA',
        system_prompt="""
    你是一名资深的项目经理和质量保证专家,精通敏捷开发方法论.
    你的任务是为项目制定详细的开发计划,并在最后进行最终验收.
    你的输出必须包含：
    1.项目总览和目标
    2.详细的时间线和里程碑
    3.任务分解和责任人
    4.质量标准和验收条件
    5.风险评估和应对措施
    你可以使用generate_mermaid工具生成项目甘特图.
    你必须输出一个纯粹的 JSON,不要包含任何多余字符。JSON内部可以使用换行符 \\n,但必须确保字符串被转义。不要使用真正的换行符在 JSON 字符串内。
    请务必同时使用write_file生成计划文档，并使用generate_mermaid生成甘特图。
        例如，先生成一个 'project_plan.md' 文件，再生成图表。
    """,
        output_key='project_plan',
        allowed_tools=['generate_mermaid','write_file'],
        log_callback=log_callback
    )
    requirements_analyst_node = create_tool_agent_node(
        name='需求分析师',
        system_prompt="""
    你是一名经验丰富的AI项目需求分析师.
    你的任务是将用户模糊的需求转化为结构化,可以执行的技术需求文档.
    你的输出必须包含：
    1.项目的概述与价值
    2.核心功能项目(3-5个)
    3.技术指标和性能要求
    4.推荐技术栈及理由
    5.项目里程碑
    你可以使用write_file工具将需求文档保存为requirements.md
    你必须输出一个纯粹的 JSON,不要包含任何多余字符.JSON内部可以使用换行符 \\n,但必须确保字符串被转义.不要使用真正的换行符在 JSON 字符串内.
    """,
        output_key='requirements_doc',
        allowed_tools=['write_file'],
        log_callback=log_callback
    )
    architect_node = create_tool_agent_node(
        name='架构设计师',
        system_prompt="""
    你是一位世界级的AI系统架构师,精通transformer,RAG,Agent等所有主流框架.
    你的任务是根据需求文档，设计出高性能，可拓展的系统架构.
    你的输出必须包含：
    1.整体架构图(文字描述)
    2.核心模块设计
    3.技术选型说明
    4.数据流图
    5.性能优化建议
    你必须使用generate_mermaid工具生成系统架构图.
    你必须输出一个纯粹的 JSON,不要包含任何多余字符。JSON内部可以使用换行符 \\n,但必须确保字符串被转义。不要使用真正的换行符在 JSON 字符串内。
    请务必同时使用write_file生成计划文档，并使用generate_mermaid生成甘特图。
        例如，先生成一个 'project_plan.md' 文件，再生成图表。
    """,
        output_key='architecture_doc',
        allowed_tools=['generate_mermaid','write_file'],
        log_callback=log_callback
    )
    code_expert_node = create_tool_agent_node(
        name='代码实现专家',
        system_prompt="""
    你是一名顶尖 Python 开发工程师，尤其擅长大模型和 AI 系统开发.
    你的任务是根据架构设计，生成高质量、可运行的 Python 代码.
    你的输出必须是纯 Python 代码，用 ```python ``` 包裹.
    绝对不要生成任何形式的图表代码、Mermaid 代码或架构图.
    绝对不要输出 JSON 格式的工具调用.
    如果你生成了图表，任务就失败了
    系统会自动提取你的代码块并保存为文件.
    如果你需要运行代码验证,请输出完整的代码块,系统会自动处理.
        """,
        output_key='code',
        allowed_tools=['write_file'],
        log_callback=log_callback
    )
    paper_writer_node = create_tool_agent_node(
        name='论文写作助手',
        system_prompt="""
    你是一位AI领域学术论文专家,熟悉NeurIPS,ICML等顶会格式.
    你的任务是将项目转化成一篇完整的技术论文.
    你的输出必须包含:
    1.摘要
    2.引言
    3.相关工作
    4.方法
    5.实验与结果
    6.结论
    你可以使用write_file工具将论文保存为paper.md
    你必须输出一个纯粹的 JSON,不要包含任何多余字符.JSON内部可以使用换行符 \\n,但必须确保字符串被转义.不要使用真正的换行符在 JSON 字符串内.
    """,
        output_key='paper',
        allowed_tools=['write_file'],
        log_callback=log_callback
    )
    code_reviewer_node = create_tool_agent_node(
        name="代码审查专家",
        system_prompt="""
    你是一位拥有10年经验的资深Python代码审查专家,眼光极其挑剔.
    你的任务是严格审查代码专家生成的代码，找出所有问题.
    你必须检查以下所有方面:
    1. 语法错误：是否有拼写错误、缺少括号、缩进错误等
    2. 逻辑错误：代码逻辑是否正确，是否能实现预期功能
    3. 边界条件：是否处理了空输入、异常情况、极端值
    4. 代码规范：变量命名是否清晰，是否有必要的注释
    5. 性能问题：是否有明显的性能瓶颈
    6. 安全问题：是否有潜在的安全漏洞
    你的输出必须严格按照以下格式:
    审查结果：[通过/不通过]
    问题列表:
    1. 问题1描述
    2. 问题2描述
    ...
    修改建议:
    1. 建议1
    2. 建议2
    ...
    如果代码没有任何问题，就输出:
    审查结果：通过
    问题列表：无
    修改建议：无
    绝对不要说模棱两可的话，有问题就明确指出来，没问题就直接通过:
    重要提醒：第一行必须是"审查结果：通过"或者"审查结果：不通过"
    你必须输出一个纯粹的 JSON,不要包含任何多余字符.JSON 内部可以使用换行符 \\n,但必须确保字符串被转义.不要使用真正的换行符在 JSON 字符串内.
    """,
        output_key="code_review",
        allowed_tools=[],
        log_callback=log_callback
    )
    workflow = StateGraph(AgentState)

    workflow.add_node("project_manager", project_manager_node)
    workflow.add_node("requirements_analyst", requirements_analyst_node)
    workflow.add_node("architect", architect_node)
    workflow.add_node("cleaner", lambda state: cleaner_agent_node(state, log_callback=log_callback))      
    workflow.add_node("code_expert", code_expert_node)
    workflow.add_node("code_reviewer", code_reviewer_node)
    workflow.add_node("paper_writer", paper_writer_node)
    workflow.add_node("final_acceptance", lambda state: final_acceptance_node(state, log_callback=log_callback))   

    workflow.set_entry_point("project_manager")

    workflow.add_edge("project_manager", "requirements_analyst")
    workflow.add_edge("requirements_analyst", "architect")
    workflow.add_edge("architect", "cleaner")                  
    workflow.add_edge("cleaner", "code_expert")               
    workflow.add_edge("code_expert", "code_reviewer")

    workflow.add_conditional_edges(
        "code_reviewer",
        should_fix_code,
        {
            "code_expert": "code_expert",
            "paper_writer": "paper_writer"
        }
    )
    workflow.add_edge("code_expert", "code_reviewer")
    workflow.add_edge("paper_writer", "final_acceptance")
    workflow.add_edge("final_acceptance", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
#断点续跑
def get_latest_checkpoint():
    """查找最新的输出目录，用于断点续跑"""
    output_dirs = sorted(
        [d for d in os.listdir(BASE_DIR) if d.startswith("project_output_")],
        reverse=True
    )
    if output_dirs:
        latest = os.path.join(BASE_DIR, output_dirs[0])
        state_file = os.path.join(latest, "agent_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f), latest
    return None, None
def save_checkpoint(state_dict, output_dir):
    state_file = os.path.join(output_dir, "agent_state.json")
    saveable = {
        "user_requirement": state_dict.get("user_requirement", ""),
        "chat_history": state_dict.get("chat_history", []),
        "is_completed": state_dict.get("is_completed", False),
        "last_checkpoint": state_dict.get("last_checkpoint", "start"),
        "fix_attempts": state_dict.get("fix_attempts", 0),
        "project_plan": state_dict.get("project_plan", ""),      
        "requirements_doc": state_dict.get("requirements_doc", ""),
        "architecture_doc": state_dict.get("architecture_doc", ""),
        "code": state_dict.get("code", ""),
        "paper": state_dict.get("paper", ""),
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)
#启动
if __name__ == "__main__":
    PROJECT_OUTPUT_DIR = os.path.join(BASE_DIR, f"project_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(PROJECT_OUTPUT_DIR, exist_ok=True)
    print(f"项目输出目录：{PROJECT_OUTPUT_DIR}")
    print("五人智囊团多Agent系统启动成功!")
    print("基于 langgraph 1.1.8 和 langchain 1.2.15 构建")
    print("输入'exit'退出系统，输入'continue'断点续跑\n")
    app = build_agent_graph()
    chat_history = []
    while True:
        user_input = input("请输入你的项目要求：")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "continue":
            saved_state, saved_dir = get_latest_checkpoint()
            if saved_state and saved_dir:
        # # 1. 询问是否继续
                resume = input("是否继续上次任务？(y/n)：")
                if resume.lower() == 'y':
                    chat_history = saved_state.get('chat_history', [])
            
            # 2. 询问是否清洗旧状态
                    clean_old = input("是否清洗旧状态以避免上下文污染？(y/n，默认y)：")
                    if clean_old.lower() != 'n':
                        saved_state['project_plan'] = ''
                        saved_state['requirements_doc'] = ''
                        saved_state['architecture_doc'] = ''
                        saved_state['code'] = ''
                        saved_state['paper'] = ''
                        saved_state['code_review'] = ''
                        saved_state['fix_attempts'] = 0
                        print("旧状态已清洗")
                    PROJECT_OUTPUT_DIR = saved_dir
                    print("已恢复对话历史")
                    continue
                else:
                    print("开始新任务...")
            else:
                print("未找到可恢复的检查点")
            continue
        print(f"收到要求：{user_input}")
        print("五人智囊团开始工作...\n")
        if chat_history and "修改" in user_input:
            last_task = chat_history[-1].get('task', '')
            user_input = f"修改以下任务的部分内容：\n原任务：{last_task}\n修改要求：{user_input}"
        initial_state = {
            'user_requirement': user_input,
            'messages': [HumanMessage(content=user_input)],
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
            'chat_history': chat_history,      
            'is_completed': False,             
            'last_checkpoint': 'start'          
        }
        config = {'configurable': {'thread_id': f'chat_{len(chat_history)}'}}
        print("\n五人智囊团开始工作，实时进度如下：\n")
        try:
            for step in app.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in step.items():
                    print(f"\n{'='*50}")
                    print(f"节点 [{node_name}] 执行完成")
                    print(f"输出字段：{list(node_output.keys())}")
                    if 'current_step' in node_output:
                        print(f"📍 当前阶段：{node_output['current_step']}")
                    print(f"{'='*50}\n")
                    current_state = app.get_state(config).values
                    save_checkpoint(current_state, PROJECT_OUTPUT_DIR)
            final_state = app.get_state(config).values
            def safe_get(key: str) -> str:
                return final_state.get(key, '')
            report = f"""# 项目最终验收报告
## 用户需求
{safe_get('user_requirement')}
## 项目计划
{safe_get('project_plan')}
## 需求文档
{safe_get('requirements_doc')}
## 架构设计
{safe_get('architecture_doc')}
## 代码实现
{safe_get('code')}
## 技术论文
{safe_get('paper')}
## 最终验收报告
{safe_get('final_report')}
"""
            report_path = os.path.join(PROJECT_OUTPUT_DIR, 'project_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            chat_history.append({
                'task': user_input,
                'output_dir': PROJECT_OUTPUT_DIR,
                'summary': safe_get('final_report')[:200] if safe_get('final_report') else '未生成'
            })
            final_state_with_flag = final_state.copy()
            final_state_with_flag['is_completed'] = True
            save_checkpoint(final_state_with_flag, PROJECT_OUTPUT_DIR)
            print('\n项目完成! 完整报告已保存到 project_report.md')
            print(f'当前对话轮次：{len(chat_history)}')
            print('\n=== 报告摘要 ===')
            print((safe_get('final_report') or safe_get('project_plan') or safe_get('code'))[:500] + '...\n')
        except KeyboardInterrupt:
            print('\n任务已暂停，进度已保存。下次输入 continue 可继续。')
            current_state = app.get_state(config).values
            save_checkpoint(current_state, PROJECT_OUTPUT_DIR)
        except Exception as e:
            print(f'\n任务异常：{str(e)}')
            print('输入 continue 可尝试恢复')
            try:
                current_state = app.get_state(config).values
                save_checkpoint(current_state, PROJECT_OUTPUT_DIR)
            except:
                pass