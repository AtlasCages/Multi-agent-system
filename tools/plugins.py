"""
内置工具插件
"""

import os
import re
import json

from agent_team.tools.registry import tool_plugin


# 在运行时由 main / web_ui 注入
_OUTPUT_DIR = None


def set_output_dir(path: str):
    global _OUTPUT_DIR
    _OUTPUT_DIR = path


def get_output_dir() -> str:
    global _OUTPUT_DIR
    return _OUTPUT_DIR or os.getcwd()


@tool_plugin("run_code", "运行 Python 代码。参数: code — 要执行的 Python 代码字符串")
def run_code(code: str) -> str:
    """安全执行 Python 代码（限制内置函数）"""
    try:
        safe_builtins = {
            "print": print, "len": len, "range": range,
            "int": int, "float": float, "str": str, "list": list,
            "dict": dict, "tuple": tuple, "set": set, "bool": bool,
            "max": max, "min": min, "sum": sum, "abs": abs,
            "round": round, "sorted": sorted, "enumerate": enumerate,
            "True": True, "False": False, "None": None,
        }
        exec(code, {"__builtins__": safe_builtins})
        return "代码执行成功"
    except Exception as e:
        return f"代码执行异常: {e}"


@tool_plugin("write_file", "将内容写入本地文件。参数: file_path — 文件名, content — 文件内容")
def write_file(file_path: str, content: str) -> str:
    """写入文件到输出目录"""
    try:
        full_path = os.path.join(get_output_dir(), os.path.basename(file_path.lstrip("/\\")))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已保存: {full_path}"
    except Exception as e:
        return f"文件写入失败: {e}"


@tool_plugin("generate_mermaid", "生成 Mermaid 图表。参数: chart_code — mermaid 代码, file_name — 保存的文件名(默认 architecture.md)")
def generate_mermaid(chart_code: str, file_name: str = "architecture.md") -> str:
    """生成 Mermaid 图文件"""
    try:
        full_path = os.path.join(get_output_dir(), os.path.basename(file_name))
        code = chart_code.strip().replace("\\n", "\n").replace("\\t", "\t")
        content = f"```mermaid\n{code}\n```"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"架构图已生成: {full_path}"
    except Exception as e:
        return f"架构图生成失败: {e}"
