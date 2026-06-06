"""
工具插件注册系统
"""

import inspect
from typing import Callable, Dict, Any

# 全局注册表
_registry: Dict[str, dict] = {}


def tool_plugin(name: str, description: str):
    """
    装饰器：将函数注册为可用工具

    用法:
        @tool_plugin("函数名", "描述")
        def my_func(param1: str, param2: int = 0) -> str:
            ...
    """
    def decorator(func: Callable):
        sig = inspect.signature(func)
        param_info = []
        for p_name, p_param in sig.parameters.items():
            p_info = {
                "name": p_name,
                "required": p_param.default is inspect.Parameter.empty,
            }
            if p_param.annotation is not inspect.Parameter.empty:
                p_info["type"] = str(p_param.annotation)
            param_info.append(p_info)

        _registry[name] = {
            "name": name,
            "description": description,
            "func": func,
            "parameters": param_info,
        }
        return func
    return decorator


def get_tool_descriptions() -> str:
    """动态生成工具描述文本（给 LLM 的 prompt）"""
    lines = ["你可以使用以下工具来完成任务:"]
    for t in _registry.values():
        params = ", ".join(f"{p['name']}" for p in t["parameters"])
        lines.append(f"- {t['name']}: {t['description']} (参数: {params})")
    lines.append("")
    lines.append("工具调用规则:")
    lines.append("1. 当你需要使用工具时，直接使用 function calling 格式")
    lines.append("2. 不要自行编造 JSON——让 LLM 的 function calling 机制处理")
    return "\n".join(lines)


def list_tools() -> list:
    """返回工具列表（供 bind_tools 使用）"""
    result = []
    for t in _registry.values():
        result.append(t)
    return result


def execute_tool(tool_name: str, parameters: dict) -> str:
    """执行已注册的工具"""
    tool = _registry.get(tool_name)
    if not tool:
        return f"错误：未知工具 '{tool_name}'"

    func = tool["func"]
    sig = inspect.signature(func)

    # 过滤合法参数
    filtered = {k: v for k, v in parameters.items() if k in sig.parameters}
    try:
        result = str(func(**filtered))
        # write_file 特殊处理：返回的是写入内容本身
        if tool_name == "write_file" and parameters.get("content"):
            return parameters["content"]
        return result
    except Exception as e:
        return f"工具执行失败: {e}"
