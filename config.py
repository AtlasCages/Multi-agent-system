"""
配置中心
所有常量、默认值、环境变量统一管理
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────
LLM_MODEL = os.getenv("TEAM_LLM_MODEL", "Qwen/Qwen3-8B")
LLM_BASE_URL = os.getenv("TEAM_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.getenv("TEAM_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or ""
LLM_TEMPERATURE = float(os.getenv("TEAM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("TEAM_MAX_TOKENS", "8192"))
LLM_TIMEOUT = int(os.getenv("TEAM_TIMEOUT", "500"))
LLM_MAX_RETRIES = int(os.getenv("TEAM_MAX_RETRIES", "3"))

# ── 系统 ─────────────────────────────────────
MAX_FIX_ATTEMPTS = 3           # 代码审查修正最大次数
OUTPUT_DIR_PREFIX = "project_output_"  # 输出目录前缀

# ── 文件 ─────────────────────────────────────
SAVED_STATE_FILENAME = "agent_state.json"
REPORT_FILENAME = "project_report.md"
