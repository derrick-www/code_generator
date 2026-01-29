#!/usr/bin/env python3
"""
generate_and_send.py (支持 Gemini via google-genai, OpenAI, 或通用 generic HTTP endpoints)

环境变量（主要）:
- FEISHU_WEBHOOK: 飞书自定义机器人 Webhook（必填）
- LLM_PROVIDER: "openai" / "gemini" / "generic"（若未设置，会基于其他变量自动推断）
- LLM_API_KEY: 用于 openai 或 generic (Bearer) 的 key / token
- LLM_API_URL: generic provider 的 endpoint（当 LLM_PROVIDER=generic 时需要）
- GEMINI_API_KEY: 若使用 google-genai 的 API key（可选，优先于 LLM_API_KEY）
- GEMINI_MODEL: 要调用的 Gemini 模型（可选，默认 "gemini-1.5"）
"""
from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

# Try OpenAI new/old clients
HAS_OPENAI_V1 = False
HAS_OPENAI_OLD = False
try:
    from openai import OpenAI  # new client
    HAS_OPENAI_V1 = True
except Exception:
    try:
        import openai  # legacy
        HAS_OPENAI_OLD = True
    except Exception:
        pass

# Try google-genai client availability will be checked at runtime in call_gemini
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Environment / config
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or None
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL")
# Gemini-specific vars (prefer these if present)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-1.5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# If LLM_API_URL is set but provider is openai or empty, auto-switch to generic
if LLM_API_URL and (LLM_PROVIDER in (None, "", "openai")):
    logging.info("Detected LLM_API_URL is set; forcing LLM_PROVIDER='generic'")
    LLM_PROVIDER = "generic"

# If GEMINI_API_KEY present and provider not set, use gemini
if GEMINI_API_KEY and not LLM_PROVIDER:
    logging.info("Detected GEMINI_API_KEY; setting LLM_PROVIDER='gemini'")
    LLM_PROVIDER = "gemini"

# Default provider fallback
if not LLM_PROVIDER:
    LLM_PROVIDER = "openai"  # keep backward compatibility
logging.info("LLM_PROVIDER=%s", LLM_PROVIDER)


PROMPT_USER = (
    "请**随机**选择一道力扣算法题，并给出详细的题解。"
    " 返回内容必须是严格的 JSON（不要包含其他文本），格式如下："
    '{"problem_id": "...", "title": "...", "difficulty": "...", "description": "...", "solution": "...", "code": "...", "complexity": "...", "link": "..."}。'
    " 各字段说明：\n"
    "- problem_id: 题目编号（例如：1, 2, 15, 206 等）\n"
    "- title: 题目名称（中文或英文）\n"
    "- difficulty: 难度等级（Easy / Medium / Hard）\n"
    "- description: 题目描述（简要说明题目要求，2-3 句话）\n"
    "- solution: 解题思路（详细讲解解题方法、算法思想，3-5 句话）\n"
    "- code: 代码实现（使用 C++ 包含注释）\n"
    "- complexity: 时间复杂度和空间复杂度分析（例如：时间复杂度 O(n)，空间复杂度 O(1)）\n"
    "- link: 力扣题目链接（格式：https://leetcode.cn/problems/xxx/ 或 https://leetcode.com/problems/xxx/）\n"
    "请保证输出是单纯的 JSON 对象，且能被标准 JSON 解析。"
)


# ------------------------------
# Gemini (google-genai) 调用
# ------------------------------
def call_gemini(api_key: str, model: str, prompt: str, temperature: float = 0.8) -> str:
    """
    使用 google-genai 客户端调用 Gemini（参考 daily_stock_analysis 中的做法）。
    - api_key: google genai API key（通常以 'AIza...' 或其他形式）
    - model: 模型名（如 gemini-1.5 / gemini-2.5 等）
    返回字符串（LLM 的原始文本）
    """
    try:
        # from google import genai
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(
            "google-genai not installed. Install with: pip install google-genai\n" f"orig: {e}"
        )

    try:
        SYSTEM_PROMPT = """你是一位大模型专家，精通大模型面试题，随机给出一个大模型面试题，并给出详细的答案。"""
        logging.info("Calling Gemini via google-genai (model=%s)", model)
        
        genai.configure(api_key=api_key)
        model_name = "gemini-3-flash-preview"
        gemini_model = genai.GenerativeModel(
                    model_name=model_name
                )
        max_retries = 3
        base_delay = 5.0
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 8192,
        }
        for attempt in range(max_retries):
        
            # 请求前增加延时（防止请求过快触发限流）
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                delay = min(delay, 60)  # 最大60秒
                logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                time.sleep(delay)
            
            response = gemini_model.generate_content(
                PROMPT_USER,
                generation_config=generation_config,
                request_options={"timeout": 120}
            )
            
            if response and response.text:
                return response.text
            else:
                raise RuntimeError(f"Gemini 请求失败: {response.text}")
        
    except Exception as e:
        # rethrow with context
        raise RuntimeError(f"Gemini call failed: {e}")

# ------------------------------
# 辅助函数：解析 / 格式化 / 发送到飞书
# ------------------------------
def parse_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def format_message(data: dict) -> str:
    message = {
        "msg_type": "interactive",
        "card": {
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "content": f"**题目ID:** {data['problem_id']}\n"
                                  f"**难度:** {data['difficulty']}\n"
                                  f"**链接:** [点击查看]({data['link']})",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "hr"
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"**题目描述**\n{data['description']}",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"**解题思路**\n{data['solution']}",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"**时间复杂度**\n{data['complexity']}",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "hr"
                },
                {
                    "tag": "div",
                    "text": {
                        "content": "**代码实现**",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "div",
                    "text": {
                        "content": f"```cpp\n{data['code']}\n```",
                        "tag": "lark_md"
                    }
                }
            ],
            "header": {
                "template": "wathet",
                "title": {
                    "content": f"📝 LeetCode题目：{data['title']}",
                    "tag": "plain_text"
                }
            }
        }
    }
    return message
    # title = data.get("title", "").strip()
    # description = data.get("description", "").strip()
    # solution = data.get("solution", "").strip()
    # code = data.get("code", "").strip()
    # source = data.get("source", "").strip()
    # parts = []
    # if title:
    #     parts.append(f"题目：{title}")
    # if description:
    #     parts.append(f"描述：{description}")
    # if solution:
    #     parts.append(f"题解：{solution}")
    # if code:
    #     parts.append(f"code：{code}")
    
    # return "\n".join(parts)


def send_to_feishu(webhook: str, text: str) -> None:
    if not webhook:
        raise RuntimeError("FEISHU_WEBHOOK is not set.")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"msg_type": "text", "content": {"text": text}}
    logging.info("Sending message to Feishu webhook")
    # r = requests.post(webhook, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    r = requests.post(webhook, headers={"Content-Type": "application/json"},
            data=json.dumps(message)
    try:
        r.raise_for_status()
    except Exception as e:
        logging.error("Feishu webhook returned status %s: %s", r.status_code, r.text)
        raise
    logging.info("Message sent to Feishu successfully.")


# ------------------------------
# 主流程
# ------------------------------
def main():
    if not FEISHU_WEBHOOK:
        logging.error("FEISHU_WEBHOOK 环境变量未设置，退出。")
        return 2

    # Determine which credentials to use
    # Priority for Gemini: GEMINI_API_KEY -> LLM_API_KEY
    gemini_key = LLM_API_KEY

    text = None
    last_err = None
    
    text = call_gemini(gemini_key, GEMINI_MODEL, PROMPT_USER, temperature=0.8)
    parsed = parse_json_from_text(text)
    if parsed:
        message = format_message(parsed)
        logging.info("Parsed JSON and formatted message:\n%s", message)
        send_to_feishu(FEISHU_WEBHOOK, message)
        return 0
    else:
        last_err = f"无法从 LLM 响应中解析出 JSON，响应文本：{(text or '')[:400]}"
        logging.warning("Attempt %d: %s", attempt + 1, last_err)

    logging.debug("LLM raw response: %s", (text or "")[:1000])

    # send_to_feishu(FEISHU_WEBHOOK, parsed)
    return 0
   

   

if __name__ == "__main__":
    raise SystemExit(main())
