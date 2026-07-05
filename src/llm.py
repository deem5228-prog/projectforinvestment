# ── llm.py ───────────────────────────────────────────────────────────────────
# Shared LLM helper — ใช้ Google Gemini แทน Anthropic
# ─────────────────────────────────────────────────────────────────────────────
"""
ใช้งาน:
    from src.llm import call_llm_json

    data = call_llm_json(
        system_prompt="You are Warren Buffett...",
        user_prompt="Facts: {...}\\nReturn JSON only.",
        fallback={"signal": "neutral", "confidence": 50, "reasoning": "error"},
    )
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# โมเดลที่ใช้ — เปลี่ยนได้ผ่าน .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    เรียก Gemini API และคืน text response ดิบ

    Returns:
        str: text ที่โมเดลตอบกลับมา

    Raises:
        Exception: ถ้า API call ล้มเหลว (ให้ caller จัดการ fallback)
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    return response.text.strip()


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    fallback: dict,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> dict:
    """
    เรียก Gemini แล้ว parse JSON อัตโนมัติ พร้อม fallback

    Returns:
        dict: parsed JSON หรือ fallback ถ้า error
    """
    raw = ""
    try:
        raw = call_llm(system_prompt, user_prompt, max_tokens, temperature)

        # ── Extract JSON from the raw response ───────────────────────────
        # Stage 1: JSON inside a code fence  ```json { ... } ```
        cleaned = raw
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(1))

        # Stage 2: first bare JSON object anywhere in the text
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))

        # Stage 3: try the whole string as-is
        return json.loads(cleaned.strip())

    except json.JSONDecodeError as e:
        logger.warning("[llm] JSON parse failed: %s | raw: %.200s", e, raw)
        return fallback
    except Exception as e:
        logger.warning("[llm] Gemini call failed: %s", e)
        return fallback
