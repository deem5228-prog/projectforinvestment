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
# gemini-2.0-flash-lite: ถูกที่สุด เร็วที่สุด โควต้า Free Tier สูงสุด (~1,500 req/day)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")


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
    from google.genai.errors import APIError
    import time

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    max_retries = 8
    backoff = 3.0

    for attempt in range(max_retries):
        try:
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
        except APIError as e:
            # Check if it's a 429 rate limit error
            err_str = str(e)
            if getattr(e, "code", None) == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt == max_retries - 1:
                    logger.error("[llm] Rate limit exceeded, max retries reached: %s", e)
                    raise e
                
                # Parse exact retry delay from the error message if present
                sleep_time = None
                # Pattern 1: "Please retry in 46.997308722s."
                m = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
                if m:
                    sleep_time = float(m.group(1)) + 2.0  # Add 2s safety buffer
                else:
                    # Pattern 2: "'retryDelay': '16s'"
                    m2 = re.search(r"retryDelay':\s*'(\d+)s'", err_str)
                    if m2:
                        sleep_time = float(m2.group(1)) + 2.0
                
                if sleep_time is None:
                    sleep_time = backoff * (2 ** attempt)

                logger.warning(
                    "[llm] Rate limit (429) hit. Waiting %.1fs before retry (attempt %d/%d)...",
                    sleep_time, attempt + 1, max_retries
                )
                time.sleep(sleep_time)
            else:
                raise e
        except Exception as e:
            raise e

    raise RuntimeError("LLM call failed after retries")


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
