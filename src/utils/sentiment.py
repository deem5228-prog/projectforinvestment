"""
News sentiment enrichment via Gemini LLM.

Batches all headlines into a single LLM call and returns the list
with ``CompanyNews.sentiment`` filled in.
"""

import json
import logging
import re

from src.llm import call_llm

logger = logging.getLogger(__name__)

# Maximum headlines to send per LLM batch (keeps prompt short & cheap).
_MAX_BATCH = 50


def enrich_news_sentiment(news: list) -> list:
    """
    Classify each headline's sentiment using a single Gemini call.

    Args:
        news: list of CompanyNews objects (sentiment may be None).

    Returns:
        The *same* list with ``.sentiment`` set to one of:
        ``"positive"`` | ``"negative"`` | ``"neutral"``

        On any error, returns the original list unchanged.
    """
    if not news:
        return news

    # Only enrich items that don't already have sentiment.
    to_enrich = [n for n in news if not n.sentiment]
    if not to_enrich:
        return news

    # Cap the batch so we don't blow up the prompt.
    batch = to_enrich[:_MAX_BATCH]

    titles = [f"{i+1}. {n.title}" for i, n in enumerate(batch)]

    system_prompt = (
        "You are a financial news sentiment classifier.\n"
        "For each numbered headline, respond with exactly one word: "
        "positive, negative, or neutral.\n"
        "Return ONLY a JSON array of strings, one per headline, in order.\n"
        "Example: [\"positive\", \"negative\", \"neutral\"]\n"
        "Do not add any other text."
    )

    user_prompt = "Classify these headlines:\n\n" + "\n".join(titles)

    try:
        raw = call_llm(
            system_prompt,
            user_prompt,
            max_tokens=max(len(batch) * 15, 128),
            temperature=0.0,
        )

        # Parse: extract the JSON array from the response.
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            logger.warning("[sentiment] no JSON array in response: %.200s", raw)
            return news

        labels = json.loads(m.group(0))

        if not isinstance(labels, list) or len(labels) != len(batch):
            logger.warning(
                "[sentiment] expected %d labels, got %s",
                len(batch), type(labels).__name__,
            )
            return news

        # Apply labels back to the CompanyNews objects.
        valid = {"positive", "negative", "neutral", "bullish", "bearish"}
        for item, label in zip(batch, labels):
            val = str(label).strip().lower()
            if val in valid:
                item.sentiment = val
            else:
                item.sentiment = "neutral"

        logger.info(
            "[sentiment] enriched %d/%d headlines",
            len(batch), len(news),
        )

    except json.JSONDecodeError as e:
        logger.warning("[sentiment] JSON parse failed: %s", e)
    except Exception as e:
        logger.warning("[sentiment] enrichment failed: %s", e)

    return news
