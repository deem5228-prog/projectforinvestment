# Caching layer for data fetching
class Cache:
    """In-memory cache for API responses."""

    def __init__(self):
        self._prices_cache: dict[str, list[dict]] = {}
        self._financial_metrics_cache: dict[str, list[dict]] = {}
        self._line_items_cache: dict[str, list[dict]] = {}
        self._insider_trades_cache: dict[str, list[dict]] = {}
        self._company_news_cache: dict[str, list[dict]] = {}

    def _merge_data(
        self,
        existing: list[dict] | None,
        new_data: list[dict],
        key_field: str,
    ) -> list[dict]:
        """Merge existing and new data, deduplicating by key_field."""
        if not existing:
            return new_data
        existing_keys = {item[key_field] for item in existing if key_field in item}
        merged = existing.copy()
        merged.extend([item for item in new_data if item.get(key_field) not in existing_keys])
        return merged

    # ── Prices ──────────────────────────────────────────────────────────────

    def get_prices(self, cache_key: str) -> list[dict] | None:
        return self._prices_cache.get(cache_key)

    def set_prices(self, cache_key: str, data: list[dict]) -> None:
        existing = self._prices_cache.get(cache_key)
        self._prices_cache[cache_key] = self._merge_data(existing, data, key_field="time")

    # ── Financial Metrics ────────────────────────────────────────────────────

    def get_financial_metrics(self, cache_key: str) -> list[dict] | None:
        return self._financial_metrics_cache.get(cache_key)

    def set_financial_metrics(self, cache_key: str, data: list[dict]) -> None:
        existing = self._financial_metrics_cache.get(cache_key)
        self._financial_metrics_cache[cache_key] = self._merge_data(existing, data, key_field="report_period")

    # ── Line Items ───────────────────────────────────────────────────────────

    def get_line_items(self, cache_key: str) -> list[dict] | None:
        return self._line_items_cache.get(cache_key)

    def set_line_items(self, cache_key: str, data: list[dict]) -> None:
        existing = self._line_items_cache.get(cache_key)
        # Long-format rows share the same report_period (e.g. revenue, net_income,
        # total_debt all dated "2024-09-30"). Deduping on report_period alone would
        # keep only the first row per period. Stamp a composite key so _merge_data
        # treats each (period, metric) pair as a distinct row, then remove it.
        _KEY = "_key"
        for item in data:
            item[_KEY] = f"{item.get('report_period', '')}|{item.get('line_item', '')}"
        if existing:
            for item in existing:
                if _KEY not in item:
                    item[_KEY] = f"{item.get('report_period', '')}|{item.get('line_item', '')}"
        merged = self._merge_data(existing, data, key_field=_KEY)
        for item in merged:
            item.pop(_KEY, None)
        self._line_items_cache[cache_key] = merged

    # ── Insider Trades ───────────────────────────────────────────────────────

    def get_insider_trades(self, cache_key: str) -> list[dict] | None:
        return self._insider_trades_cache.get(cache_key)

    def set_insider_trades(self, cache_key: str, data: list[dict]) -> None:
        existing = self._insider_trades_cache.get(cache_key)
        self._insider_trades_cache[cache_key] = self._merge_data(existing, data, key_field="filing_date")

    # ── Company News ─────────────────────────────────────────────────────────

    def get_company_news(self, cache_key: str) -> list[dict] | None:
        return self._company_news_cache.get(cache_key)

    def set_company_news(self, cache_key: str, data: list[dict]) -> None:
        existing = self._company_news_cache.get(cache_key)
        self._company_news_cache[cache_key] = self._merge_data(existing, data, key_field="date")


# Global singleton
_cache = Cache()


def get_cache() -> Cache:
    return _cache