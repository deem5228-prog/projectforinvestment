# API client or fetcher for data retrieval
import datetime
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

_cache = get_cache()


def _next_day(date_str: str) -> str:
    """Return the day after date_str.

    yfinance's ``end`` parameter is exclusive — the end date itself is never
    included in the downloaded range.  Pass every ``end=`` argument through
    this helper so the caller's intended end date is always included.
    """
    return (
        datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(days=1)
    ).strftime("%Y-%m-%d")


def _get_yf_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


# ── Prices ───────────────────────────────────────────────────────────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    try:
        df = yf.download(ticker, start=start_date, end=_next_day(end_date), auto_adjust=True, progress=False)
        if df.empty:
            return []

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.index.name = "time"
        df = df.reset_index()
        df["time"] = df["time"].astype(str)

        prices = [
            Price(
                ticker=ticker,
                time=row["time"],
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row.get("close"),
                volume=int(row.get("volume", 0)),
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.warning("yfinance get_prices failed for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ── Financial Metrics ─────────────────────────────────────────────────────────

def _yf_row(df, label, col):
    """Safely pull a single value from a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    if label not in df.index:
        return None
    val = df.at[label, col]
    if pd.isna(val):
        return None
    return float(val)


def _safe_div(a, b):
    """a / b with None/zero guard; returns float or None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    try:
        stock = _get_yf_ticker(ticker)
        info = stock.info

        # ── Pull statement DataFrames ─────────────────────────────────────
        # Use annual statements for multi-year history.
        income_df  = stock.financials           # annual income statement
        balance_df = stock.balance_sheet        # annual balance sheet
        cashflow_df = stock.cashflow            # annual cash flow

        # Determine available periods (columns) that are <= end_date,
        # sorted newest-first, capped at `limit`.
        all_periods = set()
        for df in [income_df, balance_df, cashflow_df]:
            if df is not None and not df.empty:
                all_periods.update(df.columns)
        periods = sorted(
            [c for c in all_periods if str(c)[:10] <= end_date],
            reverse=True,
        )[:limit]

        if not periods:
            # Fallback to single .info-based snapshot (old behaviour).
            periods = [end_date]

        # ── Current-period-only values from .info ─────────────────────────
        # These valuation/market ratios only exist for the latest snapshot.
        fcf = info.get("freeCashflow")
        shares_out = info.get("sharesOutstanding")
        fcf_per_share = _safe_div(fcf, shares_out)

        info_only_fields = {
            "market_cap":                info.get("marketCap"),
            "enterprise_value":          info.get("enterpriseValue"),
            "price_to_earnings_ratio":   info.get("trailingPE"),
            "price_to_book_ratio":       info.get("priceToBook"),
            "price_to_sales_ratio":      info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda":              info.get("enterpriseToEbitda"),
            "ev_to_revenue":             info.get("enterpriseToRevenue"),
            "current_ratio":             info.get("currentRatio"),
            "dividend_yield":            info.get("dividendYield"),
            "payout_ratio":              info.get("payoutRatio"),
            "beta":                      info.get("beta"),
            "earnings_per_share":        info.get("trailingEps"),
            "book_value_per_share":      info.get("bookValue"),
            "free_cash_flow_per_share":  fcf_per_share,
        }

        # ── Build one FinancialMetrics per period ─────────────────────────
        results: list[FinancialMetrics] = []

        # Pre-extract revenue and net_income series for growth calculations.
        # Map period_str -> value for quick lookups.
        rev_by_period = {}
        ni_by_period  = {}
        for col in periods:
            rev = _yf_row(income_df, "Total Revenue", col)
            ni  = _yf_row(income_df, "Net Income", col)
            col_str = str(col)[:10]
            if rev is not None:
                rev_by_period[col_str] = rev
            if ni is not None:
                ni_by_period[col_str] = ni

        period_strs = [str(c)[:10] for c in periods]

        for i, col in enumerate(periods):
            col_str = str(col)[:10]
            is_latest = (i == 0)

            # ── Raw values from statements ────────────────────────────────
            revenue        = _yf_row(income_df, "Total Revenue", col)
            gross_profit   = _yf_row(income_df, "Gross Profit", col)
            operating_inc  = _yf_row(income_df, "Operating Income", col)
            net_income     = _yf_row(income_df, "Net Income", col)
            eps            = _yf_row(income_df, "Diluted EPS", col)
            interest_exp   = _yf_row(income_df, "Interest Expense", col)

            total_assets   = _yf_row(balance_df, "Total Assets", col)
            total_liab     = _yf_row(balance_df, "Total Liabilities Net Minority Interest", col)
            equity         = _yf_row(balance_df, "Stockholders Equity", col)
            total_debt     = _yf_row(balance_df, "Total Debt", col)
            current_assets = _yf_row(balance_df, "Current Assets", col)
            current_liab   = _yf_row(balance_df, "Current Liabilities", col)

            op_cashflow    = _yf_row(cashflow_df, "Operating Cash Flow", col)
            capex          = _yf_row(cashflow_df, "Capital Expenditure", col)

            # ── Computed ratios ───────────────────────────────────────────
            gross_margin    = _safe_div(gross_profit, revenue)
            operating_margin = _safe_div(operating_inc, revenue)
            net_margin      = _safe_div(net_income, revenue)
            roe             = _safe_div(net_income, equity)
            roa             = _safe_div(net_income, total_assets)
            de_ratio        = _safe_div(total_debt, equity)
            da_ratio        = _safe_div(total_debt, total_assets)
            asset_turnover  = _safe_div(revenue, total_assets)
            cur_ratio       = _safe_div(current_assets, current_liab)
            int_coverage    = _safe_div(operating_inc, abs(interest_exp) if interest_exp else None)

            # Growth vs previous period
            rev_growth = None
            ni_growth  = None
            if i + 1 < len(period_strs):
                prev_str = period_strs[i + 1]
                prev_rev = rev_by_period.get(prev_str)
                prev_ni  = ni_by_period.get(prev_str)
                if revenue is not None and prev_rev is not None and prev_rev != 0:
                    rev_growth = (revenue - prev_rev) / abs(prev_rev)
                if net_income is not None and prev_ni is not None and prev_ni != 0:
                    ni_growth = (net_income - prev_ni) / abs(prev_ni)

            # ── Assemble FinancialMetrics ─────────────────────────────────
            m = FinancialMetrics(
                ticker=ticker,
                report_period=col_str,
                period=period,
                currency=info.get("currency", "USD"),

                # Profitability (computed from statements)
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,

                # Growth
                revenue_growth=rev_growth,
                earnings_growth=ni_growth,

                # Per share
                earnings_per_share=eps,

                # Liquidity / Leverage
                current_ratio=cur_ratio,
                debt_to_equity=de_ratio,
                debt_to_assets=da_ratio,
                interest_coverage=int_coverage,

                # Efficiency
                asset_turnover=asset_turnover,
            )

            # Enrich the latest period with .info-only valuation fields
            if is_latest:
                for field, val in info_only_fields.items():
                    if val is not None and getattr(m, field, None) is None:
                        try:
                            setattr(m, field, val)
                        except Exception:
                            pass
                # Also prefer .info margins/ROE/ROA for latest if our computed
                # values came back None (e.g., statement wasn't available yet).
                _info_fallbacks = {
                    "gross_margin":     info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "net_margin":       info.get("profitMargins"),
                    "return_on_equity": info.get("returnOnEquity"),
                    "return_on_assets": info.get("returnOnAssets"),
                    "revenue_growth":   info.get("revenueGrowth"),
                    "earnings_growth":  info.get("earningsGrowth"),
                }
                for field, val in _info_fallbacks.items():
                    if val is not None and getattr(m, field, None) is None:
                        try:
                            setattr(m, field, val)
                        except Exception:
                            pass

            results.append(m)

    except Exception as e:
        logger.warning("yfinance get_financial_metrics failed for %s: %s", ticker, e)
        return []

    if not results:
        return []

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in results])
    return results


# ── Line Items ────────────────────────────────────────────────────────────────

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    cache_key = f"{ticker}_{','.join(line_items)}_{period}_{end_date}_{limit}"
    if cached := _cache.get_line_items(cache_key):
        return [LineItem(**item) for item in cached]

    _YF_MAP = {
        # Income statement
        "revenue":                       "Total Revenue",
        "gross_profit":                  "Gross Profit",
        "operating_income":              "Operating Income",
        "net_income":                    "Net Income",
        "ebitda":                        "EBITDA",
        "eps":                           "Diluted EPS",
        "eps_diluted":                   "Diluted EPS",
        "research_and_development":      "Research And Development",
        "selling_general_and_admin":     "Selling General Administrative",
        "interest_expense":              "Interest Expense",
        # Balance sheet
        "total_assets":                  "Total Assets",
        "total_liabilities":             "Total Liabilities Net Minority Interest",
        "total_equity":                  "Stockholders Equity",
        "cash_and_equivalents":          "Cash And Cash Equivalents",
        "short_term_investments":        "Other Short Term Investments",
        "long_term_debt":                "Long Term Debt",
        "short_term_debt":               "Current Debt",
        "total_debt":                    "Total Debt",
        "goodwill_and_intangibles":      "Goodwill And Other Intangible Assets",
        "inventory":                     "Inventory",
        "accounts_receivable":           "Accounts Receivable",
        # Cash flow
        "operating_cash_flow":           "Operating Cash Flow",
        "capital_expenditure":           "Capital Expenditure",
        "free_cash_flow":                "Free Cash Flow",
        "dividends_paid":                "Common Stock Dividend Paid",
        "stock_repurchases":             "Repurchase Of Capital Stock",
        "depreciation_and_amortization": "Reconciled Depreciation",
        "shares_outstanding":            "Ordinary Shares Number",
        "shares_diluted":                "Diluted Average Shares",
    }

    try:
        stock = _get_yf_ticker(ticker)

        if period in ("ttm", "quarterly"):
            income_df   = stock.quarterly_financials
            balance_df  = stock.quarterly_balance_sheet
            cashflow_df = stock.quarterly_cashflow
        else:
            income_df   = stock.financials
            balance_df  = stock.balance_sheet
            cashflow_df = stock.cashflow

        results: list[LineItem] = []

        for item_name in line_items:
            yf_col = _YF_MAP.get(item_name)
            if not yf_col:
                logger.debug("No yfinance mapping for line item: %s", item_name)
                continue

            source_df = None
            for df in [income_df, balance_df, cashflow_df]:
                if df is not None and not df.empty and yf_col in df.index:
                    source_df = df
                    break

            if source_df is None:
                continue

            row = source_df.loc[yf_col]
            # FIX 2: filter by end_date and slice to `limit` periods *per field*,
            # not on the merged list.  This ensures every requested field gets up
            # to `limit` time periods independently instead of being truncated
            # because earlier fields consumed the shared slice budget.
            row = row[[c for c in row.index if str(c)[:10] <= end_date]]
            row = row.sort_index(ascending=False).head(limit)

            for date, value in row.items():
                if pd.isna(value):
                    continue
                results.append(
                    LineItem(
                        ticker=ticker,
                        report_period=str(date)[:10],
                        period=period,
                        currency="USD",
                        line_item=item_name,
                        value=float(value),
                    )
                )

        # Sort newest-first across all fields.
        # NOTE: no [:limit] slice here — each field already has at most `limit`
        # periods, so slicing the merged list would silently drop fields again.
        results.sort(key=lambda x: x.report_period, reverse=True)

        if results:
            _cache.set_line_items(cache_key, [item.model_dump() for item in results])

        return results

    except Exception as e:
        logger.warning("yfinance search_line_items failed for %s: %s", ticker, e)
        return []


# ── Insider Trades ────────────────────────────────────────────────────────────

def get_line_item_values(line_items: list[LineItem], field: str) -> list[float]:
    """
    Return values for a requested financial field, newest period first.

    Supports the current wide shape from search_line_items, where each record
    is one report period with financial fields as attributes, and the legacy
    long shape with line_item/value pairs.
    """
    values = []
    for item in line_items:
        value = getattr(item, field, None)
        if value is None and getattr(item, "line_item", None) == field:
            value = getattr(item, "value", None)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def get_line_item_value(
    line_items: list[LineItem],
    field: str,
    period_offset: int = 0,
    default=None,
):
    """Return one field value by recency, where period_offset=0 is newest."""
    values = get_line_item_values(line_items, field)
    if period_offset < 0 or period_offset >= len(values):
        return default
    return values[period_offset]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached]

    try:
        df = _get_yf_ticker(ticker).insider_transactions
        if df is None or df.empty:
            return []

        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        if "start_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        else:
            return []

        df = df[df["filing_date"] <= end_date]
        if start_date:
            df = df[df["filing_date"] >= start_date]

        df = df.sort_values("filing_date", ascending=False).head(limit)

        trades = []
        for _, row in df.iterrows():
            raw_shares = row.get("shares")
            txn_type = str(row.get("transaction", "")).lower()
            # yfinance always stores positive share counts; negate for sales so
            # that callers can use sign to distinguish buys (> 0) from sells (< 0).
            if raw_shares is not None and any(w in txn_type for w in ("sale", "sell")):
                raw_shares = -abs(raw_shares)
            trades.append(InsiderTrade(
                ticker=ticker,
                filing_date=row.get("filing_date"),
                transaction_date=row.get("filing_date"),
                insider_name=row.get("insider", row.get("name", "")),
                insider_title=row.get("position", row.get("title", "")),
                transaction_type=row.get("transaction", ""),
                shares=raw_shares,
                price_per_share=row.get("value"),
                total_value=None,
                shares_owned_before=None,
                shares_owned_after=None,
                sec_filing_url=None,
            ))
    except Exception as e:
        logger.warning("yfinance get_insider_trades failed for %s: %s", ticker, e)
        return []

    if not trades:
        return []

    _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    return trades


# ── Company News ──────────────────────────────────────────────────────────────

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 100,
    api_key: str = None,
) -> list[CompanyNews]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    try:
        raw_news = _get_yf_ticker(ticker).news or []
        news_list = []

        for item in raw_news:
            pub_ts = item.get("providerPublishTime") or item.get("publishedAt")
            pub_date = (
                datetime.datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d")
                if pub_ts else end_date
            )

            if pub_date > end_date:
                continue
            if start_date and pub_date < start_date:
                continue

            news_list.append(CompanyNews(
                ticker=ticker,
                date=pub_date,
                title=item.get("title", ""),
                author=item.get("publisher", ""),
                source=item.get("publisher", ""),
                url=item.get("link", ""),
                sentiment=None,
            ))

        news_list = news_list[:limit]
    except Exception as e:
        logger.warning("yfinance get_company_news failed for %s: %s", ticker, e)
        return []

    if not news_list:
        return []

    _cache.set_company_news(cache_key, [n.model_dump() for n in news_list])
    return news_list


# ── Market Cap ────────────────────────────────────────────────────────────────

def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> float | None:
    try:
        stock = _get_yf_ticker(ticker)
        info = stock.info
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        if end_date >= today:
            return info.get("marketCap")

        shares = info.get("sharesOutstanding")
        if not shares:
            return None

        # FIX 5: use a 5-day lookback window so we always capture at least one
        # trading day, and extend end by one day because yfinance's end is
        # exclusive.  The old code used start=end_date, end=end_date which
        # always produced an empty DataFrame.
        start_approx = (
            datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=5)
        ).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start=start_approx,
            end=_next_day(end_date),
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        return float(df["Close"].iloc[-1]) * shares

    except Exception as e:
        logger.warning("yfinance get_market_cap failed for %s: %s", ticker, e)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    return prices_to_df(get_prices(ticker, start_date, end_date, api_key=api_key))