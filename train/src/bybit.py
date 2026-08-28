from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

BYBIT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info"


@dataclass(frozen=True)
class BybitInstrumentRule:
    symbol: str
    status: str
    contract_type: str
    quote_coin: str
    settle_coin: str
    copy_trading: str
    qty_step: float
    min_order_qty: float
    min_notional_value: float
    max_market_order_qty: float
    tick_size: float

    @property
    def tradable_linear_perpetual(self) -> bool:
        return self.status == "Trading" and self.contract_type == "LinearPerpetual"

    @property
    def copy_tradable_usdt_perpetual(self) -> bool:
        return (
            self.tradable_linear_perpetual
            and self.quote_coin == "USDT"
            and self.settle_coin == "USDT"
            and self.copy_trading == "both"
        )


def _rule(payload: dict[str, Any]) -> BybitInstrumentRule:
    lot = payload["lotSizeFilter"]
    price = payload["priceFilter"]
    return BybitInstrumentRule(
        symbol=str(payload["symbol"]),
        status=str(payload["status"]),
        contract_type=str(payload["contractType"]),
        quote_coin=str(payload["quoteCoin"]),
        settle_coin=str(payload["settleCoin"]),
        copy_trading=str(payload["copyTrading"]),
        qty_step=float(lot["qtyStep"]),
        min_order_qty=float(lot["minOrderQty"]),
        min_notional_value=float(lot["minNotionalValue"]),
        max_market_order_qty=float(lot["maxMktOrderQty"]),
        tick_size=float(price["tickSize"]),
    )


@lru_cache(maxsize=1)
def fetch_linear_perpetual_rules() -> tuple[dict[str, BybitInstrumentRule], dict[str, Any]]:
    """Fetch the complete current Bybit linear-instrument rule set once per run."""
    query: dict[str, str] = {"category": "linear", "limit": "1000"}
    rules: dict[str, BybitInstrumentRule] = {}
    pages = 0
    while True:
        with urlopen(f"{BYBIT_INSTRUMENTS_URL}?{urlencode(query)}", timeout=30) as response:
            payload = json.load(response)
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"Bybit instruments request failed: {payload.get('retMsg', 'unknown error')}")
        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise RuntimeError("Bybit instruments response has no result list")
        for item in result["list"]:
            rule = _rule(item)
            rules[rule.symbol] = rule
        pages += 1
        cursor = str(result.get("nextPageCursor") or "")
        if not cursor:
            break
        query["cursor"] = cursor

    snapshot = {
        "source": BYBIT_INSTRUMENTS_URL,
        "category": "linear",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "instrument_count": len(rules),
        "rules": [asdict(rule) for rule in sorted(rules.values(), key=lambda item: item.symbol)],
    }
    return rules, snapshot


def copy_trading_base_symbols(rules: dict[str, BybitInstrumentRule]) -> frozenset[str]:
    """Select current Copy Trading USDT perpetuals and map them to data-file symbols."""
    symbols = frozenset(
        rule.symbol.removesuffix("USDT")
        for rule in rules.values()
        if rule.copy_tradable_usdt_perpetual
    )
    if "BTC" not in symbols:
        raise RuntimeError("Bybit Copy Trading response does not contain BTCUSDT")
    return symbols


def fetch_copy_trading_base_symbols() -> frozenset[str]:
    rules, _ = fetch_linear_perpetual_rules()
    return copy_trading_base_symbols(rules)
