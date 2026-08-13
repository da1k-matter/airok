from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

BYBIT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info"


@dataclass(frozen=True)
class BybitInstrumentRule:
    symbol: str
    status: str
    contract_type: str
    qty_step: float
    min_order_qty: float
    min_notional_value: float
    max_market_order_qty: float
    tick_size: float

    @property
    def tradable_linear_perpetual(self) -> bool:
        return self.status == "Trading" and self.contract_type == "LinearPerpetual"


def _rule(payload: dict[str, Any]) -> BybitInstrumentRule:
    lot = payload["lotSizeFilter"]
    price = payload["priceFilter"]
    return BybitInstrumentRule(
        symbol=str(payload["symbol"]),
        status=str(payload["status"]),
        contract_type=str(payload["contractType"]),
        qty_step=float(lot["qtyStep"]),
        min_order_qty=float(lot["minOrderQty"]),
        min_notional_value=float(lot["minNotionalValue"]),
        max_market_order_qty=float(lot["maxMktOrderQty"]),
        tick_size=float(price["tickSize"]),
    )


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
