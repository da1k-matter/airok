from bybit import _rule, copy_trading_base_symbols


def test_bybit_rule_parses_linear_perpetual_constraints():
    rule = _rule(
        {
            "symbol": "BTCUSDT",
            "status": "Trading",
            "contractType": "LinearPerpetual",
            "quoteCoin": "USDT",
            "settleCoin": "USDT",
            "copyTrading": "both",
            "lotSizeFilter": {
                "qtyStep": "0.001",
                "minOrderQty": "0.001",
                "minNotionalValue": "5",
                "maxMktOrderQty": "150",
            },
            "priceFilter": {"tickSize": "0.10"},
        }
    )
    assert rule.tradable_linear_perpetual
    assert rule.copy_tradable_usdt_perpetual
    assert rule.min_notional_value == 5.0
    assert rule.qty_step == 0.001


def test_copy_trading_filter_excludes_non_copy_and_non_usdt_contracts():
    copy_rule = _rule(
        {
            "symbol": "BTCUSDT", "status": "Trading", "contractType": "LinearPerpetual",
            "quoteCoin": "USDT", "settleCoin": "USDT", "copyTrading": "both",
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "5", "maxMktOrderQty": "150"},
            "priceFilter": {"tickSize": "0.10"},
        }
    )
    unsupported_rule = _rule(
        {
            "symbol": "ETHUSDT", "status": "Trading", "contractType": "LinearPerpetual",
            "quoteCoin": "USDT", "settleCoin": "USDT", "copyTrading": "none",
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "5", "maxMktOrderQty": "150"},
            "priceFilter": {"tickSize": "0.10"},
        }
    )
    non_usdt_rule = _rule(
        {
            "symbol": "BTCUSDC", "status": "Trading", "contractType": "LinearPerpetual",
            "quoteCoin": "USDC", "settleCoin": "USDC", "copyTrading": "both",
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "minNotionalValue": "5", "maxMktOrderQty": "150"},
            "priceFilter": {"tickSize": "0.10"},
        }
    )

    assert copy_trading_base_symbols({rule.symbol: rule for rule in (copy_rule, unsupported_rule, non_usdt_rule)}) == frozenset({"BTC"})
