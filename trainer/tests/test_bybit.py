from bybit import _rule


def test_bybit_rule_parses_linear_perpetual_constraints():
    rule = _rule(
        {
            "symbol": "BTCUSDT",
            "status": "Trading",
            "contractType": "LinearPerpetual",
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
    assert rule.min_notional_value == 5.0
    assert rule.qty_step == 0.001
