use chrono::Utc;
use rt_domain::{
    ConsumedLevel, ExecutionReport, ExecutionRequest, FillStatus, InstrumentRules,
    OrderBookSnapshot, PriceLevel, Side, require_positive,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotExecutionConfig {
    pub fee_bps: f64,
    pub reject_partial: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("invalid execution config: {0}")]
    InvalidConfig(String),
}

pub fn execute_snapshot_sweep(
    request: &ExecutionRequest,
    rules: &InstrumentRules,
    book: &OrderBookSnapshot,
    config: SnapshotExecutionConfig,
) -> Result<ExecutionReport, ExecutionError> {
    require_positive(request.quantity, "request.quantity")
        .map_err(|error| ExecutionError::InvalidConfig(error.to_string()))?;
    require_positive(rules.qty_step, "rules.qty_step")
        .map_err(|error| ExecutionError::InvalidConfig(error.to_string()))?;
    if !config.fee_bps.is_finite() || config.fee_bps < 0.0 {
        return Err(ExecutionError::InvalidConfig(
            "fee_bps must be finite and non-negative".to_owned(),
        ));
    }
    if request.symbol != book.symbol || request.symbol != rules.symbol {
        return Ok(rejected(request, book, "instrument mismatch"));
    }
    if !rules.is_tradable_linear_perpetual() {
        return Ok(rejected(
            request,
            book,
            "instrument is not a tradable Bybit linear perpetual",
        ));
    }

    let requested_quantity = round_down(request.quantity, rules.qty_step);
    if requested_quantity < rules.min_qty {
        return Ok(rejected(
            request,
            book,
            "quantity is below the Bybit minimum",
        ));
    }
    if requested_quantity > rules.max_market_order_qty {
        return Ok(rejected(
            request,
            book,
            "quantity exceeds the Bybit market-order maximum",
        ));
    }

    let mut levels = match request.side {
        Side::Buy => sorted_levels(&book.asks, false),
        Side::Sell => sorted_levels(&book.bids, true),
    };
    let reference_price = levels
        .first()
        .map(|level| level.price)
        .filter(|price| price.is_finite() && *price > 0.0);
    if reference_price.is_none_or(|price| requested_quantity * price < rules.min_notional_value) {
        return Ok(rejected(
            request,
            book,
            "rounded order notional is below the Bybit minimum",
        ));
    }
    let mut remaining = requested_quantity;
    let mut filled_quantity = 0.0;
    let mut notional = 0.0;
    let mut consumed_levels = Vec::new();

    for level in levels.drain(..) {
        if remaining <= 0.0 {
            break;
        }
        if !level.price.is_finite()
            || !level.quantity.is_finite()
            || level.price <= 0.0
            || level.quantity <= 0.0
        {
            continue;
        }
        let available = round_down(level.quantity, rules.qty_step);
        let quantity = round_down(remaining.min(available), rules.qty_step);
        if quantity <= 0.0 {
            continue;
        }
        let level_notional = quantity * level.price;
        filled_quantity += quantity;
        remaining = (requested_quantity - filled_quantity).max(0.0);
        notional += level_notional;
        consumed_levels.push(ConsumedLevel {
            price: level.price,
            quantity,
            notional: level_notional,
        });
    }

    if filled_quantity <= 0.0 || (config.reject_partial && remaining > 0.0) {
        return Ok(rejected(
            request,
            book,
            if filled_quantity <= 0.0 {
                "no executable liquidity in snapshot"
            } else {
                "insufficient snapshot liquidity; partial fills are disabled"
            },
        ));
    }
    if notional < rules.min_notional_value {
        return Ok(rejected(
            request,
            book,
            "filled order notional is below the Bybit minimum",
        ));
    }

    let vwap = notional / filled_quantity;
    let benchmark_mid = book.mid_price();
    let slippage_bps = benchmark_mid.map(|mid| match request.side {
        Side::Buy => (vwap / mid - 1.0) * 10_000.0,
        Side::Sell => (1.0 - vwap / mid) * 10_000.0,
    });
    let fee = notional * config.fee_bps / 10_000.0;
    Ok(ExecutionReport {
        execution_id: format!("{}:{}", request.decision_id, request.symbol),
        decision_id: request.decision_id.clone(),
        symbol: request.symbol.clone(),
        side: request.side,
        status: if remaining > 0.0 {
            FillStatus::Partial
        } else {
            FillStatus::Filled
        },
        requested_quantity,
        filled_quantity,
        remaining_quantity: remaining,
        benchmark_mid,
        vwap: Some(vwap),
        notional,
        fee,
        slippage_bps,
        consumed_levels,
        executed_at: Utc::now(),
        rejection_reason: None,
    })
}

fn rejected(request: &ExecutionRequest, book: &OrderBookSnapshot, reason: &str) -> ExecutionReport {
    ExecutionReport {
        execution_id: format!("{}:{}", request.decision_id, request.symbol),
        decision_id: request.decision_id.clone(),
        symbol: request.symbol.clone(),
        side: request.side,
        status: FillStatus::Rejected,
        requested_quantity: request.quantity,
        filled_quantity: 0.0,
        remaining_quantity: request.quantity,
        benchmark_mid: book.mid_price(),
        vwap: None,
        notional: 0.0,
        fee: 0.0,
        slippage_bps: None,
        consumed_levels: Vec::new(),
        executed_at: Utc::now(),
        rejection_reason: Some(reason.to_owned()),
    }
}

fn sorted_levels(levels: &[PriceLevel], descending: bool) -> Vec<PriceLevel> {
    let mut sorted = levels.to_vec();
    sorted.sort_by(|left, right| {
        if descending {
            right.price.total_cmp(&left.price)
        } else {
            left.price.total_cmp(&right.price)
        }
    });
    sorted
}

fn round_down(value: f64, step: f64) -> f64 {
    (value / step + 1e-10).floor() * step
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rt_domain::{
        ExecutionRequest, FillStatus, InstrumentRules, OrderBookSnapshot, PriceLevel, Side,
    };

    use super::{SnapshotExecutionConfig, execute_snapshot_sweep};

    fn book() -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: "BTCUSDT".to_owned(),
            captured_at: Utc::now(),
            update_id: 1,
            bids: vec![PriceLevel {
                price: 99.0,
                quantity: 2.0,
            }],
            asks: vec![
                PriceLevel {
                    price: 101.0,
                    quantity: 0.5,
                },
                PriceLevel {
                    price: 102.0,
                    quantity: 2.0,
                },
            ],
        }
    }

    #[test]
    fn sweeps_asks_and_records_vwap() {
        let report = execute_snapshot_sweep(
            &ExecutionRequest {
                decision_id: "d1".to_owned(),
                symbol: "BTCUSDT".to_owned(),
                side: Side::Buy,
                quantity: 1.5,
                requested_at: Utc::now(),
            },
            &InstrumentRules {
                symbol: "BTCUSDT".to_owned(),
                status: "Trading".to_owned(),
                contract_type: "LinearPerpetual".to_owned(),
                qty_step: 0.1,
                min_qty: 0.1,
                min_notional_value: 5.0,
                max_market_order_qty: 10.0,
                tick_size: 0.1,
            },
            &book(),
            SnapshotExecutionConfig {
                fee_bps: 10.0,
                reject_partial: true,
            },
        )
        .expect("execution succeeds");
        assert_eq!(report.filled_quantity, 1.5);
        assert_eq!(report.consumed_levels.len(), 2);
        assert!((report.vwap.expect("vwap") - (0.5 * 101.0 + 102.0) / 1.5).abs() < 1e-10);
        assert!(report.slippage_bps.expect("slippage") > 0.0);
    }

    #[test]
    fn rejects_a_quantity_that_rounds_below_minimum_notional() {
        let report = execute_snapshot_sweep(
            &ExecutionRequest {
                decision_id: "d2".to_owned(),
                symbol: "BTCUSDT".to_owned(),
                side: Side::Buy,
                quantity: 0.049,
                requested_at: Utc::now(),
            },
            &InstrumentRules {
                symbol: "BTCUSDT".to_owned(),
                status: "Trading".to_owned(),
                contract_type: "LinearPerpetual".to_owned(),
                qty_step: 0.01,
                min_qty: 0.01,
                min_notional_value: 5.0,
                max_market_order_qty: 10.0,
                tick_size: 0.1,
            },
            &book(),
            SnapshotExecutionConfig {
                fee_bps: 10.0,
                reject_partial: true,
            },
        )
        .expect("execution succeeds");
        assert_eq!(report.status, FillStatus::Rejected);
        assert_eq!(
            report.rejection_reason.as_deref(),
            Some("rounded order notional is below the Bybit minimum")
        );
    }

    #[test]
    fn rejects_non_trading_and_oversized_market_orders() {
        let request = ExecutionRequest {
            decision_id: "d3".to_owned(),
            symbol: "BTCUSDT".to_owned(),
            side: Side::Buy,
            quantity: 1.0,
            requested_at: Utc::now(),
        };
        let mut rules = InstrumentRules {
            symbol: "BTCUSDT".to_owned(),
            status: "PreLaunch".to_owned(),
            contract_type: "LinearPerpetual".to_owned(),
            qty_step: 0.01,
            min_qty: 0.01,
            min_notional_value: 5.0,
            max_market_order_qty: 0.5,
            tick_size: 0.1,
        };
        let config = SnapshotExecutionConfig {
            fee_bps: 10.0,
            reject_partial: true,
        };
        let unavailable =
            execute_snapshot_sweep(&request, &rules, &book(), config).expect("execution succeeds");
        assert_eq!(unavailable.status, FillStatus::Rejected);
        assert_eq!(
            unavailable.rejection_reason.as_deref(),
            Some("instrument is not a tradable Bybit linear perpetual")
        );

        rules.status = "Trading".to_owned();
        let oversized =
            execute_snapshot_sweep(&request, &rules, &book(), config).expect("execution succeeds");
        assert_eq!(oversized.status, FillStatus::Rejected);
        assert_eq!(
            oversized.rejection_reason.as_deref(),
            Some("quantity exceeds the Bybit market-order maximum")
        );
    }
}
