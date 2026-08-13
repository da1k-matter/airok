use chrono::{DateTime, Utc};
use rt_domain::{
    AccountSnapshot, ExecutionReport, ExecutionRequest, FillStatus, InstrumentRules,
    OrderBookSnapshot, PaperState, Position, Side,
};
use rt_execution::{SnapshotExecutionConfig, execute_snapshot_sweep};
use std::collections::BTreeMap;

const EPSILON: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskLimits {
    pub gross_leverage: f64,
    pub max_gross_leverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaperConfig {
    pub initial_equity_usd: f64,
    pub execution: SnapshotExecutionConfig,
    pub risk: RiskLimits,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("invalid paper configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "target would exceed max gross leverage: requested {requested:.4}, maximum {maximum:.4}"
    )]
    GrossLeverageExceeded { requested: f64, maximum: f64 },
    #[error(transparent)]
    Execution(#[from] rt_execution::ExecutionError),
}

#[derive(Debug)]
pub struct PaperEngine {
    session_id: String,
    config: PaperConfig,
    cash: f64,
    fee_paid: f64,
    positions: BTreeMap<String, Position>,
}

impl PaperEngine {
    pub fn new(session_id: String, config: PaperConfig) -> Result<Self, EngineError> {
        if !config.initial_equity_usd.is_finite() || config.initial_equity_usd <= 0.0 {
            return Err(EngineError::InvalidConfig(
                "initial_equity_usd must be positive".to_owned(),
            ));
        }
        if !config.risk.gross_leverage.is_finite()
            || !config.risk.max_gross_leverage.is_finite()
            || config.risk.gross_leverage <= 0.0
            || config.risk.max_gross_leverage <= 0.0
            || config.risk.gross_leverage > config.risk.max_gross_leverage
        {
            return Err(EngineError::InvalidConfig(
                "invalid gross leverage limits".to_owned(),
            ));
        }
        Ok(Self {
            session_id,
            config,
            cash: config.initial_equity_usd,
            fee_paid: 0.0,
            positions: BTreeMap::new(),
        })
    }

    pub fn positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }

    pub fn persistent_state(&self) -> PaperState {
        PaperState {
            session_id: self.session_id.clone(),
            cash: self.cash,
            fee_paid: self.fee_paid,
            positions: self.positions(),
        }
    }

    pub fn restore(config: PaperConfig, state: PaperState) -> Result<Self, EngineError> {
        let mut engine = Self::new(state.session_id, config)?;
        if !state.cash.is_finite()
            || state.cash <= 0.0
            || !state.fee_paid.is_finite()
            || state.fee_paid < 0.0
        {
            return Err(EngineError::InvalidConfig(
                "invalid persisted paper state".to_owned(),
            ));
        }
        engine.cash = state.cash;
        engine.fee_paid = state.fee_paid;
        engine.positions = state
            .positions
            .into_iter()
            .filter(|position| position.quantity.is_finite() && position.quantity.abs() > EPSILON)
            .map(|position| (position.symbol.clone(), position))
            .collect();
        Ok(engine)
    }

    pub fn mark(&mut self, symbol: &str, price: f64, now: DateTime<Utc>) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.mark_price = price;
            position.updated_at = now;
        }
    }

    pub fn snapshot(&self, now: DateTime<Utc>) -> AccountSnapshot {
        let unrealized_pnl = self
            .positions
            .values()
            .map(Position::unrealized_pnl)
            .sum::<f64>();
        let realized_pnl = self
            .positions
            .values()
            .map(|position| position.realized_pnl)
            .sum::<f64>();
        let gross_notional = self.positions.values().map(Position::notional).sum::<f64>();
        let net_notional = self
            .positions
            .values()
            .map(|position| position.quantity * position.mark_price)
            .sum::<f64>();
        AccountSnapshot {
            session_id: self.session_id.clone(),
            captured_at: now,
            cash: self.cash,
            equity: self.cash + unrealized_pnl,
            realized_pnl,
            unrealized_pnl,
            gross_notional,
            net_notional,
            fee_paid: self.fee_paid,
        }
    }

    pub fn rebalance_to_notional(
        &mut self,
        decision_id: &str,
        symbol: &str,
        target_notional: f64,
        rules: &InstrumentRules,
        book: &OrderBookSnapshot,
    ) -> Result<Option<ExecutionReport>, EngineError> {
        let mark = book.mid_price().ok_or_else(|| {
            EngineError::InvalidConfig("order book lacks a two-sided mid".to_owned())
        })?;
        self.mark(symbol, mark, book.captured_at);
        let current_quantity = self
            .positions
            .get(symbol)
            .map_or(0.0, |position| position.quantity);
        let target_quantity = target_notional / mark;
        let delta = target_quantity - current_quantity;
        if delta.abs() < rules.min_qty {
            return Ok(None);
        }
        let projected_gross = self.projected_gross_notional(symbol, target_quantity, mark);
        let maximum = self.snapshot(book.captured_at).equity * self.config.risk.max_gross_leverage;
        if projected_gross > maximum + EPSILON {
            return Err(EngineError::GrossLeverageExceeded {
                requested: projected_gross,
                maximum,
            });
        }
        let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
        let request = ExecutionRequest {
            decision_id: decision_id.to_owned(),
            symbol: symbol.to_owned(),
            side,
            quantity: delta.abs(),
            requested_at: book.captured_at,
        };
        let report = execute_snapshot_sweep(&request, rules, book, self.config.execution)?;
        self.apply_execution(&report, book.captured_at);
        Ok(Some(report))
    }

    fn projected_gross_notional(&self, symbol: &str, target_quantity: f64, mark: f64) -> f64 {
        self.positions
            .iter()
            .map(|(position_symbol, position)| {
                if position_symbol == symbol {
                    target_quantity.abs() * mark
                } else {
                    position.notional()
                }
            })
            .sum::<f64>()
            + if self.positions.contains_key(symbol) {
                0.0
            } else {
                target_quantity.abs() * mark
            }
    }

    fn apply_execution(&mut self, report: &ExecutionReport, now: DateTime<Utc>) {
        if matches!(report.status, FillStatus::Rejected) || report.filled_quantity <= EPSILON {
            return;
        }
        let fill_price = report.vwap.expect("filled reports must have a VWAP");
        let signed_fill = report.side.sign() * report.filled_quantity;
        self.cash -= report.fee;
        self.fee_paid += report.fee;
        let entry = self
            .positions
            .entry(report.symbol.clone())
            .or_insert_with(|| Position {
                symbol: report.symbol.clone(),
                quantity: 0.0,
                entry_vwap: fill_price,
                mark_price: fill_price,
                realized_pnl: 0.0,
                opened_at: now,
                updated_at: now,
            });
        let prior_quantity = entry.quantity;
        if prior_quantity.abs() <= EPSILON || prior_quantity.signum() == signed_fill.signum() {
            let new_quantity = prior_quantity + signed_fill;
            let prior_notional = prior_quantity.abs() * entry.entry_vwap;
            entry.entry_vwap =
                (prior_notional + signed_fill.abs() * fill_price) / new_quantity.abs();
            entry.quantity = new_quantity;
        } else {
            let closed_quantity = prior_quantity.abs().min(signed_fill.abs());
            entry.realized_pnl +=
                (fill_price - entry.entry_vwap) * prior_quantity.signum() * closed_quantity;
            let remaining = prior_quantity + signed_fill;
            entry.quantity = remaining;
            if remaining.signum() != prior_quantity.signum() && remaining.abs() > EPSILON {
                entry.entry_vwap = fill_price;
                entry.opened_at = now;
            }
        }
        entry.mark_price = fill_price;
        entry.updated_at = now;
        if entry.quantity.abs() <= EPSILON {
            self.positions.remove(&report.symbol);
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rt_domain::{InstrumentRules, OrderBookSnapshot, PriceLevel};

    use super::{PaperConfig, PaperEngine, RiskLimits};
    use rt_execution::SnapshotExecutionConfig;

    fn engine() -> PaperEngine {
        PaperEngine::new(
            "test".to_owned(),
            PaperConfig {
                initial_equity_usd: 1_000.0,
                execution: SnapshotExecutionConfig {
                    fee_bps: 10.0,
                    reject_partial: true,
                },
                risk: RiskLimits {
                    gross_leverage: 1.0,
                    max_gross_leverage: 1.0,
                },
            },
        )
        .expect("engine builds")
    }

    fn book() -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: "BTCUSDT".to_owned(),
            captured_at: Utc::now(),
            update_id: 1,
            bids: vec![PriceLevel {
                price: 99.0,
                quantity: 100.0,
            }],
            asks: vec![PriceLevel {
                price: 101.0,
                quantity: 100.0,
            }],
        }
    }

    #[test]
    fn rebalance_uses_paper_equity_and_charges_fee() {
        let mut engine = engine();
        let report = engine
            .rebalance_to_notional(
                "d1",
                "BTCUSDT",
                500.0,
                &InstrumentRules {
                    symbol: "BTCUSDT".to_owned(),
                    qty_step: 0.001,
                    min_qty: 0.001,
                    tick_size: 0.1,
                },
                &book(),
            )
            .expect("rebalance works")
            .expect("trade is needed");
        assert!(report.fee > 0.0);
        assert_eq!(engine.positions().len(), 1);
        assert!(engine.snapshot(Utc::now()).equity < 1_000.0);
    }
}
