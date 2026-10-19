use chrono::{DateTime, Utc};
use rt_domain::{
    AccountSnapshot, ExecutionReport, ExecutionRequest, FillStatus, FundingEvent,
    FundingSettlement, FundingState, InstrumentRules, OrderBookSnapshot, PaperState, Position,
    QuantityChange, Side,
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
    realized_pnl: f64,
    positions: BTreeMap<String, Position>,
    funding: FundingState,
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
            realized_pnl: 0.0,
            positions: BTreeMap::new(),
            funding: FundingState::default(),
        })
    }

    pub fn positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }

    pub fn position_symbols(&self) -> Vec<String> {
        self.positions.keys().cloned().collect()
    }

    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    pub fn persistent_state(&self) -> PaperState {
        PaperState {
            session_id: self.session_id.clone(),
            cash: self.cash,
            fee_paid: self.fee_paid,
            realized_pnl: self.realized_pnl,
            positions: self.positions(),
            funding: Some(self.funding.clone()),
        }
    }

    /// Restore a persisted session. Ledgers written before funding was simulated start
    /// accruing funding for their open positions from `now`.
    pub fn restore(
        config: PaperConfig,
        state: PaperState,
        now: DateTime<Utc>,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::new(state.session_id, config)?;
        if !state.cash.is_finite()
            || state.cash <= 0.0
            || !state.fee_paid.is_finite()
            || state.fee_paid < 0.0
            || !state.realized_pnl.is_finite()
        {
            return Err(EngineError::InvalidConfig(
                "invalid persisted paper state".to_owned(),
            ));
        }
        engine.cash = state.cash;
        engine.fee_paid = state.fee_paid;
        engine.realized_pnl = state.realized_pnl;
        engine.positions = state
            .positions
            .into_iter()
            .filter(|position| position.quantity.is_finite() && position.quantity.abs() > EPSILON)
            .map(|position| (position.symbol.clone(), position))
            .collect();
        engine.funding = match state.funding {
            Some(funding) if funding.funding_pnl.is_finite() => funding,
            Some(_) => {
                return Err(EngineError::InvalidConfig(
                    "invalid persisted funding state".to_owned(),
                ));
            }
            None => {
                let mut funding = FundingState::default();
                for (symbol, position) in &engine.positions {
                    funding.cursors.insert(symbol.clone(), now);
                    funding.quantity_log.insert(
                        symbol.clone(),
                        vec![QuantityChange {
                            at: now,
                            quantity: position.quantity,
                        }],
                    );
                }
                funding
            }
        };
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
            realized_pnl: self.realized_pnl,
            unrealized_pnl,
            gross_notional,
            net_notional,
            fee_paid: self.fee_paid,
        }
    }

    /// Cumulative funding received (positive) or paid (negative) by this session.
    pub fn funding_pnl(&self) -> f64 {
        self.funding.funding_pnl
    }

    /// Symbols whose funding settlements after the returned time are not yet accounted.
    pub fn funding_cursors(&self) -> Vec<(String, DateTime<Utc>)> {
        self.funding
            .cursors
            .iter()
            .map(|(symbol, cursor)| (symbol.clone(), *cursor))
            .collect()
    }

    /// Charge one exchange settlement against the quantity that was held at its timestamp.
    ///
    /// A fill exactly at the settlement time does not count, so a position opened at the
    /// 00:00 decision close does not pay that close's settlement. Settlements at or before the
    /// symbol's cursor were already accounted and are ignored.
    pub fn apply_funding(&mut self, settlement: &FundingSettlement) -> Option<FundingEvent> {
        let cursor = *self.funding.cursors.get(&settlement.symbol)?;
        if settlement.settled_at <= cursor
            || !settlement.funding_rate.is_finite()
            || !settlement.mark_price.is_finite()
            || settlement.mark_price <= 0.0
        {
            return None;
        }
        self.funding
            .cursors
            .insert(settlement.symbol.clone(), settlement.settled_at);
        let quantity = self
            .funding
            .quantity_log
            .get(&settlement.symbol)
            .and_then(|log| {
                log.iter()
                    .rev()
                    .find(|change| change.at < settlement.settled_at)
            })
            .map_or(0.0, |change| change.quantity);
        if quantity.abs() <= EPSILON {
            return None;
        }
        let amount = -quantity * settlement.mark_price * settlement.funding_rate;
        self.cash += amount;
        self.funding.funding_pnl += amount;
        if let Some(position) = self.positions.get_mut(&settlement.symbol) {
            position.funding_pnl += amount;
        }
        Some(FundingEvent {
            symbol: settlement.symbol.clone(),
            settled_at: settlement.settled_at,
            quantity,
            mark_price: settlement.mark_price,
            funding_rate: settlement.funding_rate,
            amount,
        })
    }

    /// Record that every settlement of `symbol` up to `checked_through` has been applied,
    /// and drop quantity history that can no longer price a future settlement.
    pub fn complete_funding_through(&mut self, symbol: &str, checked_through: DateTime<Utc>) {
        let Some(cursor) = self.funding.cursors.get_mut(symbol) else {
            return;
        };
        *cursor = (*cursor).max(checked_through);
        let cursor = *cursor;
        let Some(log) = self.funding.quantity_log.get_mut(symbol) else {
            return;
        };
        if let Some(last_known) = log.iter().rposition(|change| change.at <= cursor) {
            log.drain(..last_known);
        }
        let flat = log.len() == 1 && log[0].at <= cursor && log[0].quantity.abs() <= EPSILON;
        if log.is_empty() || flat {
            self.funding.quantity_log.remove(symbol);
            self.funding.cursors.remove(symbol);
        }
    }

    fn record_quantity_change(&mut self, symbol: &str, quantity: f64, at: DateTime<Utc>) {
        self.funding.cursors.entry(symbol.to_owned()).or_insert(at);
        let log = self
            .funding
            .quantity_log
            .entry(symbol.to_owned())
            .or_default();
        if let Some(last) = log.last_mut().filter(|last| last.at == at) {
            last.quantity = quantity;
        } else {
            log.push(QuantityChange { at, quantity });
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
        self.rebalance_to_notional_with_entry_anchor(
            decision_id,
            symbol,
            target_notional,
            rules,
            book,
            None,
            None,
        )
    }

    /// Open or rebalance a bootstrap position from a historical close, while
    /// preserving the current order book's executable impact.
    pub fn bootstrap_to_notional(
        &mut self,
        decision_id: &str,
        symbol: &str,
        target_notional: f64,
        close_price: f64,
        candle_closed_at: DateTime<Utc>,
        rules: &InstrumentRules,
        book: &OrderBookSnapshot,
    ) -> Result<Option<ExecutionReport>, EngineError> {
        if !close_price.is_finite() || close_price <= 0.0 {
            return Err(EngineError::InvalidConfig(
                "bootstrap close price must be finite and positive".to_owned(),
            ));
        }
        self.rebalance_to_notional_with_entry_anchor(
            decision_id,
            symbol,
            target_notional,
            rules,
            book,
            Some(close_price),
            Some(candle_closed_at),
        )
    }

    fn rebalance_to_notional_with_entry_anchor(
        &mut self,
        decision_id: &str,
        symbol: &str,
        target_notional: f64,
        rules: &InstrumentRules,
        book: &OrderBookSnapshot,
        entry_anchor: Option<f64>,
        position_opened_at: Option<DateTime<Utc>>,
    ) -> Result<Option<ExecutionReport>, EngineError> {
        let mark = book.mid_price().ok_or_else(|| {
            EngineError::InvalidConfig("order book lacks a two-sided mid".to_owned())
        })?;
        self.mark(symbol, mark, book.captured_at);
        let current_quantity = self
            .positions
            .get(symbol)
            .map_or(0.0, |position| position.quantity);
        let pricing_reference = entry_anchor.unwrap_or(mark);
        let target_quantity = target_notional / pricing_reference;
        let delta = target_quantity - current_quantity;
        let rounded_delta = round_down(delta.abs(), rules.qty_step);
        if rounded_delta <= EPSILON {
            return Ok(None);
        }
        let proposed_quantity = current_quantity
            + if delta > 0.0 {
                rounded_delta
            } else {
                -rounded_delta
            };
        let projected_gross =
            self.projected_gross_notional(symbol, proposed_quantity, pricing_reference);
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
            quantity: rounded_delta,
            requested_at: book.captured_at,
        };
        let mut report = execute_snapshot_sweep(&request, rules, book, self.config.execution)?;
        if let Some(close_price) = entry_anchor {
            rebase_bootstrap_execution(
                &mut report,
                book,
                close_price,
                self.config.execution.fee_bps,
            )?;
        }
        self.apply_execution(&mut report, position_opened_at.unwrap_or(book.captured_at));
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

    fn apply_execution(&mut self, report: &mut ExecutionReport, now: DateTime<Utc>) {
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
                entry_fees: 0.0,
                funding_pnl: 0.0,
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
            entry.entry_fees += report.fee;
        } else {
            let closed_quantity = prior_quantity.abs().min(signed_fill.abs());
            let gross_realized_pnl =
                (fill_price - entry.entry_vwap) * prior_quantity.signum() * closed_quantity;
            let entry_fee = entry.entry_fees * closed_quantity / prior_quantity.abs();
            let exit_fee = report.fee * closed_quantity / report.filled_quantity;
            let net_closed_pnl = gross_realized_pnl - entry_fee - exit_fee;
            report.closed_quantity = closed_quantity;
            report.closed_pnl = net_closed_pnl;
            entry.entry_fees = (entry.entry_fees - entry_fee).max(0.0);
            entry.realized_pnl += gross_realized_pnl;
            self.cash += gross_realized_pnl;
            self.realized_pnl += gross_realized_pnl;
            let remaining = prior_quantity + signed_fill;
            entry.quantity = remaining;
            if remaining.signum() != prior_quantity.signum() && remaining.abs() > EPSILON {
                entry.entry_vwap = fill_price;
                entry.entry_fees = (report.fee - exit_fee).max(0.0);
                entry.opened_at = now;
            }
        }
        entry.mark_price = fill_price;
        entry.updated_at = now;
        let quantity = entry.quantity;
        if quantity.abs() <= EPSILON {
            self.positions.remove(&report.symbol);
        }
        self.record_quantity_change(
            &report.symbol,
            if quantity.abs() <= EPSILON {
                0.0
            } else {
                quantity
            },
            now,
        );
    }
}

fn rebase_bootstrap_execution(
    report: &mut ExecutionReport,
    book: &OrderBookSnapshot,
    close_price: f64,
    fee_bps: f64,
) -> Result<(), EngineError> {
    let Some(raw_vwap) = report.vwap else {
        return Ok(());
    };
    let top_of_book = match report.side {
        Side::Buy => book.best_ask(),
        Side::Sell => book.best_bid(),
    }
    .ok_or_else(|| {
        EngineError::InvalidConfig("order book lacks an executable best quote".to_owned())
    })?;
    if !raw_vwap.is_finite() || !top_of_book.is_finite() || top_of_book <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "order book has an invalid executable price".to_owned(),
        ));
    }
    let entry_price = match report.side {
        Side::Buy => close_price + (raw_vwap - top_of_book),
        Side::Sell => close_price - (top_of_book - raw_vwap),
    };
    if !entry_price.is_finite() || entry_price <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "bootstrap close and order-book impact produce an invalid entry price".to_owned(),
        ));
    }
    report.vwap = Some(entry_price);
    report.notional = report.filled_quantity * entry_price;
    report.fee = report.notional * fee_bps / 10_000.0;
    report.slippage_bps = Some(match report.side {
        Side::Buy => (raw_vwap / top_of_book - 1.0) * 10_000.0,
        Side::Sell => (top_of_book / raw_vwap - 1.0) * 10_000.0,
    });
    Ok(())
}

fn round_down(value: f64, step: f64) -> f64 {
    (value / step + 1e-10).floor() * step
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use rt_domain::{FundingSettlement, InstrumentRules, OrderBookSnapshot, PriceLevel};

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

    fn rules() -> InstrumentRules {
        InstrumentRules {
            symbol: "BTCUSDT".to_owned(),
            status: "Trading".to_owned(),
            contract_type: "LinearPerpetual".to_owned(),
            qty_step: 0.001,
            min_qty: 0.001,
            min_notional_value: 5.0,
            max_market_order_qty: 150.0,
            tick_size: 0.1,
        }
    }

    fn book_at(hour: u32) -> OrderBookSnapshot {
        OrderBookSnapshot {
            captured_at: Utc
                .with_ymd_and_hms(2026, 10, 1, hour, 1, 30)
                .single()
                .expect("time"),
            ..book()
        }
    }

    fn settlement(hour: u32, rate: f64) -> FundingSettlement {
        FundingSettlement {
            symbol: "BTCUSDT".to_owned(),
            settled_at: Utc
                .with_ymd_and_hms(2026, 10, 1, hour, 0, 0)
                .single()
                .expect("time"),
            funding_rate: rate,
            mark_price: 100.0,
        }
    }

    #[test]
    fn short_receives_positive_funding_priced_at_the_mark() {
        let mut engine = engine();
        engine
            .rebalance_to_notional("d1", "BTCUSDT", -495.0, &rules(), &book_at(0))
            .expect("rebalance works")
            .expect("trade is needed");
        let quantity = engine.positions()[0].quantity;
        assert!(quantity < 0.0);
        let cash = engine.persistent_state().cash;
        let event = engine
            .apply_funding(&settlement(8, 0.0001))
            .expect("charged");
        assert!((event.amount - (-quantity * 100.0 * 0.0001)).abs() < 1e-12);
        assert!(event.amount > 0.0);
        assert!((engine.persistent_state().cash - cash - event.amount).abs() < 1e-12);
        assert!((engine.funding_pnl() - event.amount).abs() < 1e-12);
        assert!((engine.positions()[0].funding_pnl - event.amount).abs() < 1e-12);
        // Replaying the same settlement must not charge twice.
        assert!(engine.apply_funding(&settlement(8, 0.0001)).is_none());
    }

    #[test]
    fn settlement_uses_the_quantity_held_at_its_timestamp() {
        let mut engine = engine();
        // Opened after the 00:00 settlement: that settlement is not charged.
        engine
            .rebalance_to_notional("d1", "BTCUSDT", 495.0, &rules(), &book_at(0))
            .expect("rebalance works")
            .expect("trade is needed");
        let held = engine.positions()[0].quantity;
        assert!(engine.apply_funding(&settlement(0, 0.01)).is_none());
        // Closed at 16:01:30, before the 16:00 settlement was processed: it is still charged.
        engine
            .rebalance_to_notional("d2", "BTCUSDT", 0.0, &rules(), &book_at(16))
            .expect("rebalance works")
            .expect("trade is needed");
        assert!(engine.positions().is_empty());
        let event = engine
            .apply_funding(&settlement(16, 0.001))
            .expect("charged");
        assert_eq!(event.quantity, held);
        assert!(event.amount < 0.0, "longs pay positive funding");
        // A later settlement finds the position closed.
        assert!(engine.apply_funding(&settlement(23, 0.001)).is_none());
        engine.complete_funding_through(
            "BTCUSDT",
            Utc.with_ymd_and_hms(2026, 10, 2, 0, 0, 0)
                .single()
                .expect("time"),
        );
        assert!(engine.funding_cursors().is_empty());
    }

    #[test]
    fn funding_state_survives_restore_and_legacy_state_starts_now() {
        let mut engine = engine();
        engine
            .rebalance_to_notional("d1", "BTCUSDT", 495.0, &rules(), &book_at(0))
            .expect("rebalance works")
            .expect("trade is needed");
        engine
            .apply_funding(&settlement(8, 0.001))
            .expect("charged");
        let config = engine.config;
        let restored =
            PaperEngine::restore(config, engine.persistent_state(), Utc::now()).expect("restores");
        assert_eq!(restored.funding_pnl(), engine.funding_pnl());
        assert_eq!(restored.funding_cursors(), engine.funding_cursors());

        let mut legacy = engine.persistent_state();
        legacy.funding = None;
        let now = Utc
            .with_ymd_and_hms(2026, 10, 1, 12, 0, 0)
            .single()
            .expect("time");
        let mut restored = PaperEngine::restore(config, legacy, now).expect("restores");
        assert_eq!(
            restored.funding_cursors(),
            vec![("BTCUSDT".to_owned(), now)]
        );
        assert!(restored.apply_funding(&settlement(8, 0.001)).is_none());
        assert!(restored.apply_funding(&settlement(16, 0.001)).is_some());
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
                    status: "Trading".to_owned(),
                    contract_type: "LinearPerpetual".to_owned(),
                    qty_step: 0.001,
                    min_qty: 0.001,
                    min_notional_value: 5.0,
                    max_market_order_qty: 150.0,
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

    #[test]
    fn closing_a_position_realizes_pnl_into_cash_and_equity() {
        let mut engine = engine();
        let rules = InstrumentRules {
            symbol: "BTCUSDT".to_owned(),
            status: "Trading".to_owned(),
            contract_type: "LinearPerpetual".to_owned(),
            qty_step: 0.001,
            min_qty: 0.001,
            min_notional_value: 5.0,
            max_market_order_qty: 150.0,
            tick_size: 0.1,
        };
        let open = engine
            .rebalance_to_notional("open", "BTCUSDT", 500.0, &rules, &book())
            .expect("open works")
            .expect("open trade is needed");
        let mut closing_book = book();
        closing_book.bids[0].price = 109.0;
        closing_book.asks[0].price = 111.0;
        let close = engine
            .rebalance_to_notional("close", "BTCUSDT", 0.0, &rules, &closing_book)
            .expect("close works")
            .expect("close trade is needed");

        let snapshot = engine.snapshot(Utc::now());
        assert!(engine.positions().is_empty());
        assert!((snapshot.realized_pnl - 40.0).abs() < 1e-9);
        assert_eq!(close.closed_quantity, 5.0);
        assert!((close.closed_pnl - (40.0 - open.fee - close.fee)).abs() < 1e-9);
        assert!((snapshot.equity - snapshot.cash).abs() < 1e-9);
        assert!(snapshot.equity > 1_000.0);
    }

    #[test]
    fn bootstrap_anchors_entry_to_close_and_uses_top_of_book_impact() {
        let mut engine = engine();
        let mut impact_book = book();
        impact_book.asks = vec![
            PriceLevel {
                price: 101.0,
                quantity: 1.0,
            },
            PriceLevel {
                price: 103.0,
                quantity: 100.0,
            },
        ];
        let candle_closed_at = Utc
            .with_ymd_and_hms(2026, 8, 13, 0, 0, 0)
            .single()
            .expect("valid close timestamp");
        let report = engine
            .bootstrap_to_notional(
                "bootstrap",
                "BTCUSDT",
                200.0,
                100.0,
                candle_closed_at,
                &InstrumentRules {
                    symbol: "BTCUSDT".to_owned(),
                    status: "Trading".to_owned(),
                    contract_type: "LinearPerpetual".to_owned(),
                    qty_step: 0.001,
                    min_qty: 0.001,
                    min_notional_value: 5.0,
                    max_market_order_qty: 150.0,
                    tick_size: 0.1,
                },
                &impact_book,
            )
            .expect("bootstrap works")
            .expect("trade is needed");

        let expected_entry = 100.0 + (102.0 - 101.0);
        assert!((report.vwap.expect("fill price") - expected_entry).abs() < 1e-9);
        assert!(
            (report.slippage_bps.expect("impact") - (102.0 / 101.0 - 1.0) * 10_000.0).abs() < 1e-9
        );
        assert!((engine.positions()[0].entry_vwap - expected_entry).abs() < 1e-9);
        assert_eq!(engine.positions()[0].opened_at, candle_closed_at);
    }

    #[test]
    fn bootstrap_short_anchors_entry_to_close_and_uses_bid_impact() {
        let mut engine = engine();
        let mut impact_book = book();
        impact_book.bids = vec![
            PriceLevel {
                price: 99.0,
                quantity: 1.0,
            },
            PriceLevel {
                price: 97.0,
                quantity: 100.0,
            },
        ];
        let report = engine
            .bootstrap_to_notional(
                "bootstrap",
                "BTCUSDT",
                -200.0,
                100.0,
                impact_book.captured_at,
                &InstrumentRules {
                    symbol: "BTCUSDT".to_owned(),
                    status: "Trading".to_owned(),
                    contract_type: "LinearPerpetual".to_owned(),
                    qty_step: 0.001,
                    min_qty: 0.001,
                    min_notional_value: 5.0,
                    max_market_order_qty: 150.0,
                    tick_size: 0.1,
                },
                &impact_book,
            )
            .expect("bootstrap works")
            .expect("trade is needed");

        let expected_entry = 100.0 - (99.0 - 98.0);
        assert!((report.vwap.expect("fill price") - expected_entry).abs() < 1e-9);
        assert!(
            (report.slippage_bps.expect("impact") - (99.0 / 98.0 - 1.0) * 10_000.0).abs() < 1e-9
        );
        assert!((engine.positions()[0].entry_vwap - expected_entry).abs() < 1e-9);
    }
}
