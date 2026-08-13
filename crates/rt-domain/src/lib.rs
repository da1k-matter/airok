use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

pub type Usd = f64;
pub type Quantity = f64;
pub type Price = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn sign(self) -> f64 {
        match self {
            Self::Buy => 1.0,
            Self::Sell => -1.0,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

impl Display for Side {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => formatter.write_str("buy"),
            Self::Sell => formatter.write_str("sell"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candle {
    pub symbol: String,
    pub opened_at: DateTime<Utc>,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: f64,
    pub confirmed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: Price,
    pub quantity: Quantity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: String,
    pub captured_at: DateTime<Utc>,
    pub update_id: u64,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
}

impl OrderBookSnapshot {
    pub fn best_bid(&self) -> Option<Price> {
        self.bids
            .iter()
            .map(|level| level.price)
            .max_by(f64::total_cmp)
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks
            .iter()
            .map(|level| level.price)
            .min_by(f64::total_cmp)
    }

    pub fn mid_price(&self) -> Option<Price> {
        Some((self.best_bid()? + self.best_ask()?) / 2.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentRules {
    pub symbol: String,
    pub qty_step: Quantity,
    pub min_qty: Quantity,
    pub tick_size: Price,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub decision_id: String,
    pub symbol: String,
    pub side: Side,
    pub quantity: Quantity,
    pub requested_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillStatus {
    Filled,
    Partial,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsumedLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub notional: Usd,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub execution_id: String,
    pub decision_id: String,
    pub symbol: String,
    pub side: Side,
    pub status: FillStatus,
    pub requested_quantity: Quantity,
    pub filled_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub benchmark_mid: Option<Price>,
    pub vwap: Option<Price>,
    pub notional: Usd,
    pub fee: Usd,
    pub slippage_bps: Option<f64>,
    pub consumed_levels: Vec<ConsumedLevel>,
    pub executed_at: DateTime<Utc>,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: Quantity,
    pub entry_vwap: Price,
    pub mark_price: Price,
    pub realized_pnl: Usd,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Position {
    pub fn side(&self) -> Side {
        if self.quantity >= 0.0 {
            Side::Buy
        } else {
            Side::Sell
        }
    }

    pub fn notional(&self) -> Usd {
        self.quantity.abs() * self.mark_price
    }

    pub fn unrealized_pnl(&self) -> Usd {
        (self.mark_price - self.entry_vwap) * self.quantity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub session_id: String,
    pub captured_at: DateTime<Utc>,
    pub cash: Usd,
    pub equity: Usd,
    pub realized_pnl: Usd,
    pub unrealized_pnl: Usd,
    pub gross_notional: Usd,
    pub net_notional: Usd,
    pub fee_paid: Usd,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperState {
    pub session_id: String,
    pub cash: Usd,
    pub fee_paid: Usd,
    pub positions: Vec<Position>,
}

#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("{field} must be finite and positive")]
    NonPositive { field: &'static str },
}

pub fn require_positive(value: f64, field: &'static str) -> Result<(), DomainError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DomainError::NonPositive { field })
    }
}
