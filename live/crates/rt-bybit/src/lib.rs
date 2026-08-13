use anyhow::{Context, Result, bail};
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use rt_domain::{Candle, OrderBookSnapshot, PriceLevel};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

const REST_URL: &str = "https://api.bybit.com";
const WS_URL: &str = "wss://stream.bybit.com/v5/public/linear";

pub fn bybit_linear_symbol(base_symbol: &str) -> String {
    format!("{base_symbol}USDT")
}

pub fn base_symbol(bybit_symbol: &str) -> Result<String> {
    bybit_symbol
        .strip_suffix("USDT")
        .filter(|symbol| !symbol.is_empty())
        .map(ToOwned::to_owned)
        .with_context(|| format!("expected a USDT linear symbol, got {bybit_symbol}"))
}

#[derive(Clone)]
pub struct BybitPublicClient {
    http: Client,
    rest_url: String,
    ws_url: String,
}

impl Default for BybitPublicClient {
    fn default() -> Self {
        Self::new(REST_URL, WS_URL)
    }
}

impl BybitPublicClient {
    pub fn new(rest_url: impl Into<String>, ws_url: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            rest_url: rest_url.into(),
            ws_url: ws_url.into(),
        }
    }

    pub async fn orderbook(&self, symbol: &str, depth: u16) -> Result<OrderBookSnapshot> {
        let url = format!("{}/v5/market/orderbook", self.rest_url);
        let payload: ApiEnvelope<OrderBookResult> = self
            .http
            .get(url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &depth.to_string()),
            ])
            .send()
            .await
            .context("request Bybit order book")?
            .error_for_status()
            .context("Bybit order book HTTP status")?
            .json()
            .await
            .context("decode Bybit order book")?;
        ensure_success(&payload)?;
        let result = payload.result.context("Bybit order book missing result")?;
        Ok(OrderBookSnapshot {
            symbol: result.symbol,
            captured_at: Utc::now(),
            update_id: result.update_id,
            bids: parse_levels(result.bids)?,
            asks: parse_levels(result.asks)?,
        })
    }

    /// Retrieve the immutable order-size and tick constraints used by the paper executor.
    pub async fn instrument_rules(&self, symbol: &str) -> Result<rt_domain::InstrumentRules> {
        let url = format!("{}/v5/market/instruments-info", self.rest_url);
        let payload: ApiEnvelope<InstrumentsResult> = self
            .http
            .get(url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await
            .context("request Bybit instrument rules")?
            .error_for_status()
            .context("Bybit instrument rules HTTP status")?
            .json()
            .await
            .context("decode Bybit instrument rules")?;
        ensure_success(&payload)?;
        let result = payload
            .result
            .context("Bybit instrument rules response missing result")?;
        let instrument = result
            .list
            .into_iter()
            .find(|instrument| instrument.symbol == symbol)
            .with_context(|| format!("Bybit has no linear instrument {symbol}"))?;
        Ok(rt_domain::InstrumentRules {
            symbol: instrument.symbol,
            qty_step: instrument
                .lot_size_filter
                .qty_step
                .parse()
                .context("parse Bybit qty step")?,
            min_qty: instrument
                .lot_size_filter
                .min_order_qty
                .parse()
                .context("parse Bybit minimum order quantity")?,
            tick_size: instrument
                .price_filter
                .tick_size
                .parse()
                .context("parse Bybit tick size")?,
        })
    }

    /// List the currently tradable USDT linear contracts for safe WebSocket subscription.
    pub async fn active_linear_symbols(&self) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info", self.rest_url);
        let mut cursor: Option<String> = None;
        let mut symbols = Vec::new();
        loop {
            let mut request = self.http.get(&url).query(&[
                ("category", "linear"),
                ("status", "Trading"),
                ("limit", "1000"),
            ]);
            if let Some(value) = &cursor {
                request = request.query(&[("cursor", value)]);
            }
            let payload: ApiEnvelope<ActiveInstrumentsResult> = request
                .send()
                .await
                .context("request active Bybit linear instruments")?
                .error_for_status()
                .context("Bybit active instruments HTTP status")?
                .json()
                .await
                .context("decode active Bybit linear instruments")?;
            ensure_success(&payload)?;
            let result = payload
                .result
                .context("Bybit active instruments response missing result")?;
            symbols.extend(result.list.into_iter().map(|instrument| instrument.symbol));
            cursor = result.next_page_cursor.filter(|value| !value.is_empty());
            if cursor.is_none() {
                symbols.sort();
                symbols.dedup();
                return Ok(symbols);
            }
        }
    }

    /// Fetch all daily candles whose open time is in the requested inclusive interval.
    pub async fn daily_klines_range(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        if start > end {
            return Ok(Vec::new());
        }
        let url = format!("{}/v5/market/kline", self.rest_url);
        let mut cursor_end = end.timestamp_millis();
        let start_ms = start.timestamp_millis();
        let mut rows = BTreeMap::new();
        while cursor_end >= start_ms {
            let payload: ApiEnvelope<KlineResult> = self
                .http
                .get(&url)
                .query(&[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("interval", "D"),
                    ("end", &cursor_end.to_string()),
                    ("limit", "1000"),
                ])
                .send()
                .await
                .context("request Bybit kline range")?
                .error_for_status()
                .context("Bybit kline HTTP status")?
                .json()
                .await
                .context("decode Bybit kline range")?;
            ensure_success(&payload)?;
            let result = payload
                .result
                .context("Bybit kline response missing result")?;
            if result.list.is_empty() {
                break;
            }
            let mut earliest = i64::MAX;
            for row in result.list {
                let candle = parse_kline_row(symbol, &row, true)?;
                let timestamp = candle.opened_at.timestamp_millis();
                earliest = earliest.min(timestamp);
                if timestamp >= start_ms && timestamp <= end.timestamp_millis() {
                    rows.insert(timestamp, candle);
                }
            }
            if earliest == i64::MAX || earliest >= cursor_end {
                break;
            }
            cursor_end = earliest - 1;
        }
        Ok(rows.into_values().collect())
    }

    pub async fn connect_daily_ws(&self) -> Result<BybitDailyWs> {
        let (stream, _) = connect_async(&self.ws_url)
            .await
            .context("connect Bybit public WebSocket")?;
        Ok(BybitDailyWs {
            stream,
            pending: VecDeque::new(),
        })
    }

    pub async fn connect_minute_ws(&self) -> Result<BybitMinuteWs> {
        let (stream, _) = connect_async(&self.ws_url)
            .await
            .context("connect Bybit public minute WebSocket")?;
        Ok(BybitMinuteWs { stream })
    }
}

pub struct BybitDailyWs {
    stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    pending: VecDeque<Candle>,
}

impl BybitDailyWs {
    pub async fn subscribe(&mut self, symbols: &[String], batch_size: usize) -> Result<()> {
        if symbols.is_empty() {
            bail!("cannot subscribe to an empty daily symbol set");
        }
        let chunks = symbols.chunks(batch_size.max(1)).collect::<Vec<_>>();
        for (index, chunk) in chunks.iter().enumerate() {
            let topics = chunk
                .iter()
                .map(|symbol| format!("kline.D.{symbol}"))
                .collect::<Vec<_>>();
            self.stream
                .send(Message::Text(
                    json!({ "op": "subscribe", "args": topics })
                        .to_string()
                        .into(),
                ))
                .await
                .context("subscribe Bybit daily kline topics")?;
            self.await_subscription_ack().await?;
            if index + 1 < chunks.len() {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        Ok(())
    }

    async fn await_subscription_ack(&mut self) -> Result<()> {
        while let Some(message) = self.stream.next().await {
            let message = message.context("read Bybit subscription response")?;
            match message {
                Message::Text(text) => {
                    if let Some(candle) = parse_ws_kline(&text, "kline.D.")? {
                        self.pending.push_back(candle);
                        continue;
                    }
                    if is_subscription_ack(&text)? {
                        return Ok(());
                    }
                }
                Message::Ping(payload) => self
                    .stream
                    .send(Message::Pong(payload))
                    .await
                    .context("reply Bybit ping during subscription")?,
                Message::Close(_) => bail!("Bybit public WebSocket closed during subscription"),
                _ => {}
            }
        }
        bail!("Bybit public WebSocket ended before subscription acknowledgement")
    }

    /// Return a confirmed candle only. Partial daily candles are deliberately ignored.
    pub async fn next_confirmed_candle(&mut self) -> Result<Candle> {
        if let Some(candle) = self.pending.pop_front() {
            return Ok(candle);
        }
        while let Some(message) = self.stream.next().await {
            let message = message.context("read Bybit WebSocket frame")?;
            match message {
                Message::Text(text) => {
                    if let Some(candle) = parse_ws_kline(&text, "kline.D.")? {
                        return Ok(candle);
                    }
                }
                Message::Ping(payload) => self
                    .stream
                    .send(Message::Pong(payload))
                    .await
                    .context("reply Bybit ping")?,
                Message::Close(_) => bail!("Bybit public WebSocket closed"),
                _ => {}
            }
        }
        bail!("Bybit public WebSocket stream ended")
    }
}

pub struct BybitMinuteWs {
    stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
}

impl BybitMinuteWs {
    pub async fn subscribe(&mut self, symbols: &[String], batch_size: usize) -> Result<()> {
        if symbols.is_empty() {
            bail!("cannot subscribe to an empty minute symbol set");
        }
        for (index, chunk) in symbols.chunks(batch_size.max(1)).enumerate() {
            let topics = chunk
                .iter()
                .map(|symbol| format!("kline.1.{symbol}"))
                .collect::<Vec<_>>();
            self.stream
                .send(Message::Text(
                    json!({ "op": "subscribe", "args": topics })
                        .to_string()
                        .into(),
                ))
                .await
                .context("subscribe Bybit minute kline topics")?;
            self.await_subscription_ack().await?;
            if index + 1 < symbols.chunks(batch_size.max(1)).len() {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        Ok(())
    }

    async fn await_subscription_ack(&mut self) -> Result<()> {
        while let Some(message) = self.stream.next().await {
            match message.context("read Bybit minute subscription response")? {
                Message::Text(text) => {
                    if is_subscription_ack(&text)? {
                        return Ok(());
                    }
                }
                Message::Ping(payload) => self
                    .stream
                    .send(Message::Pong(payload))
                    .await
                    .context("reply Bybit ping during minute subscription")?,
                Message::Close(_) => {
                    bail!("Bybit public minute WebSocket closed during subscription")
                }
                _ => {}
            }
        }
        bail!("Bybit public minute WebSocket ended before subscription acknowledgement")
    }

    /// Return a completed one-minute candle only; partial candles never alter paper marks.
    pub async fn next_confirmed_candle(&mut self) -> Result<Candle> {
        while let Some(message) = self.stream.next().await {
            match message.context("read Bybit minute WebSocket frame")? {
                Message::Text(text) => {
                    if let Some(candle) = parse_ws_kline(&text, "kline.1.")? {
                        return Ok(candle);
                    }
                }
                Message::Ping(payload) => self
                    .stream
                    .send(Message::Pong(payload))
                    .await
                    .context("reply Bybit minute ping")?,
                Message::Close(_) => bail!("Bybit public minute WebSocket closed"),
                _ => {}
            }
        }
        bail!("Bybit public minute WebSocket stream ended")
    }
}

#[derive(Debug, Deserialize)]
struct ApiEnvelope<T> {
    #[serde(rename = "retCode")]
    ret_code: i64,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct OrderBookResult {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "u")]
    update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResult {
    list: Vec<InstrumentInfo>,
}

#[derive(Debug, Deserialize)]
struct ActiveInstrumentsResult {
    list: Vec<ActiveInstrumentInfo>,
    #[serde(rename = "nextPageCursor")]
    next_page_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ActiveInstrumentInfo {
    symbol: String,
}

#[derive(Debug, Deserialize)]
struct InstrumentInfo {
    symbol: String,
    #[serde(rename = "lotSizeFilter")]
    lot_size_filter: LotSizeFilter,
    #[serde(rename = "priceFilter")]
    price_filter: PriceFilter,
}

#[derive(Debug, Deserialize)]
struct LotSizeFilter {
    #[serde(rename = "qtyStep")]
    qty_step: String,
    #[serde(rename = "minOrderQty")]
    min_order_qty: String,
}

#[derive(Debug, Deserialize)]
struct PriceFilter {
    #[serde(rename = "tickSize")]
    tick_size: String,
}

fn ensure_success<T>(response: &ApiEnvelope<T>) -> Result<()> {
    if response.ret_code == 0 {
        Ok(())
    } else {
        bail!(
            "Bybit API error {}: {}",
            response.ret_code,
            response.ret_msg
        )
    }
}

fn parse_levels(levels: Vec<[String; 2]>) -> Result<Vec<PriceLevel>> {
    levels
        .into_iter()
        .map(|[price, quantity]| {
            Ok(PriceLevel {
                price: price.parse().context("parse order-book price")?,
                quantity: quantity.parse().context("parse order-book quantity")?,
            })
        })
        .collect()
}

fn parse_kline_row(symbol: &str, row: &[String], confirmed: bool) -> Result<Candle> {
    if row.len() < 6 {
        bail!("Bybit kline has {} fields, expected at least 6", row.len());
    }
    let opened_at = Utc
        .timestamp_millis_opt(row[0].parse().context("parse kline timestamp")?)
        .single()
        .context("invalid kline timestamp")?;
    Ok(Candle {
        symbol: symbol.to_owned(),
        opened_at,
        open: row[1].parse().context("parse kline open")?,
        high: row[2].parse().context("parse kline high")?,
        low: row[3].parse().context("parse kline low")?,
        close: row[4].parse().context("parse kline close")?,
        volume: row[5].parse().context("parse kline volume")?,
        confirmed,
    })
}

fn parse_ws_kline(source: &str, topic_prefix: &str) -> Result<Option<Candle>> {
    let root: Value = serde_json::from_str(source).context("parse Bybit WebSocket JSON")?;
    let Some(topic) = root.get("topic").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(symbol) = topic.strip_prefix(topic_prefix) else {
        return Ok(None);
    };
    let Some(data) = root
        .get("data")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
    else {
        return Ok(None);
    };
    if !data
        .get("confirm")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let field = |name: &str| -> Result<String> {
        data.get(name)
            .and_then(|value| match value {
                Value::String(text) => Some(text.to_owned()),
                Value::Number(number) => Some(number.to_string()),
                _ => None,
            })
            .with_context(|| format!("Bybit WebSocket candle is missing {name}"))
    };
    parse_kline_row(
        symbol,
        &[
            field("start")?,
            field("open")?,
            field("high")?,
            field("low")?,
            field("close")?,
            field("volume")?,
        ],
        true,
    )
    .map(Some)
}

fn is_subscription_ack(source: &str) -> Result<bool> {
    let root: Value = serde_json::from_str(source).context("parse Bybit subscription JSON")?;
    if root.get("op").and_then(Value::as_str) != Some("subscribe") {
        return Ok(false);
    }
    if root.get("success").and_then(Value::as_bool) == Some(true) {
        return Ok(true);
    }
    let message = root
        .get("ret_msg")
        .or_else(|| root.get("retMsg"))
        .and_then(Value::as_str)
        .unwrap_or("subscription rejected");
    bail!("Bybit WebSocket subscription failed: {message}")
}

#[cfg(test)]
mod tests {
    use super::{base_symbol, bybit_linear_symbol, is_subscription_ack, parse_ws_kline};

    #[test]
    fn accepts_only_confirmed_daily_kline() {
        let message = r#"{"topic":"kline.D.BTCUSDT","data":[{"start":1786579200000,"open":"100","high":"105","low":"99","close":"102","volume":"42","confirm":true}]}"#;
        let candle = parse_ws_kline(message, "kline.D.")
            .expect("parse succeeds")
            .expect("confirmed candle");
        assert_eq!(candle.symbol, "BTCUSDT");
        assert_eq!(candle.close, 102.0);

        let partial = message.replace("\"confirm\":true", "\"confirm\":false");
        assert!(
            parse_ws_kline(&partial, "kline.D.")
                .expect("parse succeeds")
                .is_none()
        );
    }

    #[test]
    fn accepts_only_confirmed_minute_kline() {
        let message = r#"{"topic":"kline.1.BTCUSDT","data":[{"start":1786579200000,"open":"100","high":"105","low":"99","close":"102","volume":"42","confirm":true}]}"#;
        let candle = parse_ws_kline(message, "kline.1.")
            .expect("parse succeeds")
            .expect("confirmed minute candle");
        assert_eq!(candle.symbol, "BTCUSDT");
        assert_eq!(candle.close, 102.0);
    }

    #[test]
    fn accepts_only_successful_subscription_acknowledgements() {
        assert!(is_subscription_ack(r#"{"op":"subscribe","success":true}"#).expect("ack"));
        assert!(!is_subscription_ack(r#"{"topic":"kline.D.BTCUSDT"}"#).expect("non-ack"));
        assert!(
            is_subscription_ack(r#"{"op":"subscribe","success":false,"ret_msg":"bad topic"}"#)
                .is_err()
        );
    }

    #[test]
    fn maps_contract_symbols_without_guessing_the_quote_currency() {
        assert_eq!(bybit_linear_symbol("BTC"), "BTCUSDT");
        assert_eq!(base_symbol("BTCUSDT").expect("base"), "BTC");
        assert!(base_symbol("BTCUSD").is_err());
    }
}
