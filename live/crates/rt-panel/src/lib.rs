use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, Timelike};
use rt_domain::Candle;
use std::{fs, path::Path};

pub const EPSILON: f32 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl Dense {
    pub fn nan(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![f32::NAN; rows * cols],
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.values[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.values[row * self.cols + col] = value;
    }

    pub fn row(&self, row: usize) -> &[f32] {
        &self.values[row * self.cols..(row + 1) * self.cols]
    }
}

#[derive(Debug, Clone)]
pub struct MarketPanel {
    pub dates: Vec<NaiveDate>,
    pub symbols: Vec<String>,
    pub open: Dense,
    pub high: Dense,
    pub low: Dense,
    pub close: Dense,
    pub volume: Dense,
    pub first_idx: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FeaturePanel {
    pub market: MarketPanel,
    pub oo_return: Dense,
    pub ret1: Dense,
    pub dvol20: Dense,
    pub liquidity_rank: Dense,
    pub eligible_max: Vec<bool>,
    pub beta60: Dense,
    pub vol20: Dense,
    pub btc_vol20: Vec<f32>,
    pub feature_names: Vec<String>,
    pub feature_stack: Vec<Dense>,
    pub valid_features: Vec<bool>,
    pub btc_idx: usize,
}

/// Merge confirmed UTC daily candles into the causal market panel.
pub fn merge_confirmed_daily_candles(market: &mut MarketPanel, candles: &[Candle]) -> Result<()> {
    if candles.is_empty() {
        return Ok(());
    }
    let date = candles[0].opened_at.date_naive();
    if candles.iter().any(|candle| {
        !candle.confirmed
            || candle.opened_at.date_naive() != date
            || candle.opened_at.time().num_seconds_from_midnight() != 0
    }) {
        bail!("daily candle merge requires confirmed candles opened at one UTC midnight");
    }
    let last = market
        .dates
        .last()
        .copied()
        .context("market panel has no dates")?;
    let row = if date == last {
        market.dates.len() - 1
    } else if date == last.succ_opt().context("date cannot advance")? {
        market.dates.push(date);
        extend_nan_row(&mut market.open);
        extend_nan_row(&mut market.high);
        extend_nan_row(&mut market.low);
        extend_nan_row(&mut market.close);
        extend_nan_row(&mut market.volume);
        market.dates.len() - 1
    } else {
        bail!("daily candle date {date} is not contiguous after {last}");
    };
    for candle in candles {
        let column = market
            .symbols
            .iter()
            .position(|symbol| symbol == &candle.symbol)
            .with_context(|| {
                format!(
                    "candle symbol {} is absent from model universe",
                    candle.symbol
                )
            })?;
        if ![
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            bail!("candle {} has non-finite OHLCV", candle.symbol);
        }
        market.open.set(row, column, candle.open as f32);
        market.high.set(row, column, candle.high as f32);
        market.low.set(row, column, candle.low as f32);
        market.close.set(row, column, candle.close as f32);
        market.volume.set(row, column, candle.volume as f32);
    }
    Ok(())
}

fn extend_nan_row(matrix: &mut Dense) {
    matrix
        .values
        .extend(std::iter::repeat_n(f32::NAN, matrix.cols));
    matrix.rows += 1;
}

/// Reproduce the current seven-day residual-return decile labels used by LightGBM LambdaRank.
pub fn make_target_deciles(panel: &FeaturePanel, horizon: usize) -> Result<Dense> {
    if horizon == 0 || horizon + 1 >= panel.market.dates.len() {
        bail!("invalid target horizon {horizon}");
    }
    let rows = panel.market.dates.len();
    let cols = panel.market.symbols.len();
    let mut future = Dense::nan(rows, cols);
    for row in 0..rows - horizon - 1 {
        for col in 0..cols {
            let entry = panel.market.open.get(row + 1, col);
            let exit = panel.market.open.get(row + horizon + 1, col);
            if entry.is_finite() && exit.is_finite() && entry > 0.0 && exit > 0.0 {
                future.set(row, col, (exit / entry).ln());
            }
        }
    }
    let btc_future = (0..rows)
        .map(|row| future.get(row, panel.btc_idx))
        .collect::<Vec<_>>();
    let residual = binary(
        &future,
        &binary(
            &panel.beta60,
            &broadcast(&btc_future, cols),
            |left, right| left * right,
        ),
        |left, right| left - right,
    );
    let mask = future
        .values
        .iter()
        .zip(&panel.eligible_max)
        .map(|(value, eligible)| value.is_finite() && *eligible)
        .collect::<Vec<_>>();
    let ranks = rank_rows(&residual, &mask, false, false);
    Ok(unary(&ranks, |rank| {
        if rank.is_finite() {
            (rank.clamp(0.0, 0.999_999) * 10.0).floor()
        } else {
            f32::NAN
        }
    }))
}

pub fn load_daily_csv_panel(directory: impl AsRef<Path>) -> Result<MarketPanel> {
    let mut files = fs::read_dir(directory.as_ref())
        .with_context(|| format!("read data directory {}", directory.as_ref().display()))?
        .filter_map(|item| item.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_1440.csv"))
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        bail!("no *_1440.csv files in {}", directory.as_ref().display());
    }
    let mut data = Vec::with_capacity(files.len());
    let mut start = None;
    let mut end = None;
    for path in &files {
        let symbol = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("non-UTF8 symbol filename")?
            .strip_suffix("_1440.csv")
            .context("invalid kline filename")?
            .to_owned();
        let mut reader =
            csv::Reader::from_path(path).with_context(|| format!("open {}", path.display()))?;
        let mut rows = Vec::new();
        for row in reader.deserialize::<CsvRow>() {
            let row = row.with_context(|| format!("decode {}", path.display()))?;
            let date =
                NaiveDate::parse_from_str(&row.timestamp, "%Y-%m-%d").with_context(|| {
                    format!("parse timestamp {} in {}", row.timestamp, path.display())
                })?;
            start = Some(start.map_or(date, |current: NaiveDate| current.min(date)));
            end = Some(end.map_or(date, |current: NaiveDate| current.max(date)));
            rows.push((date, row));
        }
        data.push((symbol, rows));
    }
    let start = start.context("no daily rows")?;
    let end = end.context("no daily rows")?;
    let days = (end - start).num_days() as usize + 1;
    let symbols = data
        .iter()
        .map(|(symbol, _)| symbol.clone())
        .collect::<Vec<_>>();
    let mut open = Dense::nan(days, symbols.len());
    let mut high = Dense::nan(days, symbols.len());
    let mut low = Dense::nan(days, symbols.len());
    let mut close = Dense::nan(days, symbols.len());
    let mut volume = Dense::nan(days, symbols.len());
    let mut first_idx = vec![days; symbols.len()];
    for (col, (_, rows)) in data.into_iter().enumerate() {
        for (date, row) in rows {
            let index = (date - start).num_days() as usize;
            open.set(index, col, row.open);
            high.set(index, col, row.high);
            low.set(index, col, row.low);
            close.set(index, col, row.close);
            volume.set(index, col, row.volume);
            first_idx[col] = first_idx[col].min(index);
        }
    }
    let dates = (0..days)
        .map(|offset| start + chrono::Days::new(offset as u64))
        .collect();
    Ok(MarketPanel {
        dates,
        symbols,
        open,
        high,
        low,
        close,
        volume,
        first_idx,
    })
}

pub fn build_features(
    market: MarketPanel,
    min_age: usize,
    max_universe: usize,
) -> Result<FeaturePanel> {
    let (rows, cols) = (market.close.rows, market.close.cols);
    let btc_idx = market
        .symbols
        .iter()
        .position(|symbol| symbol == "BTC")
        .context("BTC contract is required")?;
    let log_close = unary(&market.close, |value| {
        if value.is_finite() && value > 0.0 {
            value.ln()
        } else {
            f32::NAN
        }
    });
    let mut ret1 = Dense::nan(rows, cols);
    let mut oo_return = Dense::nan(rows, cols);
    for row in 1..rows {
        for col in 0..cols {
            ret1.set(
                row,
                col,
                log_close.get(row, col) - log_close.get(row - 1, col),
            );
        }
    }
    for row in 0..rows.saturating_sub(1) {
        for col in 0..cols {
            let left = market.open.get(row, col);
            let right = market.open.get(row + 1, col);
            oo_return.set(
                row,
                col,
                if left.is_finite() && right.is_finite() && left > 0.0 {
                    right / left - 1.0
                } else {
                    f32::NAN
                },
            );
        }
    }
    let dollar_volume = binary(&market.close, &market.volume, |left, right| left * right);
    let log_dollar_volume = unary(&dollar_volume, |value| {
        if value.is_finite() {
            value.max(0.0).ln_1p()
        } else {
            f32::NAN
        }
    });
    let dvol20 = rolling_mean(&log_dollar_volume, 20, 10);
    let dvol60 = rolling_mean(&log_dollar_volume, 60, 30);
    let mut base_valid = vec![false; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            base_valid[row * cols + col] = row >= market.first_idx[col] + min_age
                && dvol20.get(row, col).is_finite()
                && market.close.get(row, col).is_finite();
        }
    }
    let liquidity_rank = rank_rows(&dvol20, &base_valid, true, true);
    let eligible_max = liquidity_rank
        .values
        .iter()
        .zip(&base_valid)
        .map(|(rank, valid)| *valid && rank.is_finite() && *rank <= max_universe as f32)
        .collect::<Vec<_>>();

    let mut btc_return = vec![f32::NAN; rows];
    let mut btc_return_f64 = vec![f64::NAN; rows];
    for row in 0..rows {
        btc_return[row] = ret1.get(row, btc_idx);
        btc_return_f64[row] = f64::from(btc_return[row]);
    }
    let mean_r60 = rolling_mean(&ret1, 60, 30);
    let mean_b60 = rolling_mean_vector_f64(&btc_return_f64, 60, 30);
    let mean_rb60 = rolling_mean_product_f64(&ret1, &btc_return_f64, 60, 30);
    let mean_b2_60 = rolling_mean_vector_f64(
        &btc_return_f64
            .iter()
            .map(|value| value * value)
            .collect::<Vec<_>>(),
        60,
        30,
    );
    let mut beta60 = Dense::nan(rows, cols);
    for row in 0..rows {
        let var_b = (mean_b2_60[row] - mean_b60[row] * mean_b60[row]).max(1e-8);
        for col in 0..cols {
            beta60.set(
                row,
                col,
                ((f64::from(mean_rb60.get(row, col))
                    - f64::from(mean_r60.get(row, col)) * mean_b60[row])
                    / var_b)
                    .clamp(-3.0, 5.0) as f32,
            );
        }
    }

    let mut raw = Vec::<(String, Dense)>::new();
    for window in [1_usize, 2, 3, 5, 7, 14, 21, 30, 60, 90, 120] {
        let values = if window == 1 {
            ret1.clone()
        } else {
            lagged_log_return(&log_close, window)
        };
        raw.push((format!("ret_{window}"), values));
    }
    for window in [5_usize, 10, 20, 60] {
        raw.push((
            format!("vol_{window}"),
            rolling_std(&ret1, window, (window / 2).max(3)),
        ));
    }
    let negative = unary(&ret1, |value| value.min(0.0));
    raw.push((
        "downvol_20".to_owned(),
        unary(
            &rolling_mean(
                &binary(&negative, &negative, |left, right| left * right),
                20,
                10,
            ),
            |value| value.sqrt(),
        ),
    ));
    raw.push((
        "downvol_60".to_owned(),
        unary(
            &rolling_mean(
                &binary(&negative, &negative, |left, right| left * right),
                60,
                30,
            ),
            |value| value.sqrt(),
        ),
    ));
    let log_range = binary(&market.high, &market.low, |high, low| {
        if high.is_finite() && low.is_finite() && high > 0.0 && low > 0.0 {
            (high.max(EPSILON) / low.max(EPSILON)).ln()
        } else {
            f32::NAN
        }
    });
    raw.push(("range_1".to_owned(), log_range.clone()));
    raw.push(("range_5".to_owned(), rolling_mean(&log_range, 5, 3)));
    raw.push(("range_20".to_owned(), rolling_mean(&log_range, 20, 10)));
    let denominator = binary(&market.high, &market.low, |high, low| {
        (high - low).max(EPSILON)
    });
    let close_loc1 = ternary(
        &market.close,
        &market.low,
        &denominator,
        |close, low, denominator| (close - low) / denominator - 0.5,
    );
    raw.push(("close_loc_1".to_owned(), close_loc1.clone()));
    raw.push(("close_loc_5".to_owned(), rolling_mean(&close_loc1, 5, 3)));
    let body1 = ternary(
        &market.close,
        &market.open,
        &denominator,
        |close, open, denominator| (close - open) / denominator,
    );
    raw.push(("body_1".to_owned(), body1.clone()));
    raw.push(("body_5".to_owned(), rolling_mean(&body1, 5, 3)));
    raw.push((
        "volume_5_20".to_owned(),
        binary(
            &rolling_mean(&log_dollar_volume, 5, 3),
            &dvol20,
            |left, right| left - right,
        ),
    ));
    raw.push((
        "volume_20_60".to_owned(),
        binary(&dvol20, &dvol60, |left, right| left - right),
    ));
    raw.push(("dvol_level".to_owned(), dvol20.clone()));
    raw.push((
        "amihud_20".to_owned(),
        rolling_mean(
            &binary(&unary(&ret1, f32::abs), &dollar_volume, |ret, volume| {
                ret / volume.max(1.0)
            }),
            20,
            10,
        ),
    ));
    for window in [20_usize, 60, 120] {
        raw.push((
            format!("dist_high_{window}"),
            binary(
                &market.close,
                &rolling_max(&market.close, window, (window / 2).max(2)),
                |close, high| close / high - 1.0,
            ),
        ));
    }
    raw.push(("beta_60".to_owned(), beta60.clone()));
    let ret21 = raw
        .iter()
        .find(|(name, _)| name == "ret_21")
        .expect("ret_21 exists")
        .1
        .clone();
    let ret60 = raw
        .iter()
        .find(|(name, _)| name == "ret_60")
        .expect("ret_60 exists")
        .1
        .clone();
    let ret3 = raw
        .iter()
        .find(|(name, _)| name == "ret_3")
        .expect("ret_3 exists")
        .1
        .clone();
    let ret7 = raw
        .iter()
        .find(|(name, _)| name == "ret_7")
        .expect("ret_7 exists")
        .1
        .clone();
    let btc_ret21 = (0..rows)
        .map(|row| ret21.get(row, btc_idx))
        .collect::<Vec<_>>();
    let btc_ret60 = (0..rows)
        .map(|row| ret60.get(row, btc_idx))
        .collect::<Vec<_>>();
    raw.push((
        "resmom_21".to_owned(),
        binary(
            &ret21,
            &binary(&beta60, &broadcast(&btc_ret21, cols), |left, right| {
                left * right
            }),
            |left, right| left - right,
        ),
    ));
    raw.push((
        "resmom_60".to_owned(),
        binary(
            &ret60,
            &binary(&beta60, &broadcast(&btc_ret60, cols), |left, right| {
                left * right
            }),
            |left, right| left - right,
        ),
    ));
    raw.push(("short_reversal".to_owned(), unary(&ret3, |value| -value)));
    raw.push((
        "mom_21_ex_3".to_owned(),
        binary(&ret21, &ret3, |left, right| left - right),
    ));
    raw.push((
        "mom_60_ex_7".to_owned(),
        binary(&ret60, &ret7, |left, right| left - right),
    ));
    if raw.len() != 37 {
        bail!(
            "internal feature definition has {} raw features, expected 37",
            raw.len()
        );
    }

    let mut feature_names = Vec::new();
    let mut feature_stack = Vec::new();
    for (name, values) in raw {
        feature_names.push(format!("{name}_xrank"));
        let ranks = rank_rows(&values, &eligible_max, false, false);
        feature_stack.push(unary(&ranks, |value| value - 0.5));
    }
    let ret7_btc = (0..rows)
        .map(|row| ret7.get(row, btc_idx))
        .collect::<Vec<_>>();
    let vol20 = rolling_std(&ret1, 20, 10);
    let btc_vol20 = (0..rows)
        .map(|row| vol20.get(row, btc_idx))
        .collect::<Vec<_>>();
    let breadth = cross_mean_bool(&ret1, &eligible_max, |value| value > 0.0);
    let dispersion = cross_std(&ret1, &eligible_max);
    let regimes = [
        ("mkt_btc_ret7", ret7_btc),
        ("mkt_btc_ret21", btc_ret21),
        ("mkt_btc_vol20", btc_vol20.clone()),
        ("mkt_breadth", breadth),
        ("mkt_dispersion", dispersion),
    ];
    for (name, values) in regimes {
        feature_names.push(name.to_owned());
        feature_stack.push(broadcast(&values, cols));
    }
    if feature_names != rt_strategy::ranktrend_feature_names() {
        bail!("Rust feature order does not match the frozen RankTrend contract");
    }
    let mut valid_features = vec![true; rows * cols];
    for feature in &feature_stack {
        for (index, value) in feature.values.iter().enumerate() {
            valid_features[index] &= value.is_finite();
        }
    }
    Ok(FeaturePanel {
        market,
        oo_return,
        ret1,
        dvol20,
        liquidity_rank,
        eligible_max,
        beta60,
        vol20,
        btc_vol20,
        feature_names,
        feature_stack,
        valid_features,
        btc_idx,
    })
}

#[derive(Debug, serde::Deserialize)]
struct CsvRow {
    timestamp: String,
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    volume: f32,
}

fn unary(source: &Dense, f: impl Fn(f32) -> f32) -> Dense {
    Dense {
        rows: source.rows,
        cols: source.cols,
        values: source.values.iter().copied().map(f).collect(),
    }
}
fn binary(left: &Dense, right: &Dense, f: impl Fn(f32, f32) -> f32) -> Dense {
    assert_eq!((left.rows, left.cols), (right.rows, right.cols));
    Dense {
        rows: left.rows,
        cols: left.cols,
        values: left
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| f(*left, *right))
            .collect(),
    }
}
fn ternary(
    first: &Dense,
    second: &Dense,
    third: &Dense,
    f: impl Fn(f32, f32, f32) -> f32,
) -> Dense {
    assert_eq!((first.rows, first.cols), (second.rows, second.cols));
    assert_eq!((first.rows, first.cols), (third.rows, third.cols));
    Dense {
        rows: first.rows,
        cols: first.cols,
        values: first
            .values
            .iter()
            .zip(&second.values)
            .zip(&third.values)
            .map(|((first, second), third)| f(*first, *second, *third))
            .collect(),
    }
}
fn broadcast(values: &[f32], cols: usize) -> Dense {
    let mut output = Dense::nan(values.len(), cols);
    for (row, value) in values.iter().enumerate() {
        for col in 0..cols {
            output.set(row, col, *value);
        }
    }
    output
}

fn lagged_log_return(log_close: &Dense, window: usize) -> Dense {
    let mut output = Dense::nan(log_close.rows, log_close.cols);
    for row in window..log_close.rows {
        for col in 0..log_close.cols {
            output.set(
                row,
                col,
                log_close.get(row, col) - log_close.get(row - window, col),
            );
        }
    }
    output
}
fn rolling_mean(source: &Dense, window: usize, min_periods: usize) -> Dense {
    let mut output = Dense::nan(source.rows, source.cols);
    for col in 0..source.cols {
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for row in 0..source.rows {
            let value = source.get(row, col);
            if value.is_finite() {
                sum += value as f64;
                count += 1;
            }
            if row >= window {
                let old = source.get(row - window, col);
                if old.is_finite() {
                    sum -= old as f64;
                    count -= 1;
                }
            }
            if count >= min_periods {
                output.set(row, col, (sum / count as f64) as f32);
            }
        }
    }
    output
}
fn rolling_std(source: &Dense, window: usize, min_periods: usize) -> Dense {
    let mut output = Dense::nan(source.rows, source.cols);
    for col in 0..source.cols {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0_usize;
        for row in 0..source.rows {
            let value = source.get(row, col);
            if value.is_finite() {
                sum += value as f64;
                sum_sq += (value as f64).powi(2);
                count += 1;
            }
            if row >= window {
                let old = source.get(row - window, col);
                if old.is_finite() {
                    sum -= old as f64;
                    sum_sq -= (old as f64).powi(2);
                    count -= 1;
                }
            }
            if count >= min_periods {
                output.set(
                    row,
                    col,
                    ((sum_sq / count as f64 - (sum / count as f64).powi(2))
                        .max(0.0)
                        .sqrt()) as f32,
                );
            }
        }
    }
    output
}
fn rolling_max(source: &Dense, window: usize, min_periods: usize) -> Dense {
    let mut output = Dense::nan(source.rows, source.cols);
    for row in 0..source.rows {
        let start = row.saturating_add(1).saturating_sub(window);
        for col in 0..source.cols {
            let mut count = 0;
            let mut maximum = f32::NEG_INFINITY;
            for scan in start..=row {
                let value = source.get(scan, col);
                if value.is_finite() {
                    count += 1;
                    maximum = maximum.max(value);
                }
            }
            if count >= min_periods {
                output.set(row, col, maximum);
            }
        }
    }
    output
}
fn rolling_mean_vector_f64(source: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; source.len()];
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for row in 0..source.len() {
        if source[row].is_finite() {
            sum += source[row];
            count += 1;
        }
        if row >= window && source[row - window].is_finite() {
            sum -= source[row - window];
            count -= 1;
        }
        if count >= min_periods {
            output[row] = sum / count as f64;
        }
    }
    output
}
fn rolling_mean_product_f64(
    source: &Dense,
    multiplier: &[f64],
    window: usize,
    min_periods: usize,
) -> Dense {
    let mut output = Dense::nan(source.rows, source.cols);
    for col in 0..source.cols {
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for row in 0..source.rows {
            let value = f64::from(source.get(row, col)) * multiplier[row];
            if value.is_finite() {
                sum += value;
                count += 1;
            }
            if row >= window {
                let old = f64::from(source.get(row - window, col)) * multiplier[row - window];
                if old.is_finite() {
                    sum -= old;
                    count -= 1;
                }
            }
            if count >= min_periods {
                output.set(row, col, (sum / count as f64) as f32);
            }
        }
    }
    output
}

fn rank_rows(source: &Dense, eligible: &[bool], descending: bool, first_tie: bool) -> Dense {
    let mut output = Dense::nan(source.rows, source.cols);
    for row in 0..source.rows {
        let mut indices = (0..source.cols)
            .filter(|col| eligible[row * source.cols + *col] && source.get(row, *col).is_finite())
            .collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            if descending {
                source.get(row, *right).total_cmp(&source.get(row, *left))
            } else {
                source.get(row, *left).total_cmp(&source.get(row, *right))
            }
        });
        let count = indices.len();
        let mut start = 0;
        while start < count {
            let mut end = start + 1;
            while end < count
                && source
                    .get(row, indices[start])
                    .total_cmp(&source.get(row, indices[end]))
                    == std::cmp::Ordering::Equal
            {
                end += 1;
            }
            if first_tie {
                for (offset, index) in indices[start..end].iter().enumerate() {
                    output.set(row, *index, (start + offset + 1) as f32);
                }
            } else {
                let rank = (start + 1 + end) as f32 / 2.0 / count as f32;
                for index in &indices[start..end] {
                    output.set(row, *index, rank);
                }
            }
            start = end;
        }
    }
    output
}
fn cross_mean_bool(source: &Dense, eligible: &[bool], predicate: impl Fn(f32) -> bool) -> Vec<f32> {
    (0..source.rows)
        .map(|row| {
            let values =
                (0..source.cols)
                    .filter_map(|col| {
                        let value = source.get(row, col);
                        (eligible[row * source.cols + col] && value.is_finite())
                            .then_some(predicate(value) as u8 as f32)
                    })
                    .collect::<Vec<_>>();
            if values.is_empty() {
                f32::NAN
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        })
        .collect()
}
fn cross_std(source: &Dense, eligible: &[bool]) -> Vec<f32> {
    (0..source.rows)
        .map(|row| {
            let values = (0..source.cols)
                .filter_map(|col| {
                    let value = source.get(row, col);
                    (eligible[row * source.cols + col] && value.is_finite()).then_some(value)
                })
                .collect::<Vec<_>>();
            if values.is_empty() {
                return f32::NAN;
            }
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            (values
                .iter()
                .map(|value| (*value as f64 - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64)
                .sqrt() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Dense, MarketPanel, cross_mean_bool, merge_confirmed_daily_candles, rolling_mean};
    use chrono::{TimeZone, Utc};
    use rt_domain::Candle;

    #[test]
    fn rolling_mean_uses_available_observations_and_minimum() {
        let source = Dense {
            rows: 4,
            cols: 1,
            values: vec![1.0, f32::NAN, 3.0, 5.0],
        };
        let result = rolling_mean(&source, 3, 2);
        assert!(result.get(0, 0).is_nan());
        assert!(result.get(1, 0).is_nan());
        assert_eq!(result.get(2, 0), 2.0);
        assert_eq!(result.get(3, 0), 4.0);
    }

    #[test]
    fn breadth_only_uses_eligible_values() {
        let source = Dense {
            rows: 1,
            cols: 3,
            values: vec![1.0, -1.0, 2.0],
        };
        assert_eq!(
            cross_mean_bool(&source, &[true, true, false], |value| value > 0.0),
            vec![0.5]
        );
    }

    #[test]
    fn merges_contiguous_confirmed_candles_and_preserves_missing_assets() {
        let date = chrono::NaiveDate::from_ymd_opt(2026, 8, 1).expect("date");
        let mut market = MarketPanel {
            dates: vec![date],
            symbols: vec!["BTC".to_owned(), "ETH".to_owned()],
            open: Dense {
                rows: 1,
                cols: 2,
                values: vec![1.0, 2.0],
            },
            high: Dense {
                rows: 1,
                cols: 2,
                values: vec![1.0, 2.0],
            },
            low: Dense {
                rows: 1,
                cols: 2,
                values: vec![1.0, 2.0],
            },
            close: Dense {
                rows: 1,
                cols: 2,
                values: vec![1.0, 2.0],
            },
            volume: Dense {
                rows: 1,
                cols: 2,
                values: vec![1.0, 2.0],
            },
            first_idx: vec![0, 0],
        };
        merge_confirmed_daily_candles(
            &mut market,
            &[Candle {
                symbol: "BTC".to_owned(),
                opened_at: Utc
                    .with_ymd_and_hms(2026, 8, 2, 0, 0, 0)
                    .single()
                    .expect("time"),
                open: 3.0,
                high: 4.0,
                low: 2.0,
                close: 3.5,
                volume: 5.0,
                confirmed: true,
            }],
        )
        .expect("merge");
        assert_eq!(market.dates.len(), 2);
        assert_eq!(market.close.get(1, 0), 3.5);
        assert!(market.close.get(1, 1).is_nan());
    }
}
