use anyhow::{Context, Result};
use rt_panel::{build_features, load_daily_csv_panel};
use serde::Serialize;
use std::{env, fs, path::PathBuf};

#[derive(Serialize)]
struct Sample {
    date: String,
    symbol: String,
    liquidity_rank: f32,
    beta60: f32,
    vol20: f32,
    eligible: bool,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    dates: usize,
    symbols: usize,
    start: String,
    end: String,
    feature_names: Vec<String>,
    feature_hashes: Vec<String>,
    samples: Vec<Sample>,
}

fn main() -> Result<()> {
    let data_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/1d"));
    let market = load_daily_csv_panel(&data_dir)?;
    let panel = build_features(market, 180, 150)?;
    if let Some(path) = env::args().nth(2) {
        let mut bytes = Vec::with_capacity(
            panel.feature_stack.len()
                * panel.feature_stack[0].values.len()
                * std::mem::size_of::<f32>(),
        );
        for feature in &panel.feature_stack {
            for value in &feature.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        for value in &panel.beta60.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, bytes).with_context(|| format!("write {path}"))?;
    }
    let row = panel
        .market
        .dates
        .len()
        .checked_sub(1)
        .context("panel has no dates")?;
    let sample_symbols = ["BTC", "ETH", "SOL"];
    let samples = sample_symbols
        .into_iter()
        .filter_map(|symbol| {
            panel
                .market
                .symbols
                .iter()
                .position(|item| item == symbol)
                .map(|col| (symbol, col))
        })
        .map(|(symbol, col)| Sample {
            date: panel.market.dates[row].to_string(),
            symbol: symbol.to_owned(),
            liquidity_rank: panel.liquidity_rank.get(row, col),
            beta60: panel.beta60.get(row, col),
            vol20: panel.vol20.get(row, col),
            eligible: panel.eligible_max[row * panel.market.symbols.len() + col],
            values: panel
                .feature_stack
                .iter()
                .map(|feature| feature.get(row, col))
                .collect(),
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&Output {
            dates: panel.market.dates.len(),
            symbols: panel.market.symbols.len(),
            start: panel.market.dates[0].to_string(),
            end: panel
                .market
                .dates
                .last()
                .context("panel has no dates")?
                .to_string(),
            feature_names: panel.feature_names,
            feature_hashes: panel.feature_stack.iter().map(hash_dense).collect(),
            samples,
        })?
    );
    Ok(())
}

fn hash_dense(feature: &rt_panel::Dense) -> String {
    let mut state = 0xcbf29ce484222325_u64;
    for value in &feature.values {
        for byte in value.to_bits().to_le_bytes() {
            state ^= u64::from(byte);
            state = state.wrapping_mul(0x100000001b3);
        }
    }
    format!("{state:016x}")
}
