use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use rt_model::{LightGbmLibrary, inspect_bundle};
use rt_panel::{build_features, load_daily_csv_panel};
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let arguments = Arguments::parse()?;
    let bundle = inspect_bundle(&arguments.bundle)?;
    let market = load_daily_csv_panel(&arguments.data)?;
    if market.symbols != bundle.universe.symbols {
        bail!("dataset universe does not match immutable model contract");
    }
    let panel = build_features(market, 180, 150)?;
    if panel.feature_names != bundle.features.names {
        bail!("feature schema does not match immutable model contract");
    }
    let start = parse_date(&arguments.start)?;
    let end = parse_date(&arguments.end)?;
    if end < start {
        bail!("--end precedes --start");
    }
    let rows = (0..panel.market.dates.len())
        .filter(|row| panel.market.dates[*row] >= start && panel.market.dates[*row] <= end)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        bail!("requested date range is absent from the dataset");
    }
    let mut features = Vec::new();
    let mut locations = Vec::new();
    for row in rows {
        for col in 0..panel.market.symbols.len() {
            let index = row * panel.market.symbols.len() + col;
            if !panel.eligible_max[index] || !panel.valid_features[index] {
                continue;
            }
            for feature in &panel.feature_stack {
                features.push(feature.get(row, col));
            }
            locations.push((row, col));
        }
    }
    let library = LightGbmLibrary::linked();
    let mut predictions = Vec::new();
    for model in &bundle.manifest.models {
        let mut booster = library.load_booster(bundle.root.join(&model.file))?;
        predictions.push(booster.predict_rows(&features, panel.feature_names.len())?);
    }
    let mut writer = csv::Writer::from_path(&arguments.output)
        .with_context(|| format!("write {}", arguments.output.display()))?;
    let mut header = vec!["date".to_owned(), "symbol".to_owned()];
    header.extend(
        bundle
            .manifest
            .models
            .iter()
            .map(|model| format!("seed_{}", model.seed)),
    );
    writer.write_record(&header)?;
    for (index, (row, col)) in locations.into_iter().enumerate() {
        let mut record = vec![
            panel.market.dates[row].to_string(),
            panel.market.symbols[col].clone(),
        ];
        record.extend(predictions.iter().map(|scores| scores[index].to_string()));
        writer.write_record(&record)?;
    }
    writer.flush()?;
    println!(
        "Scored {} eligible symbol-days into {}.",
        features.len() / panel.feature_names.len(),
        arguments.output.display()
    );
    Ok(())
}

struct Arguments {
    data: PathBuf,
    bundle: PathBuf,
    start: String,
    end: String,
    output: PathBuf,
}

impl Arguments {
    fn parse() -> Result<Self> {
        let mut data = None;
        let mut bundle = None;
        let mut start = None;
        let mut end = None;
        let mut output = None;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            let value = values
                .next()
                .with_context(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--data" => data = Some(PathBuf::from(value)),
                "--bundle" => bundle = Some(PathBuf::from(value)),
                "--start" => start = Some(value),
                "--end" => end = Some(value),
                "--output" => output = Some(PathBuf::from(value)),
                _ => bail!("unknown argument {flag}"),
            }
        }
        Ok(Self {
            data: data.context("--data is required")?,
            bundle: bundle.context("--bundle is required")?,
            start: start.context("--start is required")?,
            end: end.context("--end is required")?,
            output: output.context("--output is required")?,
        })
    }
}

fn parse_date(value: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").with_context(|| format!("parse date {value}"))
}
