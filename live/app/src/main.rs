use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
};
use chrono::{DateTime, NaiveDate, Utc};
use futures_util::{StreamExt, stream};
use parking_lot::Mutex;
use rt_bybit::{BybitPublicClient, base_symbol, bybit_linear_symbol};
use rt_domain::{AccountSnapshot, Candle, InstrumentRules, OrderBookSnapshot};
use rt_engine::{PaperConfig, PaperEngine, RiskLimits};
use rt_execution::SnapshotExecutionConfig;
use rt_ledger::{EquityPoint, Ledger};
use rt_model::{BundleMetadata, LightGbmLibrary, NativeBooster, inspect_bundle};
use rt_panel::{
    FeaturePanel, MarketPanel, build_features, load_daily_csv_panel, merge_confirmed_daily_candles,
};
use rt_strategy::{
    OverlayRules, PortfolioRules, apply_volatility_overlay, build_daily_weights,
    median_rank_ensemble, smooth_fixed_window,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tower_http::trace::TraceLayer;

const PAPER_SESSION_ID: &str = "ranktrend-paper-v1";

#[derive(Debug, Clone, Deserialize)]
struct RuntimeConfig {
    model: ModelConfig,
    data: DataConfig,
    paper: PaperSection,
    server: ServerConfig,
    storage: StorageConfig,
    bybit: BybitConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfig {
    bundle: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct DataConfig {
    directory: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct PaperSection {
    initial_equity_usd: f64,
    gross_leverage: f64,
    max_gross_leverage: f64,
    fee_bps: f64,
    reject_partial_fills: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ServerConfig {
    bind: String,
}

#[derive(Debug, Clone, Deserialize)]
struct StorageConfig {
    ledger_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct BybitConfig {
    orderbook_depth: u16,
    rest_parallelism: usize,
    ws_batch_size: usize,
    close_grace_seconds: u64,
    minimum_confirmed_coverage: f64,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<PaperEngine>>,
    ledger: Arc<Mutex<Ledger>>,
    instrument_rules: Arc<Mutex<BTreeMap<String, InstrumentRules>>>,
    model: ModelView,
    runtime: Arc<Mutex<RuntimeStatus>>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelView {
    bundle_id: String,
    backend: String,
    horizon_days: u32,
    cutoff_date: String,
    seed_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeStatus {
    status: String,
    detail: String,
    last_decision_date: Option<String>,
    last_error: Option<String>,
}

impl RuntimeStatus {
    fn booting() -> Self {
        Self {
            status: "booting".to_owned(),
            detail: "Validating immutable model bundle and recovering daily candles.".to_owned(),
            last_decision_date: None,
            last_error: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct SessionView {
    status: String,
    detail: String,
    model: ModelView,
    account: AccountSnapshot,
    last_decision_date: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct PositionView {
    symbol: String,
    side: String,
    quantity: f64,
    notional: f64,
    entry_price: f64,
    mark_price: f64,
    unrealized_pnl: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (config_path, once, replay, bootstrap_previous_day) = parse_arguments()?;
    let config = load_config(&config_path)?;
    let address: SocketAddr = config.server.bind.parse().context("parse server.bind")?;
    if let Some(replay) = replay {
        let bundle = inspect_bundle(&replay.bundle)?;
        let warmup_bundle = inspect_bundle(&replay.warmup_bundle)?;
        let replay_run =
            run_historical_replay(&config, bundle.clone(), warmup_bundle, replay).await?;
        serve_replay_dashboard(address, bundle, replay_run).await?;
        return Ok(());
    }
    let bundle = inspect_bundle(&config.model.bundle)?;
    if let Some(parent) = config.storage.ledger_path.parent() {
        fs::create_dir_all(parent).context("create ledger directory")?;
    }
    let ledger = Ledger::open(&config.storage.ledger_path)?;
    let last_decision_date = ledger
        .latest_decision_id()?
        .and_then(|id| id.strip_prefix("ranktrend-1d-").map(ToOwned::to_owned));
    let paper_config = paper_config(&config);
    let restored_state = ledger.load_engine_state(PAPER_SESSION_ID)?;
    if bootstrap_previous_day && restored_state.is_some() {
        bail!(
            "--bootstrap-previous-day requires a fresh paper ledger; remove the existing local paper state first"
        );
    }
    let engine = if let Some(state) = restored_state {
        PaperEngine::restore(paper_config, state)?
    } else {
        PaperEngine::new(PAPER_SESSION_ID.to_owned(), paper_config)?
    };
    ledger.record_snapshot(&engine.snapshot(Utc::now()))?;
    ledger.save_engine_state(&engine.persistent_state(), Utc::now())?;
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        ledger: Arc::new(Mutex::new(ledger)),
        instrument_rules: Arc::new(Mutex::new(BTreeMap::new())),
        model: ModelView {
            bundle_id: bundle.manifest.bundle_id,
            backend: bundle.manifest.backend,
            horizon_days: bundle.manifest.horizon_days,
            cutoff_date: bundle.manifest.cutoff_date,
            seed_count: bundle.manifest.models.len(),
        },
        runtime: Arc::new(Mutex::new(RuntimeStatus {
            last_decision_date,
            ..RuntimeStatus::booting()
        })),
    };
    if once {
        run_previous_day_bootstrap(state.clone(), config).await?;
        return Ok(());
    }
    if !bootstrap_previous_day {
        set_status(
            &state,
            "waiting_for_daily_close",
            "Waiting for a confirmed 1D close. Use --bootstrap-previous-day only for a fresh paper session.",
        );
    }
    tokio::spawn(run_live_loop(state.clone(), config, bootstrap_previous_day));
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/api/session", get(session))
        .route("/api/positions", get(positions))
        .route("/api/executions", get(executions))
        .route("/api/equity", get(equity))
        .route("/assets/styles.css", get(styles))
        .route("/assets/app.js", get(app_js))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    println!("RankTrend paper dashboard: http://{address}");
    println!(
        "Paper only: this binary has no private exchange credentials or order-placement code."
    );
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

struct ReplayRequest {
    start: NaiveDate,
    end: NaiveDate,
    bundle: PathBuf,
    warmup_bundle: PathBuf,
    reference: Option<PathBuf>,
}

#[derive(Clone)]
struct ReplayAppState {
    model: ModelView,
    account: AccountSnapshot,
    last_decision_date: String,
    equity: Vec<EquityPoint>,
}

struct ReplayRun {
    account: AccountSnapshot,
    last_decision_date: String,
    equity: Vec<EquityPoint>,
}

fn parse_arguments() -> Result<(PathBuf, bool, Option<ReplayRequest>, bool)> {
    let mut config = PathBuf::from("config/paper.toml");
    let mut once = false;
    let mut replay_start = None;
    let mut replay_end = None;
    let mut replay_bundle = None;
    let mut warmup_bundle = None;
    let mut reference = None;
    let mut bootstrap_previous_day = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--config" => {
                config = PathBuf::from(arguments.next().context("missing --config path")?)
            }
            "--once" => once = true,
            "--bootstrap-previous-day" => bootstrap_previous_day = true,
            "--replay-start" => {
                replay_start = Some(NaiveDate::parse_from_str(
                    &arguments.next().context("missing --replay-start date")?,
                    "%Y-%m-%d",
                )?)
            }
            "--replay-end" => {
                replay_end = Some(NaiveDate::parse_from_str(
                    &arguments.next().context("missing --replay-end date")?,
                    "%Y-%m-%d",
                )?)
            }
            "--replay-bundle" => {
                replay_bundle = Some(PathBuf::from(
                    arguments.next().context("missing --replay-bundle path")?,
                ))
            }
            "--warmup-bundle" => {
                warmup_bundle = Some(PathBuf::from(
                    arguments.next().context("missing --warmup-bundle path")?,
                ))
            }
            "--reference" => {
                reference = Some(PathBuf::from(
                    arguments.next().context("missing --reference path")?,
                ))
            }
            _ => bail!("unknown argument {argument}"),
        }
    }
    let replay = match (replay_start, replay_end, replay_bundle, warmup_bundle) {
        (None, None, None, None) => None,
        (Some(start), Some(end), Some(bundle), Some(warmup_bundle)) if start <= end => {
            Some(ReplayRequest {
                start,
                end,
                bundle,
                warmup_bundle,
                reference,
            })
        }
        _ => bail!(
            "historical replay requires --replay-start YYYY-MM-DD --replay-end YYYY-MM-DD --replay-bundle PATH and --warmup-bundle PATH"
        ),
    };
    if replay.is_some() && once {
        bail!("--once cannot be combined with historical replay")
    }
    if replay.is_some() && bootstrap_previous_day {
        bail!("--bootstrap-previous-day cannot be combined with historical replay")
    }
    Ok((config, once, replay, bootstrap_previous_day))
}

fn paper_config(config: &RuntimeConfig) -> PaperConfig {
    PaperConfig {
        initial_equity_usd: config.paper.initial_equity_usd,
        execution: SnapshotExecutionConfig {
            fee_bps: config.paper.fee_bps,
            reject_partial: config.paper.reject_partial_fills,
        },
        risk: RiskLimits {
            gross_leverage: config.paper.gross_leverage,
            max_gross_leverage: config.paper.max_gross_leverage,
        },
    }
}

async fn run_live_loop(state: AppState, config: RuntimeConfig, bootstrap_previous_day: bool) {
    if bootstrap_previous_day {
        if let Err(error) = run_previous_day_bootstrap(state.clone(), config.clone()).await {
            set_error(&state, error);
        }
    }
    loop {
        let client = BybitPublicClient::default();
        let symbols = match subscription_symbols(&client, &config).await {
            Ok(symbols) => symbols,
            Err(error) => {
                set_error(&state, error);
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        };
        let mut socket = match client.connect_daily_ws().await {
            Ok(socket) => socket,
            Err(error) => {
                set_error(&state, error);
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue;
            }
        };
        if let Err(error) = socket.subscribe(&symbols, config.bybit.ws_batch_size).await {
            set_error(&state, error);
            tokio::time::sleep(Duration::from_secs(10)).await;
            continue;
        }
        set_status(
            &state,
            "waiting_for_daily_close",
            &format!(
                "Subscribed to {} active Bybit 1D contracts; waiting for a complete confirmed close.",
                symbols.len()
            ),
        );
        let mut pending: Option<Candle> = None;
        loop {
            let first = match pending.take() {
                Some(candle) => Ok(candle),
                None => socket.next_confirmed_candle().await,
            };
            match first {
                Ok(first) => match collect_confirmed_daily_batch(
                    &mut socket,
                    first,
                    Duration::from_secs(config.bybit.close_grace_seconds),
                    &mut pending,
                )
                .await
                {
                    Ok(candles) => {
                        let received = candles
                            .iter()
                            .map(|candle| &candle.symbol)
                            .collect::<BTreeSet<_>>();
                        let coverage = received.len() as f64 / symbols.len() as f64;
                        if coverage < config.bybit.minimum_confirmed_coverage {
                            set_error(
                                &state,
                                format!(
                                    "confirmed 1D coverage {:.1}% is below configured minimum {:.1}%",
                                    coverage * 100.0,
                                    config.bybit.minimum_confirmed_coverage * 100.0,
                                ),
                            );
                            continue;
                        }
                        if let Err(error) =
                            handle_confirmed_close(state.clone(), config.clone(), candles).await
                        {
                            set_error(&state, error);
                        }
                    }
                    Err(error) => {
                        set_error(&state, error);
                        break;
                    }
                },
                Err(error) => {
                    set_error(&state, error);
                    break;
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn subscription_symbols(
    client: &BybitPublicClient,
    config: &RuntimeConfig,
) -> Result<Vec<String>> {
    let bundle = inspect_bundle(&config.model.bundle)?;
    let active = client
        .active_linear_symbols()
        .await?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let symbols = bundle
        .universe
        .symbols
        .iter()
        .map(|base| bybit_linear_symbol(base))
        .filter(|symbol| active.contains(symbol))
        .collect::<Vec<_>>();
    if symbols.is_empty() {
        bail!("none of the immutable model-universe contracts are currently tradable on Bybit");
    }
    if !symbols.iter().any(|symbol| symbol == "BTCUSDT") {
        bail!("BTCUSDT is absent from the active immutable model universe");
    }
    Ok(symbols)
}

async fn collect_confirmed_daily_batch(
    socket: &mut rt_bybit::BybitDailyWs,
    first: Candle,
    grace: Duration,
    pending: &mut Option<Candle>,
) -> Result<Vec<Candle>> {
    let date = first.opened_at.date_naive();
    let mut candles = BTreeMap::from([(first.symbol.clone(), first)]);
    let deadline = tokio::time::Instant::now() + grace;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, socket.next_confirmed_candle()).await {
            Ok(Ok(candidate)) if candidate.opened_at.date_naive() == date => {
                candles.insert(candidate.symbol.clone(), candidate);
            }
            Ok(Ok(candidate)) => {
                *pending = Some(candidate);
                break;
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => break,
        }
    }
    candles
        .into_values()
        .map(|mut candle| {
            candle.symbol = base_symbol(&candle.symbol)?;
            Ok(candle)
        })
        .collect()
}

async fn run_previous_day_bootstrap(state: AppState, config: RuntimeConfig) -> Result<()> {
    set_status(
        &state,
        "validating",
        "Bootstrapping one previous confirmed UTC day into a fresh paper session.",
    );
    let bundle = inspect_bundle(&config.model.bundle)?;
    let mut market = load_daily_csv_panel(&config.data.directory)?;
    if market.symbols != bundle.universe.symbols {
        bail!("local data universe differs from the immutable model contract");
    }
    let yesterday = Utc::now()
        .date_naive()
        .pred_opt()
        .context("cannot determine latest confirmed UTC day")?;
    let local_last = market
        .dates
        .last()
        .copied()
        .context("local market panel is empty")?;
    if local_last > yesterday {
        bail!("local history extends beyond the previous confirmed UTC day {yesterday}");
    }
    repair_market_to(&mut market, &bundle, yesterday, Vec::new(), &config).await?;
    decide_for_latest(state, config, bundle, market).await
}

async fn handle_confirmed_close(
    state: AppState,
    config: RuntimeConfig,
    confirmed: Vec<Candle>,
) -> Result<()> {
    let bundle = inspect_bundle(&config.model.bundle)?;
    let mut market = load_daily_csv_panel(&config.data.directory)?;
    if market.symbols != bundle.universe.symbols {
        bail!("local data universe differs from the immutable model contract");
    }
    let date = confirmed
        .first()
        .map(|candle| candle.opened_at.date_naive())
        .context("confirmed daily candle batch is empty")?;
    repair_market_to(&mut market, &bundle, date, confirmed, &config).await?;
    decide_for_latest(state, config, bundle, market).await
}

async fn repair_market_to(
    market: &mut MarketPanel,
    bundle: &BundleMetadata,
    target: NaiveDate,
    supplied: Vec<Candle>,
    config: &RuntimeConfig,
) -> Result<()> {
    let last = market
        .dates
        .last()
        .copied()
        .context("local market panel is empty")?;
    if target <= last {
        return Ok(());
    }
    let start = last.succ_opt().context("daily date overflow")?;
    let client = BybitPublicClient::default();
    let supplied_symbols = supplied
        .iter()
        .map(|candle| candle.symbol.as_str())
        .collect::<BTreeSet<_>>();
    let symbols_to_repair = if !supplied_symbols.is_empty() && start == target {
        bundle
            .universe
            .symbols
            .iter()
            .filter(|symbol| !supplied_symbols.contains(symbol.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    } else {
        bundle.universe.symbols.clone()
    };
    let fetched = fetch_daily_range(
        &client,
        &symbols_to_repair,
        start,
        target,
        config.bybit.rest_parallelism,
    )
    .await;
    let mut by_date = BTreeMap::<NaiveDate, Vec<Candle>>::new();
    for candle in fetched {
        by_date
            .entry(candle.opened_at.date_naive())
            .or_default()
            .push(candle);
    }
    if !supplied.is_empty() {
        let supplied_date = supplied[0].opened_at.date_naive();
        let candles = by_date.entry(supplied_date).or_default();
        candles.retain(|existing| !supplied_symbols.contains(existing.symbol.as_str()));
        candles.extend(supplied);
    }
    let mut date = start;
    while date <= target {
        let candles = by_date.remove(&date).unwrap_or_default();
        if !candles.iter().any(|candle| candle.symbol == "BTC") {
            bail!("cannot repair {date}: confirmed BTC candle is unavailable");
        }
        merge_confirmed_daily_candles(market, &candles)?;
        date = date.succ_opt().context("daily date overflow")?;
    }
    Ok(())
}

async fn fetch_daily_range(
    client: &BybitPublicClient,
    symbols: &[String],
    start: NaiveDate,
    end: NaiveDate,
    parallelism: usize,
) -> Vec<Candle> {
    let start_at = utc_midnight(start);
    let end_at = utc_midnight(end);
    stream::iter(symbols.iter().cloned())
        .map(|base| {
            let client = client.clone();
            async move {
                let exchange_symbol = bybit_linear_symbol(&base);
                match client
                    .daily_klines_range(&exchange_symbol, start_at, end_at)
                    .await
                {
                    Ok(mut candles) => {
                        for candle in &mut candles {
                            candle.symbol = base.clone();
                        }
                        candles
                    }
                    Err(error) => {
                        eprintln!("daily repair skipped {base}: {error:#}");
                        Vec::new()
                    }
                }
            }
        })
        .buffer_unordered(parallelism.max(1))
        .flat_map(stream::iter)
        .collect()
        .await
}

async fn decide_for_latest(
    state: AppState,
    config: RuntimeConfig,
    bundle: BundleMetadata,
    market: MarketPanel,
) -> Result<()> {
    let date = market
        .dates
        .last()
        .copied()
        .context("market panel is empty")?;
    let decision_id = format!("ranktrend-1d-{date}");
    if state.ledger.lock().has_decision(&decision_id)? {
        set_status(
            &state,
            "waiting_for_daily_close",
            "Causal state recovered; latest daily decision is already ledgered.",
        );
        return Ok(());
    }
    set_status(
        &state,
        "calculating",
        "Rebuilding features, ranks, smoothing state, and paper targets.",
    );
    let target = {
        let mut scorer = NativeScorer::load(&bundle)?;
        scorer.target_weights(&market)?
    };
    let equity = state.engine.lock().snapshot(Utc::now()).equity;
    let mut desired = BTreeMap::<String, f64>::new();
    for (base, weight) in market.symbols.iter().zip(target) {
        if weight.abs() > 1e-10 {
            desired.insert(
                bybit_linear_symbol(base),
                weight * equity * config.paper.gross_leverage,
            );
        }
    }
    for position in state.engine.lock().positions() {
        desired.entry(position.symbol).or_insert(0.0);
    }
    let client = BybitPublicClient::default();
    let execution_inputs = fetch_execution_inputs(
        &client,
        desired,
        &state.instrument_rules,
        config.bybit.orderbook_depth,
        config.bybit.rest_parallelism,
    )
    .await;
    for (symbol, notional, rules, book) in execution_inputs {
        if book.mid_price().is_none() {
            eprintln!("skip {symbol}: order-book snapshot is not two-sided");
            continue;
        }
        let report = state.engine.lock().rebalance_to_notional(
            &decision_id,
            &symbol,
            notional,
            &rules,
            &book,
        )?;
        if let Some(report) = report {
            state.ledger.lock().record_execution(&report, &book)?;
        }
    }
    let now = Utc::now();
    let engine = state.engine.lock();
    let snapshot = engine.snapshot(now);
    let persisted = engine.persistent_state();
    drop(engine);
    let ledger = state.ledger.lock();
    ledger.record_snapshot(&snapshot)?;
    ledger.save_engine_state(&persisted, now)?;
    ledger.record_decision(&decision_id, now)?;
    drop(ledger);
    let mut runtime = state.runtime.lock();
    runtime.status = "waiting_for_daily_close".to_owned();
    runtime.detail =
        "Latest 1D decision was simulated from order-book snapshots and written to SQLite."
            .to_owned();
    runtime.last_decision_date = Some(date.to_string());
    runtime.last_error = None;
    Ok(())
}

async fn fetch_execution_inputs(
    client: &BybitPublicClient,
    desired: BTreeMap<String, f64>,
    rules_cache: &Arc<Mutex<BTreeMap<String, InstrumentRules>>>,
    orderbook_depth: u16,
    parallelism: usize,
) -> Vec<(String, f64, InstrumentRules, OrderBookSnapshot)> {
    let inputs = stream::iter(desired)
        .map(|(symbol, notional)| {
            let client = client.clone();
            let rules_cache = Arc::clone(rules_cache);
            async move {
                let cached_rules = { rules_cache.lock().get(&symbol).cloned() };
                let rules = match cached_rules {
                    Some(rules) => rules,
                    None => match client.instrument_rules(&symbol).await {
                        Ok(rules) => {
                            rules_cache.lock().insert(symbol.clone(), rules.clone());
                            rules
                        }
                        Err(error) => {
                            eprintln!(
                                "skip {symbol}: instrument constraints unavailable: {error:#}"
                            );
                            return None;
                        }
                    },
                };
                let book = match client.orderbook(&symbol, orderbook_depth).await {
                    Ok(book) => book,
                    Err(error) => {
                        eprintln!("skip {symbol}: order-book snapshot unavailable: {error:#}");
                        return None;
                    }
                };
                Some((symbol, notional, rules, book))
            }
        })
        .buffer_unordered(parallelism.max(1))
        .filter_map(|input| async move { input })
        .collect::<Vec<_>>()
        .await;
    let mut inputs = inputs;
    inputs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    inputs
}

async fn run_historical_replay(
    config: &RuntimeConfig,
    bundle: BundleMetadata,
    warmup_bundle: BundleMetadata,
    replay: ReplayRequest,
) -> Result<ReplayRun> {
    if bundle.universe.symbols != warmup_bundle.universe.symbols
        || bundle.features.names != warmup_bundle.features.names
    {
        bail!("replay and warm-up bundles have different immutable contracts");
    }
    let market = load_daily_csv_panel(&config.data.directory)?;
    if market.symbols != bundle.universe.symbols {
        bail!("local data universe differs from the replay model contract");
    }
    let panel = build_features(market, 180, 150)?;
    let start = panel
        .market
        .dates
        .iter()
        .position(|date| *date == replay.start)
        .with_context(|| {
            format!(
                "replay start {} is absent from the market panel",
                replay.start
            )
        })?;
    let end = panel
        .market
        .dates
        .iter()
        .position(|date| *date == replay.end)
        .with_context(|| format!("replay end {} is absent from the market panel", replay.end))?;
    if start < 22 || end >= panel.market.dates.len().saturating_sub(1) {
        bail!("replay needs 21 preceding decision days and one following market open");
    }
    let mut scorer = NativeScorer::load(&bundle)?;
    let mut warmup_scorer = NativeScorer::load(&warmup_bundle)?;
    let mut prior_signals = Vec::with_capacity(21);
    for decision_row in start - 22..start - 1 {
        let source = if panel.market.dates[decision_row] < replay.start {
            &mut warmup_scorer
        } else {
            &mut scorer
        };
        prior_signals.push(source.raw_weights(&panel, decision_row)?);
    }
    let mut prior = smooth_fixed_window(&prior_signals, 21)?;
    let prior_volatility = panel.btc_vol20[..start]
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    apply_volatility_overlay(&mut prior, &prior_volatility, OverlayRules::default())?;
    let mut equity = config.paper.initial_equity_usd;
    let mut equity_points = Vec::with_capacity(end - start + 1);
    let output = config
        .storage
        .ledger_path
        .parent()
        .context("ledger path has no parent directory")?
        .join("historical_replay.csv");
    let mut writer = csv::Writer::from_path(&output)
        .with_context(|| format!("create replay output {}", output.display()))?;
    writer.write_record(["date", "net_return", "turnover", "equity"])?;
    let one_way_cost = config.paper.fee_bps / 10_000.0;
    for row in start..=end {
        let mut signals = Vec::with_capacity(21);
        for decision_row in row - 21..row {
            let source = if panel.market.dates[decision_row] < replay.start {
                &mut warmup_scorer
            } else {
                &mut scorer
            };
            signals.push(source.raw_weights(&panel, decision_row)?);
        }
        let mut weights = smooth_fixed_window(&signals, 21)?;
        let volatility = panel.btc_vol20[..=row]
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        apply_volatility_overlay(&mut weights, &volatility, OverlayRules::default())?;
        let turnover = weights
            .iter()
            .zip(&prior)
            .map(|(weight, old)| (weight - old).abs())
            .sum::<f64>();
        let gross_return = weights
            .iter()
            .zip(panel.oo_return.row(row))
            .filter_map(|(weight, value)| value.is_finite().then_some(weight * f64::from(*value)))
            .sum::<f64>();
        let net_return = gross_return - one_way_cost * turnover;
        equity *= 1.0 + net_return;
        writer.serialize((panel.market.dates[row], net_return, turnover, equity))?;
        equity_points.push(EquityPoint {
            captured_at: utc_midnight(panel.market.dates[row]),
            equity,
            gross_notional: weights.iter().map(|weight| weight.abs()).sum::<f64>() * equity,
            net_notional: weights.iter().sum::<f64>() * equity,
        });
        prior = weights;
    }
    writer.flush()?;
    println!(
        "historical replay {}..{} complete: equity=${equity:.6}; rows={}; output={}",
        replay.start,
        replay.end,
        end - start + 1,
        output.display(),
    );
    if let Some(reference) = replay.reference {
        let (max_return_diff, max_turnover_diff) = compare_reference(&reference, &output)?;
        println!(
            "parity check: max_return_diff={max_return_diff:.3e}; max_turnover_diff={max_turnover_diff:.3e}"
        );
        if max_return_diff > 1e-6 || max_turnover_diff > 1e-6 {
            bail!("historical replay differs from the Python reference beyond 1e-6")
        }
        println!("parity check: PASS");
    }
    Ok(ReplayRun {
        account: AccountSnapshot {
            session_id: "ranktrend-historical-replay".to_owned(),
            captured_at: utc_midnight(replay.end),
            cash: equity,
            equity,
            realized_pnl: equity - config.paper.initial_equity_usd,
            unrealized_pnl: 0.0,
            gross_notional: 0.0,
            net_notional: 0.0,
            fee_paid: 0.0,
        },
        last_decision_date: replay.end.to_string(),
        equity: equity_points,
    })
}

async fn serve_replay_dashboard(
    address: SocketAddr,
    bundle: BundleMetadata,
    replay: ReplayRun,
) -> Result<()> {
    let state = ReplayAppState {
        model: ModelView {
            bundle_id: bundle.manifest.bundle_id,
            backend: bundle.manifest.backend,
            horizon_days: bundle.manifest.horizon_days,
            cutoff_date: bundle.manifest.cutoff_date,
            seed_count: bundle.manifest.models.len(),
        },
        account: replay.account,
        last_decision_date: replay.last_decision_date,
        equity: replay.equity,
    };
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/api/session", get(replay_session))
        .route("/api/positions", get(replay_positions))
        .route("/api/executions", get(replay_executions))
        .route("/api/equity", get(replay_equity))
        .route("/assets/styles.css", get(styles))
        .route("/assets/app.js", get(app_js))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    println!("RankTrend historical replay dashboard: http://{address}");
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ReplayCsvRow {
    date: NaiveDate,
    net_return: f64,
    turnover: f64,
}

fn compare_reference(reference: &PathBuf, output: &PathBuf) -> Result<(f64, f64)> {
    let mut expected = BTreeMap::new();
    for row in csv::Reader::from_path(reference)
        .with_context(|| format!("open Python reference {}", reference.display()))?
        .deserialize::<ReplayCsvRow>()
    {
        let row =
            row.with_context(|| format!("decode Python reference {}", reference.display()))?;
        expected.insert(row.date, row);
    }
    let mut max_return_diff = 0.0_f64;
    let mut max_turnover_diff = 0.0_f64;
    for row in csv::Reader::from_path(output)
        .with_context(|| format!("open Rust replay output {}", output.display()))?
        .deserialize::<ReplayCsvRow>()
    {
        let row = row.with_context(|| format!("decode Rust replay output {}", output.display()))?;
        let expected = expected
            .remove(&row.date)
            .with_context(|| format!("Python reference has no row for {}", row.date))?;
        max_return_diff = max_return_diff.max((row.net_return - expected.net_return).abs());
        max_turnover_diff = max_turnover_diff.max((row.turnover - expected.turnover).abs());
    }
    Ok((max_return_diff, max_turnover_diff))
}

struct NativeScorer {
    _library: LightGbmLibrary,
    boosters: Vec<NativeBooster>,
}

impl NativeScorer {
    fn load(bundle: &BundleMetadata) -> Result<Self> {
        let library = LightGbmLibrary::linked();
        let boosters = bundle
            .manifest
            .models
            .iter()
            .map(|model| library.load_booster(bundle.root.join(&model.file)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            _library: library,
            boosters,
        })
    }

    fn target_weights(&mut self, market: &MarketPanel) -> Result<Vec<f64>> {
        let panel = build_features(market.clone(), 180, 150)?;
        let final_row = panel
            .market
            .dates
            .len()
            .checked_sub(1)
            .context("empty feature panel")?;
        let first_row = final_row.saturating_sub(20);
        let mut signals = Vec::new();
        for row in first_row..=final_row {
            signals.push(self.raw_weights(&panel, row)?);
        }
        let mut weights = smooth_fixed_window(&signals, 21)?;
        let volatility = panel.btc_vol20[..=final_row]
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        apply_volatility_overlay(&mut weights, &volatility, OverlayRules::default())?;
        Ok(weights)
    }

    fn raw_weights(&mut self, panel: &FeaturePanel, row: usize) -> Result<Vec<f64>> {
        let width = panel.market.symbols.len();
        let mut matrix = Vec::new();
        let mut columns = Vec::new();
        let mut eligible = vec![false; width];
        for column in 0..width {
            let index = row * width + column;
            eligible[column] = panel.eligible_max[index];
            if !eligible[column] || !panel.valid_features[index] {
                continue;
            }
            for feature in &panel.feature_stack {
                matrix.push(feature.get(row, column));
            }
            columns.push(column);
        }
        let mut seed_scores = vec![vec![f64::NAN; width]; self.boosters.len()];
        for (seed, booster) in self.boosters.iter_mut().enumerate() {
            let scores = booster.predict_rows(&matrix, panel.feature_names.len())?;
            for (column, score) in columns.iter().zip(scores) {
                seed_scores[seed][*column] = score;
            }
        }
        let scores = median_rank_ensemble(&seed_scores, &eligible)?;
        build_daily_weights(
            &scores,
            &eligible,
            &panel
                .liquidity_rank
                .row(row)
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
            &panel
                .beta60
                .row(row)
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
            &panel
                .vol20
                .row(row)
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
            &PortfolioRules::default(),
        )
        .map_err(Into::into)
    }
}

fn utc_midnight(date: NaiveDate) -> DateTime<Utc> {
    date.and_hms_opt(0, 0, 0)
        .expect("midnight exists")
        .and_utc()
}

fn load_config(path: &PathBuf) -> Result<RuntimeConfig> {
    let source = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&source).with_context(|| format!("parse {}", path.display()))
}

fn set_status(state: &AppState, status: &str, detail: &str) {
    let mut runtime = state.runtime.lock();
    runtime.status = status.to_owned();
    runtime.detail = detail.to_owned();
}

fn set_error(state: &AppState, error: impl std::fmt::Display) {
    let mut runtime = state.runtime.lock();
    runtime.status = "degraded".to_owned();
    runtime.detail =
        "The paper loop is paused until a safe data or network retry succeeds.".to_owned();
    runtime.last_error = Some(error.to_string());
    eprintln!(
        "RankTrend paper loop: {}",
        runtime.last_error.as_deref().unwrap_or_default()
    );
}

async fn health() -> &'static str {
    "ok"
}

async fn session(State(state): State<AppState>) -> Json<SessionView> {
    let engine = state.engine.lock();
    let runtime = state.runtime.lock().clone();
    Json(SessionView {
        status: runtime.status,
        detail: runtime.detail,
        model: state.model.clone(),
        account: engine.snapshot(Utc::now()),
        last_decision_date: runtime.last_decision_date,
        last_error: runtime.last_error,
    })
}

async fn positions(State(state): State<AppState>) -> Json<Vec<PositionView>> {
    Json(
        state
            .engine
            .lock()
            .positions()
            .into_iter()
            .map(|position| {
                let unrealized_pnl = position.unrealized_pnl();
                let notional = position.notional();
                let side = if position.quantity >= 0.0 {
                    "long".to_owned()
                } else {
                    "short".to_owned()
                };
                PositionView {
                    symbol: position.symbol,
                    side,
                    quantity: position.quantity,
                    notional,
                    entry_price: position.entry_vwap,
                    mark_price: position.mark_price,
                    unrealized_pnl,
                }
            })
            .collect(),
    )
}

async fn executions(State(state): State<AppState>) -> impl IntoResponse {
    match state.ledger.lock().recent_executions(100) {
        Ok(reports) => Json(reports).into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("read executions: {error}"),
        )
            .into_response(),
    }
}

async fn equity(State(state): State<AppState>) -> impl IntoResponse {
    match state.ledger.lock().equity_curve(PAPER_SESSION_ID, 2_000) {
        Ok(points) => Json(points).into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("read equity curve: {error}"),
        )
            .into_response(),
    }
}

async fn replay_session(State(state): State<ReplayAppState>) -> Json<SessionView> {
    Json(SessionView {
        status: "historical_replay".to_owned(),
        detail: "Deterministic 1D OOS replay; no network data, order book, or slippage.".to_owned(),
        model: state.model.clone(),
        account: state.account.clone(),
        last_decision_date: Some(state.last_decision_date.clone()),
        last_error: None,
    })
}

async fn replay_positions() -> Json<Vec<PositionView>> {
    Json(Vec::new())
}

async fn replay_executions() -> Json<Vec<rt_domain::ExecutionReport>> {
    Json(Vec::new())
}

async fn replay_equity(State(state): State<ReplayAppState>) -> Json<Vec<EquityPoint>> {
    Json(state.equity.clone())
}

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("../../ui/index.html"))
}

async fn styles() -> impl IntoResponse {
    (
        [("content-type", "text/css; charset=utf-8")],
        include_str!("../../ui/styles.css"),
    )
}

async fn app_js() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("../../ui/app.js"),
    )
}
