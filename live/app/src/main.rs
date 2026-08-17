use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{Query, State},
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
use rt_ledger::{
    DailyPeriodReturn, EquityBucket, EquityCurve, EquityPoint, Ledger, PerformanceMetrics,
    summarize_periods,
};
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

const PAPER_SESSION_ID: &str = "airok-paper-v1";

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
    metrics_ledger: Arc<Mutex<Ledger>>,
    executions_ledger: Arc<Mutex<Ledger>>,
    curve_ledger: Arc<Mutex<Ledger>>,
    session_start_equity_usd: f64,
    instrument_rules: Arc<Mutex<InstrumentRuleCache>>,
    model: ModelView,
    runtime: Arc<Mutex<RuntimeStatus>>,
}

#[derive(Default)]
struct InstrumentRuleCache {
    refreshed_for: Option<NaiveDate>,
    rules: BTreeMap<String, InstrumentRules>,
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
    session_start_equity_usd: f64,
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
    opened_at: chrono::DateTime<Utc>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (config_path, once, replay, no_bootstrap, ledger_replay) = parse_arguments()?;
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
        .and_then(|id| id.strip_prefix("airok-1d-").map(ToOwned::to_owned));
    let paper_config = paper_config(&config);
    let restored_state = ledger.load_engine_state(PAPER_SESSION_ID)?;
    let bootstrap_previous_day =
        should_bootstrap_previous_day(restored_state.is_none(), no_bootstrap);
    let engine = if let Some(state) = restored_state {
        PaperEngine::restore(paper_config, state)?
    } else {
        if ledger_replay {
            bail!("ledger replay requires an existing paper state");
        }
        PaperEngine::new(PAPER_SESSION_ID.to_owned(), paper_config)?
    };
    if !ledger_replay {
        ledger.record_snapshot(&engine.snapshot(Utc::now()))?;
        ledger.save_engine_state(&engine.persistent_state(), Utc::now())?;
    }
    let session_start_equity_usd = ledger
        .session_start_equity(PAPER_SESSION_ID)?
        .context("session has no account snapshot")?;
    // Keep independent read connections so slow history/metrics queries never queue behind
    // the writer connection or behind one another. SQLite WAL allows these readers to run concurrently.
    let metrics_ledger = Ledger::open_reader(&config.storage.ledger_path)?;
    let executions_ledger = Ledger::open_reader(&config.storage.ledger_path)?;
    let curve_ledger = Ledger::open_reader(&config.storage.ledger_path)?;
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        ledger: Arc::new(Mutex::new(ledger)),
        metrics_ledger: Arc::new(Mutex::new(metrics_ledger)),
        executions_ledger: Arc::new(Mutex::new(executions_ledger)),
        curve_ledger: Arc::new(Mutex::new(curve_ledger)),
        session_start_equity_usd,
        instrument_rules: Arc::new(Mutex::new(InstrumentRuleCache::default())),
        model: ModelView {
            bundle_id: bundle.manifest.bundle_id,
            backend: bundle.manifest.backend,
            horizon_days: bundle.manifest.horizon_days,
            cutoff_date: bundle.manifest.cutoff_date,
            seed_count: bundle.manifest.models.len(),
        },
        runtime: Arc::new(Mutex::new(if ledger_replay {
            RuntimeStatus {
                status: "ledger_replay".to_owned(),
                detail: "Read-only replay from a frozen paper ledger snapshot.".to_owned(),
                last_decision_date,
                last_error: None,
            }
        } else {
            RuntimeStatus {
                last_decision_date,
                ..RuntimeStatus::booting()
            }
        })),
    };
    if once {
        run_previous_day_bootstrap(state.clone(), config).await?;
        return Ok(());
    }
    if !bootstrap_previous_day && !ledger_replay {
        set_status(
            &state,
            "waiting_for_daily_close",
            "Waiting for a confirmed 1D close.",
        );
    }
    if !ledger_replay {
        tokio::spawn(run_live_loop(
            state.clone(),
            config.clone(),
            bootstrap_previous_day,
        ));
        tokio::spawn(run_open_position_mark_loop(state.clone(), config));
    }
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/api/session", get(session))
        .route("/api/positions", get(positions))
        .route("/api/executions", get(executions))
        .route("/api/metrics", get(metrics))
        .route("/api/equity", get(equity))
        .route("/assets/styles.css", get(styles))
        .route("/assets/sort.css", get(sort_styles))
        .route("/assets/app.js", get(app_js))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    println!("airōk paper dashboard: http://{address}");
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
    periods: Vec<DailyPeriodReturn>,
}

struct ReplayRun {
    account: AccountSnapshot,
    last_decision_date: String,
    equity: Vec<EquityPoint>,
    periods: Vec<DailyPeriodReturn>,
}

fn parse_arguments() -> Result<(PathBuf, bool, Option<ReplayRequest>, bool, bool)> {
    let mut config = PathBuf::from("config/paper.toml");
    let mut once = false;
    let mut replay_start = None;
    let mut replay_end = None;
    let mut replay_bundle = None;
    let mut warmup_bundle = None;
    let mut reference = None;
    let mut no_bootstrap = false;
    let mut ledger_replay = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--config" => {
                config = PathBuf::from(arguments.next().context("missing --config path")?)
            }
            "--once" => once = true,
            "--no-bootstrap" => no_bootstrap = true,
            "--ledger-replay" => ledger_replay = true,
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
    if once && no_bootstrap {
        bail!("--once cannot be combined with --no-bootstrap")
    }
    if replay.is_some() && no_bootstrap {
        bail!("--no-bootstrap cannot be combined with historical replay")
    }
    if ledger_replay && (once || replay.is_some() || no_bootstrap) {
        bail!(
            "--ledger-replay cannot be combined with --once, historical replay, or --no-bootstrap"
        )
    }
    Ok((config, once, replay, no_bootstrap, ledger_replay))
}

fn should_bootstrap_previous_day(fresh_paper_ledger: bool, no_bootstrap: bool) -> bool {
    fresh_paper_ledger && !no_bootstrap
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
    if bootstrap_previous_day
        && let Err(error) = run_previous_day_bootstrap(state.clone(), config.clone()).await
    {
        set_error(&state, error);
    }
    loop {
        let client = BybitPublicClient::default();
        if let Err(error) = refresh_instrument_rules(&state, &client).await {
            set_error(&state, error);
            tokio::time::sleep(Duration::from_secs(10)).await;
            continue;
        }
        let symbols = match subscription_symbols(&state, &config) {
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

/// Keep one-minute Bybit candle subscriptions open only for the currently held paper book.
async fn run_open_position_mark_loop(state: AppState, config: RuntimeConfig) {
    loop {
        let symbols = open_position_symbols(&state);
        if symbols.is_empty() {
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }
        let client = BybitPublicClient::default();
        let mut socket = match client.connect_minute_ws().await {
            Ok(socket) => socket,
            Err(error) => {
                eprintln!("open-position minute mark WebSocket connect failed: {error:#}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        if let Err(error) = socket.subscribe(&symbols, config.bybit.ws_batch_size).await {
            eprintln!("open-position minute mark WebSocket subscription failed: {error:#}");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }
        let mut membership_check = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                candle = socket.next_confirmed_candle() => match candle {
                    Ok(candle) => {
                        if let Err(error) = record_open_position_mark(&state, candle).await {
                            eprintln!("open-position minute mark update failed: {error:#}");
                        }
                    }
                    Err(error) => {
                        eprintln!("open-position minute mark WebSocket failed: {error:#}");
                        break;
                    }
                },
                _ = membership_check.tick() => {
                    if open_position_symbols(&state) != symbols {
                        break;
                    }
                }
            }
        }
    }
}

fn open_position_symbols(state: &AppState) -> Vec<String> {
    state
        .engine
        .lock()
        .positions()
        .into_iter()
        .map(|position| position.symbol)
        .collect()
}

async fn record_open_position_mark(state: &AppState, candle: Candle) -> Result<()> {
    let now = Utc::now();
    let (snapshot, persisted) = {
        let mut engine = state.engine.lock();
        if !engine
            .positions()
            .iter()
            .any(|position| position.symbol == candle.symbol)
        {
            return Ok(());
        }
        engine.mark(&candle.symbol, candle.close, now);
        (engine.snapshot(now), engine.persistent_state())
    };
    let ledger = state.ledger.lock();
    ledger.record_equity_point(&snapshot)?;
    ledger.save_engine_state(&persisted, now)?;
    Ok(())
}

fn subscription_symbols(state: &AppState, config: &RuntimeConfig) -> Result<Vec<String>> {
    let bundle = inspect_bundle(&config.model.bundle)?;
    let rules = state.instrument_rules.lock();
    let symbols = bundle
        .universe
        .symbols
        .iter()
        .map(|base| bybit_linear_symbol(base))
        .filter(|symbol| {
            rules
                .rules
                .get(symbol)
                .is_some_and(InstrumentRules::is_tradable_linear_perpetual)
        })
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
    let bootstrap_closes =
        repair_market_to(&mut market, &bundle, yesterday, Vec::new(), &config).await?;
    decide_for_latest(state, config, bundle, market, Some(bootstrap_closes)).await
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
    let closes = repair_market_to(&mut market, &bundle, date, confirmed, &config).await?;
    record_confirmed_close_mark(&state, &closes)?;
    decide_for_latest(state, config, bundle, market, None).await
}

fn record_confirmed_close_mark(state: &AppState, closes: &BTreeMap<String, f64>) -> Result<()> {
    let now = Utc::now();
    let (snapshot, persisted) = {
        let mut engine = state.engine.lock();
        for position in engine.positions() {
            let base = base_symbol(&position.symbol)?;
            let close = closes
                .get(&base)
                .with_context(|| format!("daily close is unavailable for held position {base}"))?;
            engine.mark(&position.symbol, *close, now);
        }
        (engine.snapshot(now), engine.persistent_state())
    };
    let ledger = state.ledger.lock();
    ledger.record_snapshot(&snapshot)?;
    ledger.save_engine_state(&persisted, now)?;
    Ok(())
}

async fn repair_market_to(
    market: &mut MarketPanel,
    bundle: &BundleMetadata,
    target: NaiveDate,
    supplied: Vec<Candle>,
    config: &RuntimeConfig,
) -> Result<BTreeMap<String, f64>> {
    let last = market
        .dates
        .last()
        .copied()
        .context("local market panel is empty")?;
    if target <= last {
        return Ok(closes_for_market_date(market, target));
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
    let mut target_closes = BTreeMap::new();
    while date <= target {
        let candles = by_date.remove(&date).unwrap_or_default();
        if !candles.iter().any(|candle| candle.symbol == "BTC") {
            bail!("cannot repair {date}: confirmed BTC candle is unavailable");
        }
        if date == target {
            target_closes = candles
                .iter()
                .filter_map(|candle| {
                    (candle.close.is_finite() && candle.close > 0.0)
                        .then_some((candle.symbol.clone(), candle.close))
                })
                .collect();
        }
        merge_confirmed_daily_candles(market, &candles)?;
        date = date.succ_opt().context("daily date overflow")?;
    }
    Ok(target_closes)
}

fn closes_for_market_date(market: &MarketPanel, date: NaiveDate) -> BTreeMap<String, f64> {
    let Some(row) = market.dates.iter().position(|candidate| *candidate == date) else {
        return BTreeMap::new();
    };
    market
        .symbols
        .iter()
        .enumerate()
        .filter_map(|(column, symbol)| {
            let close = f64::from(market.close.get(row, column));
            (close.is_finite() && close > 0.0).then_some((symbol.clone(), close))
        })
        .collect()
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
    bootstrap_closes: Option<BTreeMap<String, f64>>,
) -> Result<()> {
    let date = market
        .dates
        .last()
        .copied()
        .context("market panel is empty")?;
    let decision_id = format!("airok-1d-{date}");
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
    let client = BybitPublicClient::default();
    refresh_instrument_rules(&state, &client).await?;
    let equity = state.engine.lock().snapshot(Utc::now()).equity;
    let mut desired = BTreeMap::<String, f64>::new();
    for (base, weight) in market.symbols.iter().zip(target) {
        if weight.abs() > 1e-10 {
            let symbol = bybit_linear_symbol(base);
            if is_tradable(&state.instrument_rules, &symbol) {
                desired.insert(symbol, weight * equity * config.paper.gross_leverage);
            } else {
                eprintln!("skip {symbol}: unavailable or not a Trading Bybit linear perpetual");
            }
        }
    }
    for position in state.engine.lock().positions() {
        if is_tradable(&state.instrument_rules, &position.symbol) {
            desired.entry(position.symbol).or_insert(0.0);
        } else {
            eprintln!(
                "cannot reduce {}: current Bybit instrument rules are unavailable",
                position.symbol
            );
        }
    }
    if let Some(closes) = bootstrap_closes.as_ref() {
        for symbol in desired.keys() {
            let base = base_symbol(symbol)?;
            if !closes.contains_key(&base) {
                bail!("bootstrap close is unavailable for {base}");
            }
        }
    }
    let execution_inputs = fetch_execution_inputs(
        &client,
        desired,
        &state.instrument_rules,
        config.bybit.orderbook_depth,
        config.bybit.rest_parallelism,
    )
    .await;
    let bootstrap_closed_at = bootstrap_closes
        .is_some()
        .then(|| {
            date.succ_opt()
                .map(utc_midnight)
                .context("bootstrap candle close timestamp overflows")
        })
        .transpose()?;
    for (symbol, notional, rules, book) in execution_inputs {
        if book.mid_price().is_none() {
            eprintln!("skip {symbol}: order-book snapshot is not two-sided");
            continue;
        }
        let report = if let Some(closes) = bootstrap_closes.as_ref() {
            let base = base_symbol(&symbol)?;
            let close = *closes
                .get(&base)
                .with_context(|| format!("bootstrap close is unavailable for {base}"))?;
            state.engine.lock().bootstrap_to_notional(
                &decision_id,
                &symbol,
                notional,
                close,
                bootstrap_closed_at.expect("bootstrap close time is set"),
                &rules,
                &book,
            )?
        } else {
            state.engine.lock().rebalance_to_notional(
                &decision_id,
                &symbol,
                notional,
                &rules,
                &book,
            )?
        };
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

async fn refresh_instrument_rules(state: &AppState, client: &BybitPublicClient) -> Result<()> {
    let today = Utc::now().date_naive();
    if state.instrument_rules.lock().refreshed_for == Some(today) {
        return Ok(());
    }
    let rules = client.linear_instrument_rules().await?;
    if rules.is_empty() {
        bail!("Bybit returned an empty linear instrument rule snapshot");
    }
    *state.instrument_rules.lock() = InstrumentRuleCache {
        refreshed_for: Some(today),
        rules,
    };
    Ok(())
}

fn is_tradable(rules: &Arc<Mutex<InstrumentRuleCache>>, symbol: &str) -> bool {
    rules
        .lock()
        .rules
        .get(symbol)
        .is_some_and(InstrumentRules::is_tradable_linear_perpetual)
}

async fn fetch_execution_inputs(
    client: &BybitPublicClient,
    desired: BTreeMap<String, f64>,
    rules_cache: &Arc<Mutex<InstrumentRuleCache>>,
    orderbook_depth: u16,
    parallelism: usize,
) -> Vec<(String, f64, InstrumentRules, OrderBookSnapshot)> {
    let inputs = stream::iter(desired)
        .map(|(symbol, notional)| {
            let client = client.clone();
            let rules_cache = Arc::clone(rules_cache);
            async move {
                let rules = match rules_cache.lock().rules.get(&symbol).cloned() {
                    Some(rules) if rules.is_tradable_linear_perpetual() => rules,
                    _ => {
                        eprintln!("skip {symbol}: instrument is unavailable or not tradable");
                        return None;
                    }
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
    let mut periods = Vec::with_capacity(end - start + 1);
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
        let start_equity = equity;
        equity *= 1.0 + net_return;
        writer.serialize((panel.market.dates[row], net_return, turnover, equity))?;
        periods.push(DailyPeriodReturn {
            period_date: panel.market.dates[row].to_string(),
            start_equity,
            end_equity: equity,
            net_return,
            pnl: equity - start_equity,
        });
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
            session_id: "airok-historical-replay".to_owned(),
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
        periods,
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
        periods: replay.periods,
    };
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/api/session", get(replay_session))
        .route("/api/positions", get(replay_positions))
        .route("/api/executions", get(replay_executions))
        .route("/api/metrics", get(replay_metrics))
        .route("/api/equity", get(replay_equity))
        .route("/assets/styles.css", get(styles))
        .route("/assets/sort.css", get(sort_styles))
        .route("/assets/app.js", get(app_js))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    println!("airōk historical replay dashboard: http://{address}");
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
        "airōk paper loop: {}",
        runtime.last_error.as_deref().unwrap_or_default()
    );
}

async fn health() -> &'static str {
    "ok"
}

async fn session(State(state): State<AppState>) -> Json<SessionView> {
    // Do not touch SQLite on the hot dashboard path: session start equity is immutable
    // for the lifetime of a paper session and is cached when the process starts.
    let account = state.engine.lock().snapshot(Utc::now());
    let runtime = state.runtime.lock().clone();
    Json(SessionView {
        status: runtime.status,
        detail: runtime.detail,
        model: state.model.clone(),
        account,
        session_start_equity_usd: state.session_start_equity_usd,
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
                    opened_at: position.opened_at,
                }
            })
            .collect(),
    )
}

async fn executions(State(state): State<AppState>) -> impl IntoResponse {
    match state.executions_ledger.lock().recent_executions(100) {
        Ok(reports) => Json(reports).into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("read executions: {error}"),
        )
            .into_response(),
    }
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    match state
        .metrics_ledger
        .lock()
        .performance_metrics(PAPER_SESSION_ID)
    {
        Ok(metrics) => Json(metrics).into_response(),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("read performance metrics: {error}"),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct EquityQuery {
    max_points: Option<usize>,
}

async fn equity(
    Query(query): Query<EquityQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let max_points = query.max_points.unwrap_or(2_000).clamp(200, 5_000);
    match state
        .curve_ledger
        .lock()
        .equity_curve(PAPER_SESSION_ID, max_points)
    {
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
        session_start_equity_usd: state.account.equity - state.account.realized_pnl,
        last_decision_date: Some(state.last_decision_date.clone()),
        last_error: None,
    })
}

fn max_drawdown(points: &[EquityPoint]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut maximum = 0.0_f64;
    for point in points {
        peak = peak.max(point.equity);
        if peak > 0.0 {
            maximum = maximum.min(point.equity / peak - 1.0);
        }
    }
    maximum
}

async fn replay_positions() -> Json<Vec<PositionView>> {
    Json(Vec::new())
}

async fn replay_executions() -> Json<Vec<rt_domain::ExecutionReport>> {
    Json(Vec::new())
}

fn replay_performance_metrics(state: &ReplayAppState) -> PerformanceMetrics {
    let period_metrics = summarize_periods(&state.periods);
    let returns = state
        .periods
        .iter()
        .map(|period| period.net_return)
        .collect::<Vec<_>>();
    let sharpe = period_metrics.average_daily_return.and_then(|mean| {
        (returns.len() > 1)
            .then(|| {
                let variance = returns
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (returns.len() - 1) as f64;
                (variance > 0.0).then(|| mean / variance.sqrt() * 365.0_f64.sqrt())
            })
            .flatten()
    });
    PerformanceMetrics {
        max_drawdown: max_drawdown(&state.equity),
        sharpe,
        average_daily_return: period_metrics.average_daily_return,
        profit_factor: period_metrics.profit_factor,
        profit_factor_unbounded: period_metrics.profit_factor_unbounded,
        win_rate: period_metrics.win_rate,
        period_count: period_metrics.period_count,
        closed_trades: 0,
    }
}

async fn replay_metrics(State(state): State<ReplayAppState>) -> Json<PerformanceMetrics> {
    Json(replay_performance_metrics(&state))
}

async fn replay_equity(State(state): State<ReplayAppState>) -> Json<EquityCurve> {
    let metrics = replay_performance_metrics(&state);
    let mut peak = f64::NEG_INFINITY;
    let points = state
        .equity
        .iter()
        .map(|point| {
            peak = peak.max(point.equity);
            let drawdown = if peak > 0.0 {
                point.equity / peak - 1.0
            } else {
                0.0
            };
            EquityBucket {
                captured_at: point.captured_at,
                equity: point.equity,
                low: point.equity,
                high: point.equity,
                drawdown,
                drawdown_at: point.captured_at,
            }
        })
        .collect::<Vec<_>>();
    Json(EquityCurve {
        total_points: state.equity.len(),
        points,
        periods: state.periods,
        metrics,
    })
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

async fn sort_styles() -> impl IntoResponse {
    (
        [("content-type", "text/css; charset=utf-8")],
        include_str!("../../ui/sort.css"),
    )
}

async fn app_js() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("../../ui/app.js"),
    )
}

#[cfg(test)]
mod tests {
    use super::should_bootstrap_previous_day;

    #[test]
    fn bootstrap_is_default_only_for_a_fresh_paper_ledger() {
        assert!(should_bootstrap_previous_day(true, false));
        assert!(!should_bootstrap_previous_day(true, true));
        assert!(!should_bootstrap_previous_day(false, false));
        assert!(!should_bootstrap_previous_day(false, true));
    }
}
