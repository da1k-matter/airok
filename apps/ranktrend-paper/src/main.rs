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
use rt_domain::{AccountSnapshot, Candle};
use rt_engine::{PaperConfig, PaperEngine, RiskLimits};
use rt_execution::SnapshotExecutionConfig;
use rt_ledger::Ledger;
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
    model_bundle: Arc<str>,
    runtime: Arc<Mutex<RuntimeStatus>>,
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
    model_bundle: String,
    account: AccountSnapshot,
    last_decision_date: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct PositionView {
    symbol: String,
    quantity: f64,
    mark_price: f64,
    unrealized_pnl: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (config_path, once) = parse_arguments()?;
    let config = load_config(&config_path)?;
    let address: SocketAddr = config.server.bind.parse().context("parse server.bind")?;
    if let Some(parent) = config.storage.ledger_path.parent() {
        fs::create_dir_all(parent).context("create ledger directory")?;
    }
    let ledger = Ledger::open(&config.storage.ledger_path)?;
    let last_decision_date = ledger
        .latest_decision_id()?
        .and_then(|id| id.strip_prefix("ranktrend-1d-").map(ToOwned::to_owned));
    let paper_config = paper_config(&config);
    let engine = if let Some(state) = ledger.load_engine_state(PAPER_SESSION_ID)? {
        PaperEngine::restore(paper_config, state)?
    } else {
        PaperEngine::new(PAPER_SESSION_ID.to_owned(), paper_config)?
    };
    ledger.record_snapshot(&engine.snapshot(Utc::now()))?;
    ledger.save_engine_state(&engine.persistent_state(), Utc::now())?;
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        ledger: Arc::new(Mutex::new(ledger)),
        model_bundle: Arc::from(config.model.bundle.to_string_lossy().to_string()),
        runtime: Arc::new(Mutex::new(RuntimeStatus {
            last_decision_date,
            ..RuntimeStatus::booting()
        })),
    };
    if once {
        run_bootstrap_and_decide(state.clone(), config).await?;
        return Ok(());
    }
    tokio::spawn(run_live_loop(state.clone(), config));
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/api/session", get(session))
        .route("/api/positions", get(positions))
        .route("/api/executions", get(executions))
        .route("/api/equity", get(equity))
        .route("/assets/dashboard.js", get(dashboard_js))
        .route("/assets/dashboard_bg.wasm", get(dashboard_wasm))
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

fn parse_arguments() -> Result<(PathBuf, bool)> {
    let mut config = PathBuf::from("configs/paper.toml");
    let mut once = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--config" => {
                config = PathBuf::from(arguments.next().context("missing --config path")?)
            }
            "--once" => once = true,
            _ => bail!("unknown argument {argument}"),
        }
    }
    Ok((config, once))
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

async fn run_live_loop(state: AppState, config: RuntimeConfig) {
    if let Err(error) = run_bootstrap_and_decide(state.clone(), config.clone()).await {
        set_error(&state, error);
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

async fn run_bootstrap_and_decide(state: AppState, config: RuntimeConfig) -> Result<()> {
    set_status(
        &state,
        "validating",
        "Checking model contracts and rebuilding causal daily state.",
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
    for (symbol, notional) in desired {
        let rules = match client.instrument_rules(&symbol).await {
            Ok(rules) => rules,
            Err(error) => {
                eprintln!("skip {symbol}: instrument constraints unavailable: {error:#}");
                continue;
            }
        };
        let book = match client
            .orderbook(&symbol, config.bybit.orderbook_depth)
            .await
        {
            Ok(book) => book,
            Err(error) => {
                eprintln!("skip {symbol}: order-book snapshot unavailable: {error:#}");
                continue;
            }
        };
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
        model_bundle: state.model_bundle.to_string(),
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
                PositionView {
                    symbol: position.symbol,
                    quantity: position.quantity,
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

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("../assets/index.html"))
}

async fn dashboard_js() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("../assets/dashboard.js"),
    )
}

async fn dashboard_wasm() -> impl IntoResponse {
    (
        [("content-type", "application/wasm")],
        include_bytes!("../assets/dashboard_bg.wasm").as_slice(),
    )
}
