use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rt_domain::{AccountSnapshot, ExecutionReport, OrderBookSnapshot, PaperState};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

pub struct Ledger {
    connection: Connection,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquityPoint {
    pub captured_at: DateTime<Utc>,
    pub equity: f64,
    pub gross_notional: f64,
    pub net_notional: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EquityBucket {
    pub captured_at: DateTime<Utc>,
    pub equity: f64,
    pub low: f64,
    pub high: f64,
    pub drawdown: f64,
    pub drawdown_at: DateTime<Utc>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformanceMetrics {
    pub max_drawdown: f64,
    pub sharpe: Option<f64>,
    pub average_daily_return: Option<f64>,
    pub profit_factor: Option<f64>,
    pub profit_factor_unbounded: bool,
    pub win_rate: Option<f64>,
    pub period_count: usize,
    pub closed_trades: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DailyPeriodReturn {
    pub period_date: String,
    pub start_equity: f64,
    pub end_equity: f64,
    pub net_return: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PeriodMetrics {
    pub average_daily_return: Option<f64>,
    pub profit_factor: Option<f64>,
    pub profit_factor_unbounded: bool,
    pub win_rate: Option<f64>,
    pub period_count: usize,
}

pub fn summarize_periods(periods: &[DailyPeriodReturn]) -> PeriodMetrics {
    let positive_returns = periods
        .iter()
        .filter(|period| period.net_return > 0.0)
        .map(|period| period.net_return)
        .collect::<Vec<_>>();
    let negative_returns = periods
        .iter()
        .filter(|period| period.net_return < 0.0)
        .map(|period| period.net_return)
        .collect::<Vec<_>>();
    let gross_positive = positive_returns.iter().sum::<f64>();
    let gross_negative = negative_returns
        .iter()
        .map(|value| value.abs())
        .sum::<f64>();
    let profit_factor_unbounded = gross_positive > 0.0 && gross_negative == 0.0;
    let profit_factor = (gross_negative > 0.0).then(|| gross_positive / gross_negative);
    let nonzero_periods = positive_returns.len() + negative_returns.len();
    let win_rate =
        (nonzero_periods > 0).then(|| positive_returns.len() as f64 / nonzero_periods as f64);
    let average_daily_return = (!periods.is_empty()).then(|| {
        periods.iter().map(|period| period.net_return).sum::<f64>() / periods.len() as f64
    });
    PeriodMetrics {
        average_daily_return,
        profit_factor,
        profit_factor_unbounded,
        win_rate,
        period_count: periods.len(),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EquityCurve {
    pub points: Vec<EquityBucket>,
    pub total_points: usize,
    pub periods: Vec<DailyPeriodReturn>,
    pub metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Copy)]
pub struct SessionPerformance {
    pub max_drawdown: f64,
}

impl Ledger {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path).context("open SQLite ledger")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        let ledger = Self { connection };
        ledger.migrate()?;
        Ok(ledger)
    }

    pub fn record_snapshot(&self, snapshot: &AccountSnapshot) -> Result<()> {
        let captured_at = timestamp(snapshot.captured_at);
        let transaction = self.connection.unchecked_transaction()?;
        transaction.execute(
            "INSERT INTO account_snapshots (
                session_id, captured_at, cash, equity, realized_pnl, unrealized_pnl,
                gross_notional, net_notional, fee_paid
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                snapshot.session_id,
                captured_at,
                snapshot.cash,
                snapshot.equity,
                snapshot.realized_pnl,
                snapshot.unrealized_pnl,
                snapshot.gross_notional,
                snapshot.net_notional,
                snapshot.fee_paid,
            ],
        )?;
        record_equity_point_in(&transaction, snapshot)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn record_equity_point(&self, snapshot: &AccountSnapshot) -> Result<()> {
        let transaction = self.connection.unchecked_transaction()?;
        record_equity_point_in(&transaction, snapshot)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn record_execution(
        &self,
        report: &ExecutionReport,
        book: &OrderBookSnapshot,
    ) -> Result<()> {
        let report_json = serde_json::to_string(report)?;
        let book_json = serde_json::to_string(book)?;
        self.connection.execute(
            "INSERT OR REPLACE INTO executions (
                execution_id, decision_id, symbol, status, executed_at, report_json, orderbook_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                report.execution_id,
                report.decision_id,
                report.symbol,
                serde_json::to_string(&report.status)?,
                timestamp(report.executed_at),
                report_json,
                book_json,
            ],
        )?;
        Ok(())
    }

    pub fn recent_executions(&self, limit: usize) -> Result<Vec<ExecutionReport>> {
        let mut statement = self
            .connection
            .prepare("SELECT report_json FROM executions ORDER BY executed_at DESC LIMIT ?1")?;
        let reports = statement
            .query_map([limit as i64], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|json| serde_json::from_str(&json).context("decode execution report"))
            .collect::<Result<Vec<_>>>()?;
        Ok(reports)
    }

    pub fn equity_curve(&self, session_id: &str, max_buckets: usize) -> Result<EquityCurve> {
        let (count, first_second, last_second): (i64, Option<i64>, Option<i64>) =
            self.connection.query_row(
                "SELECT COUNT(*), MIN(unixepoch(captured_at)), MAX(unixepoch(captured_at))
                 FROM equity_curve_points WHERE session_id=?1",
                [session_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let total_points = usize::try_from(count).unwrap_or(usize::MAX);
        let mut points = Vec::new();
        if count > 0 {
            let first_minute = first_second.context("equity curve start is absent")? / 60;
            let last_minute = last_second.context("equity curve end is absent")? / 60;
            let duration = last_minute - first_minute + 1;
            let bucket_minutes = if total_points <= max_buckets.max(1) {
                1
            } else {
                (duration + max_buckets.max(1) as i64 - 1) / max_buckets.max(1) as i64
            };
            let mut statement = self.connection.prepare(
                "WITH equity_bucketed AS (
                    SELECT captured_at, minute, equity,
                        CAST((unixepoch(captured_at) / 60 - ?2) / ?3 AS INTEGER) AS bucket
                    FROM equity_curve_points WHERE session_id=?1
                 ), equity_ranked AS (
                    SELECT captured_at, equity, bucket,
                        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY minute DESC) AS reverse_rank
                    FROM equity_bucketed
                 ), equity_extrema AS (
                    SELECT bucket, MIN(equity) AS low, MAX(equity) AS high
                    FROM equity_bucketed GROUP BY bucket
                 ), drawdown_bucketed AS (
                    SELECT id, captured_at, equity,
                        CAST((unixepoch(captured_at) / 60 - ?2) / ?3 AS INTEGER) AS bucket,
                        MAX(equity) OVER (
                            ORDER BY captured_at, id
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS peak_equity
                    FROM drawdown_points WHERE session_id=?1
                 ), drawdown_marked AS (
                    SELECT id, captured_at, bucket,
                        CASE WHEN peak_equity > 0.0
                             THEN equity / peak_equity - 1.0
                             ELSE 0.0
                        END AS drawdown
                    FROM drawdown_bucketed
                 ), drawdown_ranked AS (
                    SELECT id, captured_at, bucket, drawdown,
                        ROW_NUMBER() OVER (
                            PARTITION BY bucket
                            ORDER BY drawdown ASC, captured_at ASC, id ASC
                        ) AS drawdown_rank
                    FROM drawdown_marked
                 )
                 SELECT latest.captured_at, latest.equity,
                        extrema.low, extrema.high,
                        worst.drawdown, worst.captured_at
                 FROM equity_ranked AS latest
                 JOIN equity_extrema AS extrema USING (bucket)
                 JOIN drawdown_ranked AS worst USING (bucket)
                 WHERE latest.reverse_rank=1 AND worst.drawdown_rank=1
                 ORDER BY latest.bucket",
            )?;
            points = statement
                .query_map(params![session_id, first_minute, bucket_minutes], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .map(|(captured_at, equity, low, high, drawdown, drawdown_at)| {
                    Ok(EquityBucket {
                        captured_at: parse_timestamp(&captured_at)?,
                        equity,
                        low,
                        high,
                        drawdown,
                        drawdown_at: parse_timestamp(&drawdown_at)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        }
        let periods = self.completed_period_returns(session_id)?;
        let metrics = self.performance_metrics(session_id, &periods)?;
        Ok(EquityCurve {
            points,
            total_points,
            periods,
            metrics,
        })
    }

    pub fn completed_period_returns(&self, session_id: &str) -> Result<Vec<DailyPeriodReturn>> {
        let mut statement = self.connection.prepare(
            "WITH decision_marks AS (
                SELECT substr(decisions.decision_id, 10) AS period_date,
                       (
                           SELECT snapshots.equity
                           FROM account_snapshots AS snapshots
                           WHERE snapshots.session_id=?1
                             AND snapshots.captured_at <= decisions.decided_at
                           ORDER BY snapshots.captured_at DESC, snapshots.id DESC
                           LIMIT 1
                       ) AS end_equity
                FROM decisions
                WHERE decisions.decision_id LIKE 'airok-1d-%'
            ), ordered AS (
                SELECT period_date, end_equity,
                       LAG(end_equity) OVER (ORDER BY period_date) AS start_equity
                FROM decision_marks
                WHERE end_equity IS NOT NULL
            )
            SELECT period_date, start_equity, end_equity,
                   end_equity / start_equity - 1.0 AS net_return,
                   end_equity - start_equity AS pnl
            FROM ordered
                WHERE start_equity > 0.0 AND end_equity > 0.0
            ORDER BY period_date",
        )?;
        statement
            .query_map([session_id], |row| {
                Ok(DailyPeriodReturn {
                    period_date: row.get(0)?,
                    start_equity: row.get(1)?,
                    end_equity: row.get(2)?,
                    net_return: row.get(3)?,
                    pnl: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn session_performance(&self, session_id: &str) -> Result<Option<SessionPerformance>> {
        self.connection
            .query_row(
                "SELECT max_drawdown FROM session_performance WHERE session_id=?1",
                [session_id],
                |row| {
                    Ok(SessionPerformance {
                        max_drawdown: row.get(0)?,
                    })
                },
            )
            .optional()
            .context("read session performance")
    }

    pub fn session_start_equity(&self, session_id: &str) -> Result<Option<f64>> {
        self.connection
            .query_row(
                "SELECT equity FROM account_snapshots WHERE session_id=?1 ORDER BY id ASC LIMIT 1",
                [session_id],
                |row| row.get(0),
            )
            .optional()
            .context("read session start equity")
    }

    fn performance_metrics(
        &self,
        session_id: &str,
        periods: &[DailyPeriodReturn],
    ) -> Result<PerformanceMetrics> {
        let max_drawdown = self
            .session_performance(session_id)?
            .map_or(0.0, |value| value.max_drawdown);
        let (mut count, mut mean, mut m2) = self.connection.query_row(
            "SELECT return_count, mean_return, return_m2 FROM equity_return_stats WHERE session_id=?1",
            [session_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?)),
        ).optional()?.unwrap_or((0, 0.0, 0.0));
        let latest = self.connection.prepare(
            "SELECT equity FROM equity_curve_points WHERE session_id=?1 ORDER BY minute DESC LIMIT 2"
        )?.query_map([session_id], |row| row.get::<_, f64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if latest.len() == 2 && latest[1] > 0.0 {
            let value = latest[0] / latest[1] - 1.0;
            count += 1;
            let delta = value - mean;
            mean += delta / count as f64;
            m2 += delta * (value - mean);
        }
        let sharpe = (count > 1 && m2 > 0.0)
            .then(|| mean / (m2 / (count - 1) as f64).sqrt() * (365.0_f64 * 24.0 * 60.0).sqrt());
        let period_metrics = summarize_periods(periods);
        let mut closed_trades = 0;
        let mut statement = self
            .connection
            .prepare("SELECT report_json FROM executions ORDER BY executed_at")?;
        for report in statement.query_map([], |row| row.get::<_, String>(0))? {
            let report: ExecutionReport = serde_json::from_str(&report?)?;
            if report.closed_quantity > 0.0 {
                closed_trades += 1;
            }
        }
        Ok(PerformanceMetrics {
            max_drawdown,
            sharpe,
            average_daily_return: period_metrics.average_daily_return,
            profit_factor: period_metrics.profit_factor,
            profit_factor_unbounded: period_metrics.profit_factor_unbounded,
            win_rate: period_metrics.win_rate,
            closed_trades,
            period_count: period_metrics.period_count,
        })
    }

    pub fn save_engine_state(&self, state: &PaperState, updated_at: DateTime<Utc>) -> Result<()> {
        self.connection.execute(
            "INSERT INTO paper_state (session_id, updated_at, state_json) VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, state_json=excluded.state_json",
            params![state.session_id, timestamp(updated_at), serde_json::to_string(state)?],
        )?;
        Ok(())
    }

    pub fn load_engine_state(&self, session_id: &str) -> Result<Option<PaperState>> {
        let state = self
            .connection
            .query_row(
                "SELECT state_json FROM paper_state WHERE session_id=?1",
                [session_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        state
            .map(|value| serde_json::from_str(&value).context("decode paper state"))
            .transpose()
    }

    pub fn has_decision(&self, decision_id: &str) -> Result<bool> {
        self.connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM decisions WHERE decision_id=?1)",
                [decision_id],
                |row| row.get(0),
            )
            .context("query processed decision")
    }

    pub fn record_decision(&self, decision_id: &str, decided_at: DateTime<Utc>) -> Result<()> {
        self.connection.execute(
            "INSERT INTO decisions (decision_id, decided_at) VALUES (?1, ?2)",
            params![decision_id, timestamp(decided_at)],
        )?;
        Ok(())
    }

    pub fn latest_decision_id(&self) -> Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT decision_id FROM decisions ORDER BY decided_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .context("query latest paper decision")
    }

    fn migrate(&self) -> Result<()> {
        self.connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                gross_notional REAL NOT NULL,
                net_notional REAL NOT NULL,
                fee_paid REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS account_snapshots_session_time
                ON account_snapshots(session_id, captured_at);
            CREATE TABLE IF NOT EXISTS equity_curve_points (
                session_id TEXT NOT NULL,
                minute TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                equity REAL NOT NULL,
                gross_notional REAL NOT NULL,
                net_notional REAL NOT NULL,
                PRIMARY KEY(session_id, minute)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS drawdown_points (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                equity REAL NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS drawdown_points_identity
                ON drawdown_points(session_id, captured_at, equity);
            CREATE INDEX IF NOT EXISTS drawdown_points_session_time
                ON drawdown_points(session_id, id);
            CREATE INDEX IF NOT EXISTS drawdown_points_session_captured_at
                ON drawdown_points(session_id, captured_at, id);
            CREATE TABLE IF NOT EXISTS equity_return_stats (
                session_id TEXT PRIMARY KEY,
                return_count INTEGER NOT NULL,
                mean_return REAL NOT NULL,
                return_m2 REAL NOT NULL,
                updated_through TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_performance (
                session_id TEXT PRIMARY KEY,
                peak_equity REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                report_json TEXT NOT NULL,
                orderbook_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS executions_time ON executions(executed_at DESC);
            CREATE TABLE IF NOT EXISTS paper_state (
                session_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                decided_at TEXT NOT NULL
            );
            ",
        )?;
        self.connection
            .execute_batch("DROP TABLE IF EXISTS daily_account_snapshots;")?;
        self.connection.execute_batch(
            "WITH ordered AS (
                SELECT session_id, id, captured_at, equity,
                    MAX(equity) OVER (
                        PARTITION BY session_id ORDER BY captured_at, id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS peak_equity
                FROM account_snapshots
            ), stats AS (
                SELECT session_id,
                    MAX(peak_equity) AS peak_equity,
                    MIN(CASE WHEN peak_equity > 0.0 THEN equity / peak_equity - 1.0 ELSE 0.0 END) AS max_drawdown,
                    MAX(captured_at) AS updated_at
                FROM ordered
                GROUP BY session_id
                )
            INSERT INTO session_performance (
                session_id, peak_equity, max_drawdown, updated_at
            )
            SELECT stats.session_id, stats.peak_equity, stats.max_drawdown, stats.updated_at
            FROM stats
            WHERE true
            ON CONFLICT(session_id) DO UPDATE SET
                peak_equity = excluded.peak_equity,
                max_drawdown = excluded.max_drawdown,
                updated_at = excluded.updated_at;",
        )?;
        self.connection.execute_batch(
            "INSERT OR IGNORE INTO equity_curve_points (
                session_id, minute, captured_at, equity, gross_notional, net_notional
             )
             SELECT snapshots.session_id,
                    strftime('%Y-%m-%dT%H:%M:00Z', snapshots.captured_at),
                    snapshots.captured_at, snapshots.equity,
                    snapshots.gross_notional, snapshots.net_notional
             FROM account_snapshots AS snapshots
             JOIN (
                SELECT session_id,
                       strftime('%Y-%m-%dT%H:%M:00Z', captured_at) AS minute,
                       MAX(id) AS id
                FROM account_snapshots GROUP BY session_id, minute
             ) AS latest ON latest.id=snapshots.id;",
        )?;
        self.connection.execute_batch(
            "INSERT OR IGNORE INTO drawdown_points (session_id, captured_at, equity)
             SELECT session_id, captured_at, equity FROM account_snapshots;
             INSERT OR IGNORE INTO drawdown_points (session_id, captured_at, equity)
             SELECT session_id, captured_at, equity FROM equity_curve_points;",
        )?;
        self.connection.execute_batch(
            "WITH ordered AS (
                SELECT session_id, id, captured_at, equity,
                    MAX(equity) OVER (
                        PARTITION BY session_id ORDER BY captured_at, id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS peak_equity
                FROM drawdown_points
            ), stats AS (
                SELECT session_id,
                    MAX(peak_equity) AS peak_equity,
                    MIN(CASE WHEN peak_equity > 0.0 THEN equity / peak_equity - 1.0 ELSE 0.0 END) AS max_drawdown,
                    MAX(captured_at) AS updated_at
                FROM ordered
                GROUP BY session_id
            )
            INSERT INTO session_performance (
                session_id, peak_equity, max_drawdown, updated_at
            )
            SELECT stats.session_id, stats.peak_equity, stats.max_drawdown, stats.updated_at
            FROM stats
            WHERE true
            ON CONFLICT(session_id) DO UPDATE SET
                peak_equity = excluded.peak_equity,
                max_drawdown = excluded.max_drawdown,
                updated_at = excluded.updated_at;",
        )?;
        let has_return_stats: bool = self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM equity_return_stats)",
            [],
            |row| row.get(0),
        )?;
        if !has_return_stats {
            self.connection.execute_batch(
                "WITH ordered AS (
                SELECT session_id, minute, equity,
                       LAG(equity) OVER (PARTITION BY session_id ORDER BY minute) AS previous,
                       ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY minute DESC) AS reverse_rank
                FROM equity_curve_points
             ), returns AS (
                SELECT session_id, minute, equity / previous - 1.0 AS value
                FROM ordered WHERE previous > 0.0 AND reverse_rank > 1
             ), means AS (
                SELECT session_id, COUNT(*) AS return_count, AVG(value) AS mean_return,
                       MAX(minute) AS updated_through
                FROM returns GROUP BY session_id
             )
             INSERT INTO equity_return_stats (
                session_id, return_count, mean_return, return_m2, updated_through
             )
             SELECT returns.session_id, means.return_count, means.mean_return,
                    SUM((returns.value - means.mean_return) * (returns.value - means.mean_return)),
                    means.updated_through
             FROM returns JOIN means USING(session_id) GROUP BY returns.session_id;",
            )?;
        }
        Ok(())
    }
}

fn upsert_session_performance_in(
    transaction: &Transaction<'_>,
    session_id: &str,
    peak_equity: f64,
    max_drawdown: f64,
    updated_at: &str,
) -> Result<()> {
    transaction.execute(
        "INSERT INTO session_performance (
            session_id, peak_equity, max_drawdown, updated_at
         ) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(session_id) DO UPDATE SET
            peak_equity = excluded.peak_equity,
            max_drawdown = excluded.max_drawdown,
            updated_at = excluded.updated_at",
        params![session_id, peak_equity, max_drawdown, updated_at],
    )?;
    Ok(())
}

fn recompute_session_performance_in(transaction: &Transaction<'_>, session_id: &str) -> Result<()> {
    let (peak_equity, max_drawdown, updated_at): (Option<f64>, Option<f64>, Option<String>) =
        transaction.query_row(
            "WITH ordered AS (
            SELECT id, captured_at, equity,
                MAX(equity) OVER (
                    ORDER BY captured_at, id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS peak_equity
            FROM drawdown_points
            WHERE session_id=?1
        )
        SELECT MAX(peak_equity),
               MIN(CASE WHEN peak_equity > 0.0 THEN equity / peak_equity - 1.0 ELSE 0.0 END),
               MAX(captured_at)
        FROM ordered",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
    if let (Some(peak_equity), Some(max_drawdown), Some(updated_at)) =
        (peak_equity, max_drawdown, updated_at)
    {
        upsert_session_performance_in(
            transaction,
            session_id,
            peak_equity,
            max_drawdown,
            &updated_at,
        )?;
    }
    Ok(())
}

fn record_equity_point_in(transaction: &Transaction<'_>, snapshot: &AccountSnapshot) -> Result<()> {
    let captured_at = timestamp(snapshot.captured_at);
    let previous_latest: Option<String> = transaction
        .query_row(
            "SELECT captured_at
             FROM drawdown_points
             WHERE session_id=?1
             ORDER BY captured_at DESC, id DESC
             LIMIT 1",
            [snapshot.session_id.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    transaction.execute(
        "INSERT OR IGNORE INTO drawdown_points (session_id, captured_at, equity)
         VALUES (?1, ?2, ?3)",
        params![snapshot.session_id, captured_at, snapshot.equity,],
    )?;
    let drawdown_inserted = transaction.changes() > 0;
    let minute = snapshot
        .captured_at
        .format("%Y-%m-%dT%H:%M:00Z")
        .to_string();
    let exists: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM equity_curve_points WHERE session_id=?1 AND minute=?2)",
        params![snapshot.session_id, minute],
        |row| row.get(0),
    )?;
    if !exists {
        let previous = transaction
            .prepare(
                "SELECT minute, equity FROM equity_curve_points
                 WHERE session_id=?1 ORDER BY minute DESC LIMIT 2",
            )?
            .query_map([&snapshot.session_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if previous.len() == 2 && previous[1].1 > 0.0 {
            let value = previous[0].1 / previous[1].1 - 1.0;
            transaction.execute(
                "INSERT INTO equity_return_stats (
                    session_id, return_count, mean_return, return_m2, updated_through
                 ) VALUES (?1, 1, ?2, 0.0, ?3)
                 ON CONFLICT(session_id) DO UPDATE SET
                    return_m2 = equity_return_stats.return_m2
                        + (?2 - equity_return_stats.mean_return)
                        * (?2 - (equity_return_stats.mean_return
                            + (?2 - equity_return_stats.mean_return)
                              / (equity_return_stats.return_count + 1))),
                    mean_return = equity_return_stats.mean_return
                        + (?2 - equity_return_stats.mean_return)
                          / (equity_return_stats.return_count + 1),
                    return_count = equity_return_stats.return_count + 1,
                    updated_through = excluded.updated_through
                 WHERE excluded.updated_through > equity_return_stats.updated_through",
                params![snapshot.session_id, value, previous[0].0],
            )?;
        }
    }
    transaction.execute(
        "INSERT INTO equity_curve_points (
            session_id, minute, captured_at, equity, gross_notional, net_notional
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(session_id, minute) DO UPDATE SET
            captured_at=excluded.captured_at,
            equity=excluded.equity,
            gross_notional=excluded.gross_notional,
            net_notional=excluded.net_notional",
        params![
            snapshot.session_id,
            minute,
            captured_at,
            snapshot.equity,
            snapshot.gross_notional,
            snapshot.net_notional,
        ],
    )?;
    if drawdown_inserted {
        let chronological = previous_latest
            .as_ref()
            .map_or(true, |previous| captured_at >= *previous);
        // Chronological appends use the persisted running peak; backfills recompute history.
        if chronological {
            let previous_performance: Option<(f64, f64)> = transaction
                .query_row(
                    "SELECT peak_equity, max_drawdown
                     FROM session_performance
                     WHERE session_id=?1",
                    [snapshot.session_id.as_str()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((previous_peak, previous_max_drawdown)) = previous_performance {
                let current_drawdown = if previous_peak > 0.0 {
                    snapshot.equity / previous_peak - 1.0
                } else {
                    0.0
                };
                upsert_session_performance_in(
                    transaction,
                    &snapshot.session_id,
                    previous_peak.max(snapshot.equity),
                    previous_max_drawdown.min(current_drawdown),
                    &captured_at,
                )?;
            } else {
                recompute_session_performance_in(transaction, &snapshot.session_id)?;
            }
        } else {
            recompute_session_performance_in(transaction, &snapshot.session_id)?;
        }
    }
    Ok(())
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)
        .context("parse equity timestamp")?
        .with_timezone(&Utc))
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Timelike, Utc};
    use rt_domain::{
        AccountSnapshot, ExecutionReport, FillStatus, OrderBookSnapshot, PriceLevel, Side,
    };
    use rusqlite::Connection;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{DailyPeriodReturn, Ledger, summarize_periods};

    #[test]
    fn stores_account_snapshots_in_wal_database() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        ledger
            .record_snapshot(&AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at: Utc::now(),
                cash: 1_000.0,
                equity: 1_000.0,
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                gross_notional: 0.0,
                net_notional: 0.0,
                fee_paid: 0.0,
            })
            .expect("snapshot persists");
    }

    #[test]
    fn preserves_all_time_drawdown_across_minute_equity_points() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let started_at = Utc::now() - Duration::days(10);
        for (offset, equity) in [1_000.0, 980.0, 1_010.0, 1_005.0].into_iter().enumerate() {
            let captured_at = started_at + Duration::days(offset as i64);
            let snapshot = AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at,
                cash: equity,
                equity,
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                gross_notional: 0.0,
                net_notional: 0.0,
                fee_paid: 0.0,
            };
            ledger
                .record_snapshot(&snapshot)
                .expect("snapshot persists");
            ledger
                .record_decision(
                    &format!("airok-1d-{}", captured_at.date_naive()),
                    captured_at,
                )
                .expect("decision persists");
        }

        let chart = ledger.equity_curve("test", 100).expect("chart reads");
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance exists");
        let starting_equity = ledger
            .session_start_equity("test")
            .expect("starting equity reads")
            .expect("starting equity exists");

        assert_eq!(chart.points.len(), 4);
        assert_eq!(chart.total_points, 4);
        assert_eq!(starting_equity, 1_000.0);
        assert_eq!(chart.points[0].drawdown, 0.0);
        assert!((chart.points[1].drawdown + 0.02).abs() < 1e-12);
        assert_eq!(chart.points[2].drawdown, 0.0);
        assert!((chart.points[3].drawdown - (1_005.0 / 1_010.0 - 1.0)).abs() < 1e-12);
        assert_eq!(
            chart.points[1].drawdown_at.timestamp_millis(),
            (started_at + Duration::days(1)).timestamp_millis()
        );
        assert!((performance.max_drawdown + 0.02).abs() < 1e-12);
        let expected_daily_return = (-0.02 + 1_010.0 / 980.0 - 1.0 + 1_005.0 / 1_010.0 - 1.0) / 3.0;
        assert!(
            (chart
                .metrics
                .average_daily_return
                .expect("daily return exists")
                - expected_daily_return)
                .abs()
                < 1e-12
        );
        assert_eq!(chart.metrics.period_count, 3);
        assert!((chart.metrics.win_rate.expect("period win rate") - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn keeps_one_latest_curve_point_per_minute_without_retention_limit() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let minute = Utc::now()
            .with_second(0)
            .and_then(|value| value.with_nanosecond(0))
            .expect("minute truncates");
        for (seconds, equity) in [(1, 1_000.0), (15, 990.0), (59, 995.0)] {
            ledger
                .record_equity_point(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at: minute + Duration::seconds(seconds),
                    cash: equity,
                    equity,
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("minute point persists");
        }
        let curve = ledger.equity_curve("test", 100).expect("curve reads");
        assert_eq!(curve.total_points, 1);
        assert_eq!(curve.points[0].equity, 995.0);
        assert!((curve.points[0].drawdown + 0.01).abs() < 1e-12);
        assert_eq!(
            curve.points[0].drawdown_at.timestamp(),
            (minute + Duration::seconds(15)).timestamp()
        );
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance exists");
        assert!((performance.max_drawdown + 0.01).abs() < 1e-12);
    }

    #[test]
    fn drawdown_uses_capture_time_for_out_of_order_marks() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let started_at = Utc::now() - Duration::days(3);
        for (offset, equity) in [(1, 800.0), (0, 1_000.0), (2, 900.0)] {
            let captured_at = started_at + Duration::days(offset);
            ledger
                .record_equity_point(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at,
                    cash: equity,
                    equity,
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("out-of-order mark persists");
        }

        let curve = ledger.equity_curve("test", 100).expect("curve reads");
        assert_eq!(curve.points.len(), 3);
        assert_eq!(curve.points[0].equity, 1_000.0);
        assert_eq!(curve.points[0].drawdown, 0.0);
        assert!((curve.points[1].drawdown + 0.2).abs() < 1e-12);
        assert!((curve.points[2].drawdown + 0.1).abs() < 1e-12);
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance exists");
        assert!((performance.max_drawdown + 0.2).abs() < 1e-12);
    }

    #[test]
    fn downsampling_preserves_all_points_and_bucket_extrema() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let start = Utc::now()
            .with_second(0)
            .and_then(|value| value.with_nanosecond(0))
            .expect("minute truncates");
        for offset in 0..120 {
            let equity = if offset == 57 {
                700.0
            } else {
                1_000.0 + offset as f64
            };
            ledger
                .record_equity_point(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at: start + Duration::minutes(offset),
                    cash: equity,
                    equity,
                    realized_pnl: 0.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("point persists");
        }
        let curve = ledger.equity_curve("test", 10).expect("curve reads");
        assert_eq!(curve.total_points, 120);
        assert!(curve.points.len() <= 10);
        assert_eq!(
            curve.points.iter().map(|point| point.low).reduce(f64::min),
            Some(700.0)
        );
        assert!(
            curve
                .points
                .iter()
                .map(|point| point.drawdown)
                .reduce(f64::min)
                .expect("drawdown points exist")
                < -0.33
        );
    }

    #[test]
    fn closed_execution_pnl_does_not_drive_period_metrics() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let now = Utc::now();
        let book = OrderBookSnapshot {
            symbol: "BTCUSDT".to_owned(),
            captured_at: now,
            update_id: 1,
            bids: vec![PriceLevel {
                price: 99.0,
                quantity: 10.0,
            }],
            asks: vec![PriceLevel {
                price: 101.0,
                quantity: 10.0,
            }],
        };
        for (index, pnl) in [12.0, -4.0, 0.0].into_iter().enumerate() {
            ledger
                .record_execution(
                    &ExecutionReport {
                        execution_id: format!("close-{index}"),
                        decision_id: "d1".to_owned(),
                        symbol: "BTCUSDT".to_owned(),
                        side: Side::Sell,
                        status: FillStatus::Filled,
                        requested_quantity: 1.0,
                        filled_quantity: 1.0,
                        remaining_quantity: 0.0,
                        benchmark_mid: Some(100.0),
                        vwap: Some(99.0),
                        notional: 99.0,
                        fee: 0.1,
                        closed_quantity: 1.0,
                        closed_pnl: pnl,
                        slippage_bps: Some(0.0),
                        consumed_levels: Vec::new(),
                        executed_at: now + Duration::seconds(index as i64),
                        rejection_reason: None,
                    },
                    &book,
                )
                .expect("execution persists");
        }
        let metrics = ledger
            .equity_curve("test", 100)
            .expect("metrics read")
            .metrics;
        assert_eq!(metrics.closed_trades, 3);
        assert_eq!(metrics.period_count, 0);
        assert_eq!(metrics.win_rate, None);
        assert_eq!(metrics.profit_factor, None);
        assert!(!metrics.profit_factor_unbounded);
    }

    #[test]
    fn period_win_rate_and_profit_factor_ignore_zero_returns() {
        let periods = [0.02, -0.01, 0.03, -0.005]
            .into_iter()
            .enumerate()
            .map(|(index, net_return)| DailyPeriodReturn {
                period_date: format!("2026-08-{:02}", index + 1),
                start_equity: 1_000.0,
                end_equity: 1_000.0 * (1.0 + net_return),
                net_return,
                pnl: 1_000.0 * net_return,
            })
            .collect::<Vec<_>>();
        let metrics = summarize_periods(&periods);
        assert_eq!(metrics.period_count, 4);
        assert_eq!(metrics.win_rate, Some(0.5));
        assert!((metrics.profit_factor.expect("profit factor") - 3.333333333333333).abs() < 1e-12);

        let with_zero = [0.02, 0.0, -0.01]
            .into_iter()
            .enumerate()
            .map(|(index, net_return)| DailyPeriodReturn {
                period_date: format!("2026-09-{:02}", index + 1),
                start_equity: 1_000.0,
                end_equity: 1_000.0 * (1.0 + net_return),
                net_return,
                pnl: 1_000.0 * net_return,
            })
            .collect::<Vec<_>>();
        let metrics = summarize_periods(&with_zero);
        assert_eq!(metrics.period_count, 3);
        assert_eq!(metrics.win_rate, Some(0.5));
        assert!((metrics.profit_factor.expect("profit factor") - 2.0).abs() < 1e-12);

        let only_positive = summarize_periods(&with_zero[..1]);
        assert_eq!(only_positive.profit_factor, None);
        assert!(only_positive.profit_factor_unbounded);
        let only_negative = summarize_periods(&with_zero[2..]);
        assert_eq!(only_negative.profit_factor, Some(0.0));
        assert!(!only_negative.profit_factor_unbounded);
    }

    #[test]
    fn period_mark_uses_latest_captured_snapshot_not_insert_id() {
        let ledger = Ledger::open(":memory:").expect("ledger opens");
        let first_date = Utc::now() - Duration::days(3);
        let second_date = first_date + Duration::days(1);
        ledger
            .record_snapshot(&AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at: first_date,
                cash: 1_000.0,
                equity: 1_000.0,
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                gross_notional: 0.0,
                net_notional: 0.0,
                fee_paid: 0.0,
            })
            .expect("first snapshot persists");
        ledger
            .record_decision(&format!("airok-1d-{}", first_date.date_naive()), first_date)
            .expect("first decision persists");
        ledger
            .record_snapshot(&AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at: second_date,
                cash: 1_020.0,
                equity: 1_020.0,
                realized_pnl: 20.0,
                unrealized_pnl: 0.0,
                gross_notional: 0.0,
                net_notional: 0.0,
                fee_paid: 0.0,
            })
            .expect("latest snapshot persists");
        ledger
            .record_snapshot(&AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at: second_date - Duration::hours(1),
                cash: 1_010.0,
                equity: 1_010.0,
                realized_pnl: 10.0,
                unrealized_pnl: 0.0,
                gross_notional: 0.0,
                net_notional: 0.0,
                fee_paid: 0.0,
            })
            .expect("out-of-order snapshot persists");
        ledger
            .record_decision(
                &format!("airok-1d-{}", second_date.date_naive()),
                second_date + Duration::hours(1),
            )
            .expect("second decision persists");

        let periods = ledger
            .completed_period_returns("test")
            .expect("periods read");
        assert_eq!(periods.len(), 1);
        assert_eq!(periods[0].end_equity, 1_020.0);
        assert!((periods[0].net_return - 0.02).abs() < 1e-12);
    }

    #[test]
    fn period_metrics_ignore_partial_marks_and_survive_restart() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("airok-period-metrics-{suffix}.sqlite3"));
        let started_at = Utc::now() - Duration::days(20);
        {
            let ledger = Ledger::open(&path).expect("ledger opens");
            for (offset, equity) in [(0, 1_000.0), (1, 1_020.0), (2, 1_009.8)] {
                let captured_at = started_at + Duration::days(offset as i64);
                ledger
                    .record_snapshot(&AccountSnapshot {
                        session_id: "test".to_owned(),
                        captured_at,
                        cash: equity,
                        equity,
                        realized_pnl: 0.0,
                        unrealized_pnl: 0.0,
                        gross_notional: 0.0,
                        net_notional: 0.0,
                        fee_paid: 0.0,
                    })
                    .expect("snapshot persists");
                ledger
                    .record_decision(
                        &format!("airok-1d-{}", captured_at.date_naive()),
                        captured_at,
                    )
                    .expect("decision persists");
            }
            let before = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(before.metrics.period_count, 2);
            assert_eq!(before.metrics.win_rate, Some(0.5));
            assert!((before.metrics.profit_factor.expect("profit factor") - 2.0).abs() < 1e-12);

            let partial_book = OrderBookSnapshot {
                symbol: "BTCUSDT".to_owned(),
                captured_at: started_at + Duration::days(1) + Duration::hours(12),
                update_id: 1,
                bids: vec![PriceLevel {
                    price: 99.0,
                    quantity: 10.0,
                }],
                asks: vec![PriceLevel {
                    price: 101.0,
                    quantity: 10.0,
                }],
            };
            ledger
                .record_execution(
                    &ExecutionReport {
                        execution_id: "partial-reduce".to_owned(),
                        decision_id: "airok-1d-partial-reduce".to_owned(),
                        symbol: "BTCUSDT".to_owned(),
                        side: Side::Sell,
                        status: FillStatus::Filled,
                        requested_quantity: 8.0,
                        filled_quantity: 8.0,
                        remaining_quantity: 0.0,
                        benchmark_mid: Some(100.0),
                        vwap: Some(99.0),
                        notional: 792.0,
                        fee: 0.8,
                        closed_quantity: 8.0,
                        closed_pnl: 500.0,
                        slippage_bps: Some(0.0),
                        consumed_levels: Vec::new(),
                        executed_at: partial_book.captured_at,
                        rejection_reason: None,
                    },
                    &partial_book,
                )
                .expect("partial reduction persists");
            let after_partial_reduction = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(
                after_partial_reduction.metrics.period_count,
                before.metrics.period_count
            );
            assert_eq!(
                after_partial_reduction.metrics.win_rate,
                before.metrics.win_rate
            );
            assert_eq!(
                after_partial_reduction.metrics.profit_factor,
                before.metrics.profit_factor
            );

            ledger
                .record_equity_point(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at: Utc::now(),
                    cash: 2_000.0,
                    equity: 2_000.0,
                    realized_pnl: 1_000.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("partial mark persists");
            let after = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(after.metrics.period_count, before.metrics.period_count);
            assert_eq!(after.metrics.win_rate, before.metrics.win_rate);
            assert_eq!(after.metrics.profit_factor, before.metrics.profit_factor);

            let today = Utc::now();
            let today_equity = 1_019.898;
            ledger
                .record_snapshot(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at: today,
                    cash: today_equity,
                    equity: today_equity,
                    realized_pnl: today_equity - 1_000.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("current-day decision mark persists");
            ledger
                .record_decision(&format!("airok-1d-{}", today.date_naive()), today)
                .expect("current-day decision persists");
            let current_period = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(current_period.metrics.period_count, 3);
            assert_eq!(current_period.metrics.win_rate, Some(2.0 / 3.0));
            assert_eq!(current_period.metrics.profit_factor, Some(3.0));
            let latest_period = current_period
                .periods
                .last()
                .expect("current period exists");
            assert_eq!(latest_period.period_date, today.date_naive().to_string());
            assert!((latest_period.net_return - 0.01).abs() < 1e-12);

            ledger
                .record_snapshot(&AccountSnapshot {
                    session_id: "test".to_owned(),
                    captured_at: today + Duration::minutes(1),
                    cash: 1_025.0,
                    equity: 1_025.0,
                    realized_pnl: 25.0,
                    unrealized_pnl: 0.0,
                    gross_notional: 0.0,
                    net_notional: 0.0,
                    fee_paid: 0.0,
                })
                .expect("open current period mark persists");
            let open_period = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(
                open_period.metrics.period_count,
                current_period.metrics.period_count
            );
            assert_eq!(
                open_period.metrics.win_rate,
                current_period.metrics.win_rate
            );
            assert_eq!(
                open_period.metrics.profit_factor,
                current_period.metrics.profit_factor
            );
            assert_eq!(open_period.periods.len(), current_period.periods.len());
            assert_eq!(
                open_period.periods.last().map(|period| period.end_equity),
                Some(today_equity)
            );
        }
        {
            let ledger = Ledger::open(&path).expect("ledger reopens");
            let after_restart = ledger.equity_curve("test", 100).expect("metrics read");
            assert_eq!(after_restart.metrics.period_count, 3);
            assert_eq!(after_restart.metrics.win_rate, Some(2.0 / 3.0));
            assert!(
                (after_restart.metrics.profit_factor.expect("profit factor") - 3.0).abs() < 1e-12
            );
        }
        fs::remove_file(path).expect("temporary ledger removes");
    }

    #[test]
    fn backfills_all_time_drawdown_for_an_existing_ledger() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("airok-ledger-{suffix}.sqlite3"));
        let connection = Connection::open(&path).expect("legacy ledger opens");
        connection
            .execute_batch(
                "CREATE TABLE account_snapshots (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    gross_notional REAL NOT NULL,
                    net_notional REAL NOT NULL,
                    fee_paid REAL NOT NULL
                );
                INSERT INTO account_snapshots VALUES
                    (1, 'test', '2026-08-13T00:01:00Z', 980.0, 980.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (2, 'test', '2026-08-13T00:00:00Z', 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (3, 'test', '2026-08-13T00:02:00Z', 1010.0, 1010.0, 0.0, 0.0, 0.0, 0.0, 0.0);",
            )
            .expect("legacy snapshots persist");
        drop(connection);

        let ledger = Ledger::open(&path).expect("ledger migrates");
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance backfilled");
        assert!((performance.max_drawdown + 0.02).abs() < 1e-12);
        drop(ledger);
        fs::remove_file(path).expect("temporary ledger removes");
    }
}
