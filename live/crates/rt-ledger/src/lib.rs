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
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformanceMetrics {
    pub max_drawdown: f64,
    pub sharpe: Option<f64>,
    pub average_daily_return: Option<f64>,
    pub profit_factor: Option<f64>,
    pub win_rate: Option<f64>,
    pub closed_trades: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EquityCurve {
    pub points: Vec<EquityBucket>,
    pub total_points: usize,
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
            let first_second = first_second.context("equity curve start is absent")?;
            let duration = last_second.context("equity curve end is absent")? - first_second + 1;
            let bucket_seconds = if total_points <= max_buckets.max(1) {
                1
            } else {
                (duration + max_buckets.max(1) as i64 - 1) / max_buckets.max(1) as i64
            };
            let mut statement = self.connection.prepare(
                "WITH bucketed AS (
                    SELECT captured_at, equity,
                        CAST((unixepoch(captured_at) - ?2) / ?3 AS INTEGER) AS bucket
                    FROM equity_curve_points WHERE session_id=?1
                 ), ranked AS (
                    SELECT captured_at, equity, bucket,
                        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY captured_at DESC) AS reverse_rank
                    FROM bucketed
                 )
                 SELECT MAX(CASE WHEN reverse_rank=1 THEN captured_at END),
                        MAX(CASE WHEN reverse_rank=1 THEN equity END),
                        MIN(equity), MAX(equity)
                 FROM ranked GROUP BY bucket ORDER BY bucket",
            )?;
            points = statement
                .query_map(params![session_id, first_second, bucket_seconds], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .map(|(captured_at, equity, low, high)| {
                    Ok(EquityBucket {
                        captured_at: parse_timestamp(&captured_at)?,
                        equity,
                        low,
                        high,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(EquityCurve {
            points,
            total_points,
            metrics: self.performance_metrics(session_id)?,
        })
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

    fn performance_metrics(&self, session_id: &str) -> Result<PerformanceMetrics> {
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
        let mut wins = Vec::new();
        let mut losses = Vec::new();
        let mut closed_trades = 0;
        let mut statement = self
            .connection
            .prepare("SELECT report_json FROM executions ORDER BY executed_at")?;
        for report in statement.query_map([], |row| row.get::<_, String>(0))? {
            let report: ExecutionReport = serde_json::from_str(&report?)?;
            if report.closed_quantity > 0.0 {
                closed_trades += 1;
                if report.closed_pnl > 0.0 {
                    wins.push(report.closed_pnl);
                }
                if report.closed_pnl < 0.0 {
                    losses.push(report.closed_pnl);
                }
            }
        }
        let profit_factor = (!wins.is_empty() && !losses.is_empty())
            .then(|| wins.iter().sum::<f64>() / losses.iter().sum::<f64>().abs());
        let win_rate = (closed_trades > 0).then(|| wins.len() as f64 / closed_trades as f64);
        let average_daily_return = self
            .connection
            .query_row(
                "WITH daily_marks AS (
                    SELECT snapshots.session_id,
                           date(snapshots.captured_at) AS day,
                           snapshots.equity,
                           ROW_NUMBER() OVER (
                               PARTITION BY snapshots.session_id, date(snapshots.captured_at)
                               ORDER BY snapshots.captured_at DESC, snapshots.id DESC
                           ) AS reverse_rank
                    FROM account_snapshots AS snapshots
                    WHERE snapshots.session_id=?1
                 ), daily_returns AS (
                    SELECT equity / LAG(equity) OVER (ORDER BY day) - 1.0 AS value
                    FROM daily_marks
                    WHERE reverse_rank=1
                 )
                 SELECT AVG(value) FROM daily_returns WHERE value IS NOT NULL",
                [session_id],
                |row| row.get::<_, Option<f64>>(0),
            )
            .context("calculate average daily return")?;
        Ok(PerformanceMetrics {
            max_drawdown,
            sharpe,
            average_daily_return,
            profit_factor,
            win_rate,
            closed_trades,
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
                SELECT session_id, id, equity,
                    MAX(equity) OVER (
                        PARTITION BY session_id ORDER BY id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS peak_equity
                FROM account_snapshots
            ), stats AS (
                SELECT session_id,
                    MAX(peak_equity) AS peak_equity,
                    MIN(CASE WHEN peak_equity > 0.0 THEN equity / peak_equity - 1.0 ELSE 0.0 END) AS max_drawdown,
                    MAX(id) AS latest_id
                FROM ordered
                GROUP BY session_id
            )
            INSERT OR IGNORE INTO session_performance (
                session_id, peak_equity, max_drawdown, updated_at
            )
            SELECT stats.session_id, stats.peak_equity, stats.max_drawdown, snapshots.captured_at
            FROM stats
            JOIN account_snapshots AS snapshots ON snapshots.id = stats.latest_id;",
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

fn record_equity_point_in(transaction: &Transaction<'_>, snapshot: &AccountSnapshot) -> Result<()> {
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
    let captured_at = timestamp(snapshot.captured_at);
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
    transaction.execute(
        "INSERT INTO session_performance (
            session_id, peak_equity, max_drawdown, updated_at
         ) VALUES (?1, ?2, 0.0, ?3)
         ON CONFLICT(session_id) DO UPDATE SET
            max_drawdown = MIN(
                session_performance.max_drawdown,
                CASE WHEN session_performance.peak_equity > 0.0
                    THEN excluded.peak_equity / session_performance.peak_equity - 1.0
                    ELSE 0.0
                END
            ),
            peak_equity = MAX(session_performance.peak_equity, excluded.peak_equity),
            updated_at = excluded.updated_at",
        params![snapshot.session_id, snapshot.equity, captured_at],
    )?;
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

    use super::Ledger;

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
        let started_at = Utc::now();
        for (offset, equity) in [1_000.0, 980.0, 1_010.0, 1_005.0].into_iter().enumerate() {
            let snapshot = AccountSnapshot {
                session_id: "test".to_owned(),
                captured_at: started_at + Duration::days(offset as i64),
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
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance exists");
        assert!((performance.max_drawdown + 0.01).abs() < 1e-12);
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
    }

    #[test]
    fn profit_factor_and_win_rate_use_closed_execution_pnl() {
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
        assert_eq!(metrics.win_rate, Some(1.0 / 3.0));
        assert_eq!(metrics.profit_factor, Some(3.0));
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
                    (1, 'test', '2026-08-13T00:00:00Z', 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (2, 'test', '2026-08-13T00:01:00Z', 980.0, 980.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
