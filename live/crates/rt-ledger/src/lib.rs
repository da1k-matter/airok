use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rt_domain::{AccountSnapshot, ExecutionReport, OrderBookSnapshot, PaperState};
use rusqlite::{Connection, OptionalExtension, params};
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
        transaction.commit()?;
        Ok(())
    }

    pub fn record_daily_snapshot(&self, date: &str, snapshot: &AccountSnapshot) -> Result<()> {
        self.connection.execute(
            "INSERT INTO daily_account_snapshots (
                session_id, date, captured_at, equity, gross_notional, net_notional
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(session_id, date) DO UPDATE SET
                captured_at=excluded.captured_at,
                equity=excluded.equity,
                gross_notional=excluded.gross_notional,
                net_notional=excluded.net_notional",
            params![
                snapshot.session_id,
                date,
                timestamp(snapshot.captured_at),
                snapshot.equity,
                snapshot.gross_notional,
                snapshot.net_notional,
            ],
        )?;
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

    pub fn daily_equity_curve(&self, session_id: &str) -> Result<Vec<EquityPoint>> {
        let mut statement = self.connection.prepare(
            "SELECT captured_at, equity, gross_notional, net_notional
             FROM daily_account_snapshots
             WHERE session_id=?1
             ORDER BY date ASC",
        )?;
        let mut values = statement
            .query_map(params![session_id], |row| {
                let captured_at: String = row.get(0)?;
                Ok((captured_at, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|(captured_at, equity, gross_notional, net_notional)| {
                Ok(EquityPoint {
                    captured_at: DateTime::parse_from_rfc3339(&captured_at)
                        .context("parse account snapshot timestamp")?
                        .with_timezone(&Utc),
                    equity,
                    gross_notional,
                    net_notional,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(start) = self.session_start_point(session_id)? {
            values.insert(0, start);
        }
        Ok(values)
    }

    fn session_start_point(&self, session_id: &str) -> Result<Option<EquityPoint>> {
        self.connection
            .query_row(
                "SELECT captured_at, equity, gross_notional, net_notional
                 FROM account_snapshots WHERE session_id=?1 ORDER BY id ASC LIMIT 1",
                [session_id],
                |row| {
                    let captured_at: String = row.get(0)?;
                    Ok((captured_at, row.get(1)?, row.get(2)?, row.get(3)?))
                },
            )
            .optional()?
            .map(|(captured_at, equity, gross_notional, net_notional)| {
                Ok(EquityPoint {
                    captured_at: DateTime::parse_from_rfc3339(&captured_at)
                        .context("parse session start timestamp")?
                        .with_timezone(&Utc),
                    equity,
                    gross_notional,
                    net_notional,
                })
            })
            .transpose()
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

    pub fn daily_max_drawdown(&self, session_id: &str) -> Result<f64> {
        let mut peak = f64::NEG_INFINITY;
        let mut maximum = 0.0_f64;
        for point in self.daily_equity_curve(session_id)? {
            peak = peak.max(point.equity);
            if peak > 0.0 {
                maximum = maximum.min(point.equity / peak - 1.0);
            }
        }
        Ok(maximum)
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
            CREATE TABLE IF NOT EXISTS daily_account_snapshots (
                session_id TEXT NOT NULL,
                date TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                equity REAL NOT NULL,
                gross_notional REAL NOT NULL,
                net_notional REAL NOT NULL,
                PRIMARY KEY(session_id, date)
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
            "INSERT OR IGNORE INTO daily_account_snapshots (
                session_id, date, captured_at, equity, gross_notional, net_notional
            )
            SELECT snapshots.session_id,
                substr(snapshots.captured_at, 1, 10),
                snapshots.captured_at,
                snapshots.equity,
                snapshots.gross_notional,
                snapshots.net_notional
            FROM account_snapshots AS snapshots
            JOIN (
                SELECT session_id, substr(captured_at, 1, 10) AS date, MAX(id) AS id
                FROM account_snapshots
                WHERE substr(captured_at, 1, 10) < date('now')
                GROUP BY session_id, substr(captured_at, 1, 10)
            ) AS final_snapshot ON final_snapshot.id = snapshots.id;",
        )?;
        Ok(())
    }
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use rt_domain::AccountSnapshot;
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
    fn preserves_all_time_drawdown_across_daily_equity_points() {
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
            if offset > 0 {
                ledger
                    .record_daily_snapshot(
                        &snapshot.captured_at.date_naive().to_string(),
                        &snapshot,
                    )
                    .expect("daily snapshot persists");
            }
        }

        let chart = ledger.daily_equity_curve("test").expect("chart reads");
        let performance = ledger
            .session_performance("test")
            .expect("performance reads")
            .expect("performance exists");
        let starting_equity = ledger
            .session_start_equity("test")
            .expect("starting equity reads")
            .expect("starting equity exists");

        assert_eq!(chart.len(), 4);
        assert_eq!(starting_equity, 1_000.0);
        assert!((performance.max_drawdown + 0.02).abs() < 1e-12);
        assert!(
            (ledger
                .daily_max_drawdown("test")
                .expect("daily drawdown reads")
                + 0.02)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn backfills_all_time_drawdown_for_an_existing_ledger() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ranktrend-ledger-{suffix}.sqlite3"));
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
