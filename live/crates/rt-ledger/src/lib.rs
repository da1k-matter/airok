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
        self.connection.execute(
            "INSERT INTO account_snapshots (
                session_id, captured_at, cash, equity, realized_pnl, unrealized_pnl,
                gross_notional, net_notional, fee_paid
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                snapshot.session_id,
                timestamp(snapshot.captured_at),
                snapshot.cash,
                snapshot.equity,
                snapshot.realized_pnl,
                snapshot.unrealized_pnl,
                snapshot.gross_notional,
                snapshot.net_notional,
                snapshot.fee_paid,
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

    pub fn equity_curve(&self, session_id: &str, limit: usize) -> Result<Vec<EquityPoint>> {
        let mut statement = self.connection.prepare(
            "SELECT captured_at, equity, gross_notional, net_notional
             FROM account_snapshots WHERE session_id=?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let mut values = statement
            .query_map(params![session_id, limit as i64], |row| {
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
        values.reverse();
        Ok(values)
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
        Ok(())
    }
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rt_domain::AccountSnapshot;

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
}
