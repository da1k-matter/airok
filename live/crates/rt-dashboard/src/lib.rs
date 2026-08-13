use std::cell::RefCell;

use wasm_bindgen::{JsCast, JsValue, closure::Closure, prelude::wasm_bindgen};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, Response, Window};

thread_local! {
    static POLLER: RefCell<Option<i32>> = const { RefCell::new(None) };
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    element("app")?.set_inner_html(DASHBOARD_SHELL);
    refresh();
    let callback = Closure::<dyn FnMut()>::new(refresh);
    let id = window()?.set_interval_with_callback_and_timeout_and_arguments_0(
        callback.as_ref().unchecked_ref(),
        10_000,
    )?;
    callback.forget();
    POLLER.with(|poller| *poller.borrow_mut() = Some(id));
    Ok(())
}

#[wasm_bindgen]
pub fn refresh() {
    spawn_local(async {
        if let Err(error) = refresh_inner().await {
            set_text("error", &format!("Dashboard refresh failed: {error:?}"));
        }
    });
}

async fn refresh_inner() -> Result<(), JsValue> {
    let session = fetch_json("/api/session").await?;
    let account = field(&session, "account")?;
    set_text(
        "status",
        &string(&session, "status")?.replace('_', " ").to_uppercase(),
    );
    set_text("equity", &money(number(&account, "equity")?));
    set_text("cash", &money(number(&account, "cash")?));
    set_text("gross", &money(number(&account, "gross_notional")?));
    set_text("fees", &money(number(&account, "fee_paid")?));
    set_text("net", &money(number(&account, "net_notional")?));
    set_text("detail", &string(&session, "detail")?);
    set_text(
        "decision",
        &optional_string(&session, "last_decision_date")?.unwrap_or_else(|| "—".to_owned()),
    );
    set_text(
        "error",
        &optional_string(&session, "last_error")?.unwrap_or_else(|| "—".to_owned()),
    );

    let positions = fetch_json("/api/positions").await?;
    render_positions(&positions)?;
    let executions = fetch_json("/api/executions").await?;
    render_executions(&executions)?;
    let equity = fetch_json("/api/equity").await?;
    draw_equity(&equity)?;
    Ok(())
}

async fn fetch_json(path: &str) -> Result<JsValue, JsValue> {
    let response = JsFuture::from(window()?.fetch_with_str(path)).await?;
    let response: Response = response.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!(
            "{path} returned HTTP {}",
            response.status()
        )));
    }
    JsFuture::from(response.json()?).await
}

fn render_positions(values: &JsValue) -> Result<(), JsValue> {
    let rows = js_sys::Array::from(values);
    let mut output = String::new();
    for value in rows.iter() {
        let quantity = number(&value, "quantity")?;
        let mark = number(&value, "mark_price")?;
        let pnl = number(&value, "unrealized_pnl")?;
        let side = if quantity >= 0.0 { "LONG" } else { "SHORT" };
        let class = if quantity >= 0.0 { "pos" } else { "neg" };
        let pnl_class = if pnl >= 0.0 { "pos" } else { "neg" };
        output.push_str(&format!(
            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td class=\"{}\">{}</td></tr>",
            escape(&string(&value, "symbol")?),
            class,
            side,
            money((quantity * mark).abs()),
            pnl_class,
            money(pnl)
        ));
    }
    if output.is_empty() {
        output.push_str("<tr><td colspan=\"4\">No positions</td></tr>");
    }
    element("positions")?.set_inner_html(&output);
    Ok(())
}

fn render_executions(values: &JsValue) -> Result<(), JsValue> {
    let rows = js_sys::Array::from(values);
    let mut output = String::new();
    for value in rows.iter().take(8) {
        let vwap = optional_number(&value, "vwap")?
            .map(money)
            .unwrap_or_else(|| "—".to_owned());
        let slippage = optional_number(&value, "slippage_bps")?
            .map(|value| format!("{value:.1} bps"))
            .unwrap_or_else(|| "—".to_owned());
        output.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape(&string(&value, "executed_at")?),
            escape(&string(&value, "symbol")?),
            vwap,
            slippage
        ));
    }
    if output.is_empty() {
        output.push_str("<tr><td colspan=\"4\">No executions</td></tr>");
    }
    element("trades")?.set_inner_html(&output);
    Ok(())
}

fn draw_equity(values: &JsValue) -> Result<(), JsValue> {
    let points = js_sys::Array::from(values);
    let canvas: HtmlCanvasElement = element("curve")?.dyn_into()?;
    let context: CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("missing canvas context"))?
        .dyn_into()?;
    let width = canvas.width() as f64;
    let height = canvas.height() as f64;
    context.set_fill_style_str("#111619");
    context.fill_rect(0.0, 0.0, width, height);
    if points.length() == 0 {
        return Ok(());
    }
    let values = points
        .iter()
        .map(|value| number(&value, "equity"))
        .collect::<Result<Vec<_>, _>>()?;
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (high - low).max(0.01);
    context.set_stroke_style_str("#77b99b");
    context.set_line_width(2.0);
    context.begin_path();
    for (index, value) in values.iter().enumerate() {
        let x = index as f64 * width / (values.len().saturating_sub(1).max(1) as f64);
        let y = height - 16.0 - (value - low) / span * (height - 32.0);
        if index == 0 {
            context.move_to(x, y);
        } else {
            context.line_to(x, y);
        }
    }
    context.stroke();
    Ok(())
}

fn window() -> Result<Window, JsValue> {
    web_sys::window().ok_or_else(|| JsValue::from_str("window unavailable"))
}
fn document() -> Result<Document, JsValue> {
    window()?
        .document()
        .ok_or_else(|| JsValue::from_str("document unavailable"))
}
fn element(id: &str) -> Result<web_sys::Element, JsValue> {
    document()?
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(id))
}
fn set_text(id: &str, value: &str) {
    if let Ok(node) = element(id) {
        node.set_text_content(Some(value));
    }
}
fn field(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    js_sys::Reflect::get(value, &JsValue::from_str(name))
}
fn string(value: &JsValue, name: &str) -> Result<String, JsValue> {
    field(value, name)?
        .as_string()
        .ok_or_else(|| JsValue::from_str(name))
}
fn optional_string(value: &JsValue, name: &str) -> Result<Option<String>, JsValue> {
    Ok(field(value, name)?.as_string())
}
fn number(value: &JsValue, name: &str) -> Result<f64, JsValue> {
    field(value, name)?
        .as_f64()
        .ok_or_else(|| JsValue::from_str(name))
}
fn optional_number(value: &JsValue, name: &str) -> Result<Option<f64>, JsValue> {
    Ok(field(value, name)?.as_f64())
}
fn money(value: f64) -> String {
    format!("${value:.2}")
}
fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const DASHBOARD_SHELL: &str = r#"
<style>
:root{color-scheme:dark;--b:#090c0e;--s:#111619;--r:#29343a;--t:#edf2f3;--m:#91a0a8;--a:#77b99b;--n:#de8278}
*{box-sizing:border-box}body{margin:0;background:var(--b);color:var(--t);font:14px Inter,system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:30px}header{display:flex;justify-content:space-between;border-bottom:1px solid var(--r);padding-bottom:18px;letter-spacing:.14em;font-weight:700}
.hero{display:grid;grid-template-columns:1.5fr 1fr;gap:42px;padding:46px 0;border-bottom:1px solid var(--r)}h1{font-size:clamp(48px,8vw,108px);letter-spacing:-.07em;margin:10px 0}
.label{color:var(--m);font-size:11px;letter-spacing:.13em;text-transform:uppercase}.line{display:flex;justify-content:space-between;border-top:1px solid var(--r);padding:14px 0}.good,.pos{color:var(--a)}.neg{color:var(--n)}
.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:42px;padding-top:42px}.panel{min-width:0}.panel h2{font-size:15px;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin:0 0 16px}canvas{width:100%;height:230px;background:var(--s);border:1px solid var(--r)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;border-top:1px solid var(--r);padding:11px 5px}th{color:var(--m);font-size:10px;text-transform:uppercase;letter-spacing:.1em}th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap;color:var(--m);font:12px ui-monospace,monospace}
@media(max-width:850px){main{padding:20px}.hero,.grid{grid-template-columns:1fr;gap:25px}}
</style>
<main><header><span>RANKTREND / PAPER</span><span id="status" class="good">CONNECTING</span></header>
<section class="hero"><div><div class="label">Portfolio equity</div><h1 id="equity">—</h1><canvas id="curve" width="900" height="230"></canvas><div class="line"><span>Cash</span><strong id="cash">—</strong></div><div class="line"><span>Gross exposure</span><strong id="gross">—</strong></div><div class="line"><span>Fees paid</span><strong id="fees">—</strong></div></div>
<div><div class="label">Live process</div><p id="detail">Loading runtime state…</p><div class="line"><span>Last decision</span><strong id="decision">—</strong></div><div class="line"><span>Net exposure</span><strong id="net">—</strong></div><div class="label">Last error</div><pre id="error">—</pre></div></section>
<section class="grid"><div class="panel"><h2>Current positions</h2><table><thead><tr><th>Contract</th><th>Side</th><th>Notional</th><th>Unrealized</th></tr></thead><tbody id="positions"></tbody></table></div><div class="panel"><h2>Trade tape</h2><table><thead><tr><th>Time</th><th>Contract</th><th>Fill</th><th>Slip</th></tr></thead><tbody id="trades"></tbody></table></div></section></main>
"#;
