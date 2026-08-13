const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
const quantity = new Intl.NumberFormat('en-US', { maximumFractionDigits: 5 });
const money = value => Number.isFinite(value) ? currency.format(value) : '—';
const signedMoney = value => Number.isFinite(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${currency.format(Math.abs(value))}` : '—';
const priceFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 12 });
const price = value => Number.isFinite(value) ? priceFormat.format(Number(value)) : '—';
const slippage = value => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${Number(value).toFixed(2)} bps` : '—';
const classFor = value => value > 0 ? 'pos' : value < 0 ? 'neg' : '';
const escapeHtml = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
const timestamp = value => value ? new Date(value).toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }) : '—';
const percent = value => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%` : '—';
let latestEquity = [];
let latestPositions = [];
let latestTrades = [];
let currentEquity = 0;
let latestMetrics = {};
let totalEquityPoints = 0;
let nextCurveRefresh = 0;
const sorting = {
  positions: { key: 'symbol', direction: 'asc' },
  trades: { key: 'executed_at', direction: 'desc' },
};

document.querySelectorAll('.nav-button').forEach(button => button.addEventListener('click', () => {
  const viewId = button.dataset.view;
  document.querySelectorAll('.nav-button').forEach(item => item.classList.toggle('active', item === button));
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === viewId));
  if (viewId === 'equity-view') requestAnimationFrame(() => drawChart(latestEquity));
}));

document.querySelectorAll('.sort-button').forEach(button => button.addEventListener('click', () => {
  const table = button.dataset.table, state = sorting[table], key = button.dataset.sortKey;
  state.direction = state.key === key && state.direction === 'asc' ? 'desc' : 'asc';
  state.key = key;
  if (table === 'positions') renderPositions(latestPositions, latestTrades, currentEquity);
  if (table === 'trades') renderTrades(latestTrades);
}));

function chartFrame(canvas) {
  const { width, height } = canvas.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(width, 1), H = Math.max(height, 1), L = W * .075, R = W * .03, T = H * .08, B = H * .13;
  canvas.width = Math.round(W * ratio); canvas.height = Math.round(H * ratio);
  const context = canvas.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, W, H);
  return { context, W, H, L, R, T, B, plotW: W - L - R, plotH: H - T - B };
}

function drawChart(rows) {
  const canvas = document.querySelector('#chart'), tip = document.querySelector('#chart-tooltip');
  const { context, W, H, L, R, T, B, plotW, plotH } = chartFrame(canvas);
  if (!rows.length) { tip.style.opacity = '0'; return; }
  const lows = rows.map(row => row.low), highs = rows.map(row => row.high), initial = rows[0].equity;
  const low = Math.min(...lows), high = Math.max(...highs), pad = Math.max((high - low) * .12, Math.abs(initial) * .0025), min = low - pad, max = high + pad, range = max - min;
  const x = index => L + index / Math.max(rows.length - 1, 1) * plotW, y = value => T + (max - value) / range * plotH;
  context.strokeStyle = '#29211d'; context.fillStyle = '#7e7068'; context.font = '11px ui-monospace, monospace'; context.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) { const value = min + range * index / 4, yy = y(value); context.beginPath(); context.moveTo(L, yy); context.lineTo(W - R, yy); context.stroke(); context.textAlign = 'right'; context.fillText(money(value), L - 12, yy + 4); }
  context.strokeStyle = 'rgba(217, 150, 71, .34)'; context.lineWidth = 1;
  rows.forEach((row, index) => { if (row.high !== row.low) { context.beginPath(); context.moveTo(x(index), y(row.low)); context.lineTo(x(index), y(row.high)); context.stroke(); } });
  const gradient = context.createLinearGradient(0, T, 0, T + plotH); gradient.addColorStop(0, 'rgba(217,150,71,.28)'); gradient.addColorStop(1, 'rgba(217,150,71,0)');
  context.beginPath(); context.moveTo(x(0), T + plotH); rows.forEach((row, index) => context.lineTo(x(index), y(row.equity))); context.lineTo(x(rows.length - 1), T + plotH); context.closePath(); context.fillStyle = gradient; context.fill();
  context.beginPath(); rows.forEach((row, index) => index ? context.lineTo(x(index), y(row.equity)) : context.moveTo(x(index), y(row.equity))); context.strokeStyle = '#d99647'; context.lineWidth = 2.25; context.stroke();
  context.fillStyle = '#7e7068'; context.textAlign = 'left'; context.fillText(timestamp(rows[0].captured_at), L, H - 15); context.textAlign = 'center'; context.fillText(`${totalEquityPoints.toLocaleString()} stored minute points`, W / 2, H - 15); context.textAlign = 'right'; context.fillText(timestamp(rows.at(-1).captured_at), W - R, H - 15);
  canvas.onmousemove = event => { const rect = canvas.getBoundingClientRect(), index = Math.max(0, Math.min(rows.length - 1, Math.round(((event.clientX - rect.left - L) / plotW) * (rows.length - 1)))), row = rows[index], cx = x(index), cy = y(row.equity); tip.innerHTML = `${timestamp(row.captured_at)}<br><strong>${money(row.equity)}</strong>${row.low !== row.high ? `<br>${money(row.low)} — ${money(row.high)}` : ''}`; tip.style.left = `${Math.max(72, Math.min(W - 72, cx)) / W * 100}%`; tip.style.top = `${Math.max(42, cy) / H * 100}%`; tip.style.opacity = '1'; };
  canvas.onmouseleave = () => { tip.style.opacity = '0'; };
}

function renderSortIndicators(table) {
  const state = sorting[table];
  document.querySelectorAll(`.sort-button[data-table="${table}"]`).forEach(button => {
    const active = button.dataset.sortKey === state.key;
    button.dataset.direction = active ? state.direction : '';
    button.setAttribute('aria-sort', active ? (state.direction === 'asc' ? 'ascending' : 'descending') : 'none');
  });
}

function sortRows(rows, table) {
  const { key, direction } = sorting[table], multiplier = direction === 'asc' ? 1 : -1;
  return [...rows].sort((left, right) => {
    const a = left[key], b = right[key];
    if (typeof a === 'string' || typeof b === 'string') return multiplier * String(a ?? '').localeCompare(String(b ?? ''));
    return multiplier * ((Number(a) || 0) - (Number(b) || 0));
  });
}

function renderPerformance(equity, initialEquity, metrics) {
  const totalPnl = equity - initialEquity, totalReturn = initialEquity ? totalPnl / initialEquity : null;
  const set = (id, value, className = '') => { const element = document.querySelector(id); element.textContent = value; element.className = className; };
  set('#total-pnl', Number.isFinite(totalPnl) ? signedMoney(totalPnl) : '—', classFor(totalPnl));
  set('#total-pnl-return', Number.isFinite(totalReturn) ? `${percent(totalReturn)} since inception` : '—', classFor(totalReturn));
  set('#max-drawdown', Number.isFinite(metrics.max_drawdown) ? percent(metrics.max_drawdown) : '—', classFor(metrics.max_drawdown));
  set('#sharpe', Number.isFinite(metrics.sharpe) ? metrics.sharpe.toFixed(2) : '—', classFor(metrics.sharpe));
  set('#profit-factor', Number.isFinite(metrics.profit_factor) ? metrics.profit_factor.toFixed(2) : '—', classFor((metrics.profit_factor ?? 1) - 1));
  set('#win-rate', Number.isFinite(metrics.win_rate) ? `${(metrics.win_rate * 100).toFixed(1)}%` : '—', classFor((metrics.win_rate ?? .5) - .5));
  set('#average-return', Number.isFinite(metrics.average_return) ? percent(metrics.average_return) : '—', classFor(metrics.average_return));
}

function renderPositions(rows, executions, equity) {
  latestPositions = rows;
  const latestSlippage = new Map();
  executions.forEach(row => { if (!latestSlippage.has(row.symbol)) latestSlippage.set(row.symbol, row.slippage_bps); });
  const preparedRows = rows.map(row => ({ ...row, lot_pct: equity > 0 ? row.notional / equity * 100 : 0, slippage_bps: latestSlippage.get(row.symbol) }));
  const sortedRows = sortRows(preparedRows, 'positions');
  document.querySelector('#position-count').textContent = rows.length.toLocaleString(); document.querySelector('#positions-meta').textContent = rows.length ? `${rows.length} active instruments` : 'No active positions';
  document.querySelector('#positions-body').innerHTML = sortedRows.map(row => `<tr><td class="symbol">${escapeHtml(row.symbol)}</td><td><span class="side side-${escapeHtml(row.side)}">${escapeHtml(row.side)}</span></td><td class="num">${row.lot_pct.toFixed(2)}%</td><td class="num">${money(row.notional)}</td><td class="num">${price(row.entry_price)}</td><td class="num">${price(row.mark_price)}</td><td class="num ${classFor(row.unrealized_pnl)}">${signedMoney(row.unrealized_pnl)}</td><td class="num warn">${slippage(row.slippage_bps)}</td><td>${timestamp(row.opened_at)}</td></tr>`).join('');
  document.querySelector('#positions-empty').style.display = rows.length ? 'none' : 'block';
  renderSortIndicators('positions');
}

function renderTrades(rows) {
  latestTrades = rows;
  const sortedRows = sortRows(rows, 'trades');
  document.querySelector('#trade-count').textContent = rows.length.toLocaleString();
  document.querySelector('#trades-body').innerHTML = sortedRows.map(row => `<tr><td class="symbol">${escapeHtml(row.symbol)}</td><td><span class="side side-${escapeHtml(row.side)}">${escapeHtml(row.side)}</span></td><td class="num">${quantity.format(row.filled_quantity)}</td><td class="num">${price(row.vwap)}</td><td class="num">${money(row.notional)}</td><td class="num warn">${money(row.fee)}</td><td class="num warn">${slippage(row.slippage_bps)}</td><td class="${row.status === 'filled' ? 'pos' : row.status === 'rejected' ? 'neg' : 'warn'}">${escapeHtml(row.status)}</td><td>${timestamp(row.executed_at)}</td><td>${escapeHtml(row.decision_id)}</td></tr>`).join('');
  document.querySelector('#trades-empty').style.display = rows.length ? 'none' : 'block';
  renderSortIndicators('trades');
}

async function refresh() {
  try {
    const now = Date.now(), shouldRefreshCurve = now >= nextCurveRefresh;
    const curveUrl = `/api/equity?max_points=${Math.max(400, Math.min(5000, Math.ceil(window.innerWidth * 2)))}`;
    const urls = ['/api/session', '/api/positions', '/api/executions'];
    if (shouldRefreshCurve) urls.push(curveUrl);
    const payloads = await Promise.all(urls.map(url => fetch(url, { cache: 'no-store' }).then(response => { if (!response.ok) throw new Error(`${url} unavailable`); return response.json(); })));
    const [session, positions, executions, curve] = payloads;
    if (curve) { latestEquity = curve.points; latestMetrics = curve.metrics; totalEquityPoints = curve.total_points; nextCurveRefresh = now + 15000; }
    const account = session.account, replay = session.status === 'historical_replay';
    document.querySelector('#live-dot').classList.remove('offline'); document.querySelector('#status').textContent = session.status.replaceAll('_', ' ').toUpperCase(); document.querySelector('#last-updated').textContent = '3s refresh'; document.querySelector('#mode-label').textContent = replay ? 'Historical OOS replay / frozen model' : 'Cross-sectional crypto ranking';
    document.querySelector('#equity').textContent = money(account.equity); document.querySelector('#fees').textContent = money(account.fee_paid); document.querySelector('#equity-meta').textContent = session.last_decision_date || 'Awaiting first close'; document.querySelector('#model-info').textContent = `Frozen ${session.model.backend.toUpperCase()} ensemble · h${session.model.horizon_days} · ${session.model.seed_count} seeds · cut-off ${session.model.cutoff_date}`;
    currentEquity = account.equity; renderPerformance(currentEquity, session.session_start_equity_usd, latestMetrics); renderTrades(executions); renderPositions(positions, executions, currentEquity); if (curve && document.querySelector('#equity-view').classList.contains('active')) drawChart(latestEquity);
  } catch (_) { document.querySelector('#live-dot').classList.add('offline'); document.querySelector('#status').textContent = 'API UNAVAILABLE'; document.querySelector('#last-updated').textContent = 'Retrying'; }
}
refresh(); setInterval(refresh, 3000); window.addEventListener('resize', () => { if (document.querySelector('#equity-view').classList.contains('active')) drawChart(latestEquity); });
