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

function chartFrame(svg) {
  const { width, height } = svg.getBoundingClientRect();
  const W = Math.max(width, 1), H = Math.max(height, 1), L = W * .075, R = W * .03, T = H * .08, B = H * .13;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
  return { W, H, L, R, T, B, plotW: W - L - R, plotH: H - T - B };
}

function drawChart(rows) {
  const svg = document.querySelector('#chart'), tip = document.querySelector('#chart-tooltip');
  const { W, H, L, R, T, B, plotW, plotH } = chartFrame(svg);
  if (!rows.length) { tip.style.opacity = '0'; return; }
  const values = rows.map(row => row.equity), initial = values[0], current = values.at(-1);
  if (rows.length < 2 || new Set(rows.map(row => row.captured_at.slice(0, 10))).size < 2) {
    const y = T + plotH / 2, lx = L + 120, rx = W - R - 120, delta = current - initial;
    svg.innerHTML = `<line class="chart-grid" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/><text class="chart-axis-label" x="${L}" y="${T + 24}">BOOTSTRAP COMPLETE — FIRST DAILY MARK PENDING</text><text class="chart-axis-label" x="${lx}" y="${y - 26}" text-anchor="middle">STARTING EQUITY</text><text fill="#f4ede4" x="${lx}" y="${y + 7}" text-anchor="middle" font-size="24" font-weight="700">${money(initial)}</text><line class="chart-line" x1="${lx + 105}" y1="${y}" x2="${rx - 105}" y2="${y}"/><circle class="chart-marker" cx="${lx + 105}" cy="${y}" r="4"/><circle class="chart-marker" cx="${rx - 105}" cy="${y}" r="4"/><text class="chart-axis-label" x="${rx}" y="${y - 26}" text-anchor="middle">CURRENT AFTER ENTRY FEES</text><text fill="#d99647" x="${rx}" y="${y + 7}" text-anchor="middle" font-size="24" font-weight="700">${money(current)}</text><text class="chart-axis-label" x="${W / 2}" y="${y + 38}" text-anchor="middle">${signedMoney(delta)} · ${(initial ? delta / initial * 100 : 0).toFixed(3)}%</text><text class="chart-axis-label" x="${L}" y="${H - 15}">${rows[0].captured_at.slice(0, 10)}</text><text class="chart-axis-label" x="${W - R}" y="${H - 15}" text-anchor="end">NEXT CONFIRMED 1D CLOSE ADDS THE FIRST CURVE POINT</text>`;
    tip.style.opacity = '0'; return;
  }
  const low = Math.min(...values), high = Math.max(...values), pad = Math.max((high - low) * .12, Math.abs(initial) * .0025), min = low - pad, max = high + pad, range = max - min;
  const x = index => L + index / (rows.length - 1) * plotW, y = value => T + (max - value) / range * plotH;
  let markup = '<defs><linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#d99647" stop-opacity=".28"/><stop offset="100%" stop-color="#d99647" stop-opacity="0"/></linearGradient></defs>';
  for (let index = 0; index < 5; index += 1) { const value = min + range * index / 4, yy = y(value); markup += `<line class="chart-grid" x1="${L}" y1="${yy}" x2="${W - R}" y2="${yy}"/><text class="chart-axis-label" x="${L - 12}" y="${yy + 4}" text-anchor="end">${money(value)}</text>`; }
  for (const index of [...new Set([0, Math.floor((rows.length - 1) / 2), rows.length - 1])]) { const anchor = index === 0 ? 'start' : index === rows.length - 1 ? 'end' : 'middle'; markup += `<text class="chart-axis-label" x="${x(index)}" y="${H - 15}" text-anchor="${anchor}">${rows[index].captured_at.slice(0, 10)}</text>`; }
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(' ');
  svg.innerHTML = `${markup}<polygon class="chart-area" points="${L},${T + plotH} ${points} ${W - R},${T + plotH}"/><polyline class="chart-line" points="${points}"/><circle class="chart-marker" cx="${x(0)}" cy="${y(values[0])}" r="4"/><circle class="chart-marker" cx="${x(values.length - 1)}" cy="${y(values.at(-1))}" r="4"/><line id="chart-crosshair" class="chart-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${T + plotH}" opacity="0"/><circle id="chart-hover-dot" class="chart-marker" cx="${L}" cy="${T}" r="4" opacity="0"/>`;
  svg.onmousemove = event => { const rect = svg.getBoundingClientRect(), raw = (event.clientX - rect.left) / rect.width, index = Math.max(0, Math.min(rows.length - 1, Math.round((raw * W - L) / plotW))), row = rows[index], cx = x(index), cy = y(row.equity); const crosshair = svg.querySelector('#chart-crosshair'), dot = svg.querySelector('#chart-hover-dot'); crosshair.setAttribute('x1', cx); crosshair.setAttribute('x2', cx); crosshair.setAttribute('opacity', '1'); dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('opacity', '1'); tip.innerHTML = `${row.captured_at.slice(0, 10)}<br><strong>${money(row.equity)}</strong>`; tip.style.left = `${Math.max(72, Math.min(W - 72, cx)) / W * 100}%`; tip.style.top = `${Math.max(42, cy) / H * 100}%`; tip.style.opacity = '1'; };
  svg.onmouseleave = () => { svg.querySelector('#chart-crosshair').setAttribute('opacity', '0'); svg.querySelector('#chart-hover-dot').setAttribute('opacity', '0'); tip.style.opacity = '0'; };
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

function closedDailyEquity(rows) {
  const today = new Date().toISOString().slice(0, 10), byDate = new Map();
  rows.forEach(row => {
    const date = row.captured_at.slice(0, 10);
    if (date < today) byDate.set(date, row.equity);
  });
  return [...byDate.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, equity]) => equity);
}

function performanceMetrics(rows, currentEquity) {
  if (!rows.length) return {};
  const initial = rows[0].equity, totalPnl = currentEquity - initial, totalReturn = initial ? totalPnl / initial : null;
  let peak = -Infinity, maxDrawdown = 0;
  rows.forEach(row => { peak = Math.max(peak, row.equity); if (peak > 0) maxDrawdown = Math.min(maxDrawdown, row.equity / peak - 1); });
  const dailyEquity = closedDailyEquity(rows), returns = dailyEquity.slice(1).map((value, index) => value / dailyEquity[index] - 1).filter(Number.isFinite);
  if (returns.length < 2) return { totalPnl, totalReturn, maxDrawdown };
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1);
  const losses = returns.filter(value => value < 0), gains = returns.filter(value => value > 0);
  return {
    totalPnl, totalReturn, maxDrawdown,
    sharpe: variance > 0 ? mean / Math.sqrt(variance) * Math.sqrt(365) : null,
    profitFactor: losses.length && gains.length ? gains.reduce((sum, value) => sum + value, 0) / Math.abs(losses.reduce((sum, value) => sum + value, 0)) : null,
    winRate: gains.length + losses.length ? gains.length / (gains.length + losses.length) : null,
    averageReturn: mean,
  };
}

function renderPerformance(rows, equity) {
  const metrics = performanceMetrics(rows, equity);
  const set = (id, value, className = '') => { const element = document.querySelector(id); element.textContent = value; element.className = className; };
  set('#total-pnl', Number.isFinite(metrics.totalPnl) ? `${signedMoney(metrics.totalPnl)} · ${percent(metrics.totalReturn)}` : '—', classFor(metrics.totalPnl));
  set('#max-drawdown', Number.isFinite(metrics.maxDrawdown) ? percent(metrics.maxDrawdown) : '—', metrics.maxDrawdown < 0 ? 'neg' : '');
  set('#sharpe', Number.isFinite(metrics.sharpe) ? metrics.sharpe.toFixed(2) : '—', classFor(metrics.sharpe));
  set('#profit-factor', Number.isFinite(metrics.profitFactor) ? metrics.profitFactor.toFixed(2) : '—', classFor((metrics.profitFactor ?? 1) - 1));
  set('#win-rate', Number.isFinite(metrics.winRate) ? `${(metrics.winRate * 100).toFixed(1)}%` : '—', classFor((metrics.winRate ?? .5) - .5));
  set('#average-return', percent(metrics.averageReturn), classFor(metrics.averageReturn));
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
    const [session, positions, equity, executions] = await Promise.all(['/api/session', '/api/positions', '/api/equity', '/api/executions'].map(url => fetch(url, { cache: 'no-store' }).then(response => { if (!response.ok) throw new Error(`${url} unavailable`); return response.json(); })));
    const account = session.account, replay = session.status === 'historical_replay';
    document.querySelector('#live-dot').classList.remove('offline'); document.querySelector('#status').textContent = session.status.replaceAll('_', ' ').toUpperCase(); document.querySelector('#last-updated').textContent = '3s refresh'; document.querySelector('#mode-label').textContent = replay ? 'Historical OOS replay / frozen model' : 'Cross-sectional crypto ranking';
    document.querySelector('#equity').textContent = money(account.equity); document.querySelector('#fees').textContent = money(account.fee_paid); document.querySelector('#equity-meta').textContent = session.last_decision_date || 'Awaiting first close'; document.querySelector('#model-info').textContent = `Frozen ${session.model.backend.toUpperCase()} ensemble · h${session.model.horizon_days} · ${session.model.seed_count} seeds · cut-off ${session.model.cutoff_date}`;
    currentEquity = account.equity; latestEquity = equity; renderPerformance(equity, currentEquity); renderTrades(executions); renderPositions(positions, executions, currentEquity); if (document.querySelector('#equity-view').classList.contains('active')) drawChart(equity);
  } catch (_) { document.querySelector('#live-dot').classList.add('offline'); document.querySelector('#status').textContent = 'API UNAVAILABLE'; document.querySelector('#last-updated').textContent = 'Retrying'; }
}
refresh(); setInterval(refresh, 3000); window.addEventListener('resize', () => { if (document.querySelector('#equity-view').classList.contains('active')) drawChart(latestEquity); });
