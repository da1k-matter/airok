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
let latestPeriods = [];
let sessionInitialEquity = 0;
let sessionFees = 0;
let totalEquityPoints = 0;
let nextCurveRefresh = 0;
let selectedChart = 'equity';
const sorting = {
  positions: { key: 'symbol', direction: 'asc' },
  trades: { key: 'executed_at', direction: 'desc' },
};

const terminalPhrases = [
  'Δx · Δp ≥ ħ / 2 — certainty has a cost.',
  '|Ψ⟩ = (|00⟩ + |11⟩) / √2 — distance is not separation.',
  'S = k_B ln Ω — disorder is possibility counted honestly.',
  'We observe the light; dark matter keeps the ledger.',
  'The vacuum is not empty. It is restless.',
  'Quarks are never alone. Perhaps facts are not either.',
  'The arrow of time may be the price of remembering.',
  'Every measurement is a question asked in public.',
  'The universe is unfinished, and so is the observer.',
  'c is constant. Everything else is learning to change.',
];

const typewriter = document.querySelector('#mode-label');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const terminalError = 'Error: dashboard API unreachable.';
let terminalMode = null;
let terminalRun = 0;

const sleep = milliseconds => new Promise(resolve => window.setTimeout(resolve, milliseconds));

async function typePhrase(phrase, run) {
  typewriter.textContent = '';
  for (const character of phrase) {
    if (run !== terminalRun) return false;
    typewriter.textContent += character;
    await sleep(character === ' ' ? 38 : 34 + Math.round(Math.random() * 32));
  }
  return run === terminalRun;
}

async function erasePhrase(run) {
  while (typewriter.textContent.length > 0) {
    if (run !== terminalRun) return false;
    typewriter.textContent = typewriter.textContent.slice(0, -1);
    await sleep(18 + Math.round(Math.random() * 18));
  }
  return run === terminalRun;
}

async function runPhraseCycle(run) {
  let phraseIndex = 0;
  typewriter.textContent = '';
  await sleep(420);
  while (run === terminalRun && terminalMode === 'normal' && !reduceMotion.matches) {
    if (!await typePhrase(terminalPhrases[phraseIndex], run)) return;
    await sleep(3600 + Math.round(Math.random() * 1100));
    if (!await erasePhrase(run)) return;
    await sleep(360);
    phraseIndex = (phraseIndex + 1) % terminalPhrases.length;
  }
}

async function runErrorMessage(run) {
  typewriter.textContent = '';
  await sleep(160);
  await typePhrase(terminalError, run);
}

function restartTerminal() {
  const run = ++terminalRun;
  typewriter.classList.toggle('typewriter-error', terminalMode === 'error');
  if (reduceMotion.matches) {
    typewriter.textContent = terminalMode === 'error' ? terminalError : terminalPhrases[0];
    return;
  }
  if (terminalMode === 'error') runErrorMessage(run);
  else runPhraseCycle(run);
}

function setTerminalMode(mode) {
  if (terminalMode === mode) return;
  terminalMode = mode;
  restartTerminal();
}

reduceMotion.addEventListener('change', restartTerminal);

setTerminalMode('normal');

function openView(viewId) {
  const button = document.querySelector(`.nav-button[data-view="${viewId}"]`);
  document.querySelectorAll('.nav-button').forEach(item => item.classList.toggle('active', item === button));
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === viewId));
  if (viewId === 'chart-view') requestAnimationFrame(drawSelectedChart);
}

document.querySelectorAll('.nav-button').forEach(button => button.addEventListener('click', () => openView(button.dataset.view)));

function selectMetricChart(metric) {
  selectedChart = metric.dataset.chart;
  document.querySelectorAll('.metric').forEach(item => {
    const selected = item === metric;
    item.classList.toggle('selected', selected);
    item.setAttribute('aria-pressed', String(selected));
  });
  openView('chart-view');
  document.querySelector('#chart-view').scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
}

document.querySelectorAll('.metric[data-chart]').forEach(metric => {
  metric.setAttribute('aria-pressed', 'false');
  metric.addEventListener('click', () => selectMetricChart(metric));
  metric.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault(); selectMetricChart(metric);
  });
});

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

const chartDefinitions = {
  equity: { title: 'Equity curve', copy: 'Every stored UTC-minute mark-to-market point. Dense history preserves the extrema while the complete minute series remains in SQLite.', tone: '#d99647', format: money },
  pnl: { title: 'Total P&L', copy: 'Cumulative profit and loss measured from the persisted session starting equity.', tone: '#4fc494', format: signedMoney },
  drawdown: { title: 'Max drawdown', copy: 'Distance from the running equity peak across every stored minute mark.', tone: '#e8716d', format: percent },
  sharpe: { title: 'Rolling Sharpe', copy: 'Annualized rolling Sharpe estimated from minute equity returns.', tone: '#4fc494', format: value => value.toFixed(2) },
  'profit-factor': { title: 'Profit factor', copy: 'Gross positive daily-period return divided by gross negative daily-period return.', tone: '#d99647', format: value => value.toFixed(2), unavailable: true },
  'win-rate': { title: 'Win rate', copy: 'Share of completed daily periods with a positive portfolio return.', tone: '#d99647', format: value => `${(value * 100).toFixed(1)}%`, unavailable: true },
  returns: { title: 'Average daily return', copy: 'Mean of completed UTC-day portfolio returns. The chart uses the same canonical daily periods as Win Rate and Profit Factor.', tone: '#4fc494', format: percent },
  fees: { title: 'Fees paid', copy: 'Cumulative execution fees from the available paper execution reports.', tone: '#d99a4d', format: money },
};

function selectedSeries(kind) {
  if (kind === 'profit-factor' || kind === 'win-rate') return [];
  if (kind === 'fees') {
    const fees = [...latestTrades].filter(row => Number.isFinite(row.fee) && row.fee > 0).sort((a, b) => new Date(a.executed_at) - new Date(b.executed_at));
    let cumulative = Math.max(0, sessionFees - fees.reduce((sum, row) => sum + row.fee, 0));
    return fees.map(row => ({ value: cumulative += row.fee, captured_at: row.executed_at }));
  }
  if (kind === 'returns') return periodReturnRows();
  if (kind === 'drawdown') {
    return latestEquity
      .filter(row => Number.isFinite(Number(row.drawdown)))
      .map(row => ({ value: Number(row.drawdown), captured_at: row.drawdown_at }));
  }
  const values = metricSeries(kind);
  const offset = kind === 'sharpe' ? 1 : 0;
  return values.map((value, index) => ({ value, captured_at: latestEquity[index + offset]?.captured_at }));
}

function drawSeriesChart(rows, definition) {
  const canvas = document.querySelector('#chart'), tip = document.querySelector('#chart-tooltip');
  const { context, W, H, L, R, T, B, plotW, plotH } = chartFrame(canvas);
  if (rows.length < 2) { tip.style.opacity = '0'; return; }
  const values = rows.map(row => row.value), isDrawdown = definition === chartDefinitions.drawdown;
  const low = Math.min(...values), high = Math.max(...values), span = high - low || Math.max(Math.abs(high), 1);
  const pad = span * .12, min = isDrawdown ? Math.min(low - pad, -pad) : low - pad, max = isDrawdown ? Math.max(high + pad, pad) : high + pad, range = max - min;
  const x = index => L + index / Math.max(rows.length - 1, 1) * plotW, y = value => T + (max - value) / range * plotH;
  context.strokeStyle = '#29211d'; context.fillStyle = '#7e7068'; context.font = '11px ui-monospace, monospace'; context.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) { const value = min + range * index / 4, yy = y(value); context.beginPath(); context.moveTo(L, yy); context.lineTo(W - R, yy); context.stroke(); context.textAlign = 'right'; context.fillText(definition.format(value), L - 12, yy + 4); }
  if (isDrawdown) { const baseline = y(0); context.setLineDash([3, 5]); context.strokeStyle = 'rgba(244,235,221,.28)'; context.beginPath(); context.moveTo(L, baseline); context.lineTo(W - R, baseline); context.stroke(); context.setLineDash([]); }
  const gradient = context.createLinearGradient(0, T, 0, T + plotH); gradient.addColorStop(0, `${definition.tone}42`); gradient.addColorStop(1, `${definition.tone}00`);
  const fillBaseline = isDrawdown ? y(0) : T + plotH;
  context.beginPath(); context.moveTo(x(0), fillBaseline); rows.forEach((row, index) => context.lineTo(x(index), y(row.value))); context.lineTo(x(rows.length - 1), fillBaseline); context.closePath(); context.fillStyle = gradient; context.fill();
  context.beginPath(); rows.forEach((row, index) => index ? context.lineTo(x(index), y(row.value)) : context.moveTo(x(index), y(row.value))); context.strokeStyle = definition.tone; context.lineWidth = 2.25; context.stroke();
  context.fillStyle = '#7e7068'; context.textAlign = 'left'; context.fillText(timestamp(rows[0].captured_at), L, H - 15); context.textAlign = 'center'; context.fillText(`${rows.length.toLocaleString()} plotted points`, W / 2, H - 15); context.textAlign = 'right'; context.fillText(timestamp(rows.at(-1).captured_at), W - R, H - 15);
  canvas.onmousemove = event => { const rect = canvas.getBoundingClientRect(), index = Math.max(0, Math.min(rows.length - 1, Math.round(((event.clientX - rect.left - L) / plotW) * (rows.length - 1)))), row = rows[index], cx = x(index), cy = y(row.value); tip.innerHTML = `${timestamp(row.captured_at)}<br><strong style="color:${definition.tone}">${definition.format(row.value)}</strong>`; tip.style.left = `${Math.max(72, Math.min(W - 72, cx)) / W * 100}%`; tip.style.top = `${Math.max(42, cy) / H * 100}%`; tip.style.opacity = '1'; };
  canvas.onmouseleave = () => { tip.style.opacity = '0'; };
}

function drawSelectedChart() {
  const definition = chartDefinitions[selectedChart];
  document.querySelector('#chart-title').textContent = definition.title;
  document.querySelector('#chart-copy').textContent = definition.copy;
  if (selectedChart === 'equity') {
    document.querySelector('#chart-meta').textContent = totalEquityPoints ? `${totalEquityPoints.toLocaleString()} stored points` : 'Awaiting first close';
    drawChart(latestEquity);
    return;
  }
  const rows = selectedSeries(selectedChart);
  const periodMetric = ['returns', 'profit-factor', 'win-rate'].includes(selectedChart);
  document.querySelector('#chart-meta').textContent = rows.length
    ? `${rows.length.toLocaleString()} plotted points`
    : periodMetric ? 'Awaiting completed periods' : 'Awaiting closed trades';
  drawSeriesChart(rows, definition);
}

function metricSeries(kind) {
  if (kind === 'drawdown') {
    return latestEquity
      .map(row => Number(row.drawdown))
      .filter(Number.isFinite);
  }
  const equities = latestEquity.map(row => Number(row.equity)).filter(Number.isFinite);
  if (kind === 'returns') return periodReturnRows().map(row => row.value);
  if (!equities.length) return [];
  if (kind === 'equity') return equities;
  if (kind === 'pnl') return equities.map(value => value - sessionInitialEquity);
  const returns = equities.slice(1).map((value, index) => equities[index] ? value / equities[index] - 1 : 0);
  const rolling = (values, windowSize, mapper) => values.map((_, index) => mapper(values.slice(Math.max(0, index - windowSize + 1), index + 1)));
  if (kind === 'sharpe') return rolling(returns, 60, values => {
    if (values.length < 2) return 0;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    return variance > 0 ? mean / Math.sqrt(variance) * Math.sqrt(365 * 24 * 60) : 0;
  });
  if (kind === 'fees') {
    const fees = [...latestTrades].filter(row => Number.isFinite(row.fee) && row.fee > 0).sort((a, b) => new Date(a.executed_at) - new Date(b.executed_at));
    let cumulative = Math.max(0, sessionFees - fees.reduce((sum, row) => sum + row.fee, 0));
    return fees.map(row => { cumulative += row.fee; return cumulative; });
  }
  return [];
}

function periodReturnRows() {
  return latestPeriods
    .filter(row => Number.isFinite(Number(row.net_return)) && row.period_date)
    .map(row => ({ value: Number(row.net_return), captured_at: row.period_date }));
}

function drawMetricChart(canvas) {
  const values = metricSeries(canvas.dataset.series);
  const { width, height } = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(width * ratio)); canvas.height = Math.max(1, Math.round(height * ratio));
  const context = canvas.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
  const tone = canvas.closest('.metric').dataset.tone;
  const colors = { positive: '#4fc494', negative: '#e8716d', warning: '#d99a4d', neutral: '#d99647' };
  const color = colors[tone] || colors.neutral;
  if (values.length < 2 || canvas.dataset.series === 'empty') {
    context.setLineDash([3, 5]); context.strokeStyle = 'rgba(244,235,221,.12)'; context.lineWidth = 1;
    context.beginPath(); context.moveTo(0, height * .66); context.lineTo(width, height * .66); context.stroke();
    return;
  }
  const isDrawdown = canvas.dataset.series === 'drawdown';
  const low = Math.min(...values), high = Math.max(...values), span = high - low || Math.max(Math.abs(high), 1);
  const pad = span * .12, min = isDrawdown ? Math.min(low - pad, -pad) : low - pad, max = isDrawdown ? Math.max(high + pad, pad) : high + pad, range = max - min, padX = 1, padY = 8;
  const x = index => padX + index / (values.length - 1) * (width - padX * 2);
  const y = value => padY + (max - value) / range * (height - padY * 2);
  const gradient = context.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, `${color}38`); gradient.addColorStop(1, `${color}00`);
  const fillBaseline = isDrawdown ? y(0) : height;
  context.beginPath(); context.moveTo(x(0), fillBaseline); values.forEach((value, index) => context.lineTo(x(index), y(value))); context.lineTo(x(values.length - 1), fillBaseline); context.closePath(); context.fillStyle = gradient; context.fill();
  if (isDrawdown) { const baseline = y(0); context.setLineDash([3, 5]); context.strokeStyle = 'rgba(244,235,221,.22)'; context.lineWidth = 1; context.beginPath(); context.moveTo(0, baseline); context.lineTo(width, baseline); context.stroke(); context.setLineDash([]); }
  context.beginPath(); values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
  context.strokeStyle = color; context.lineWidth = 1.45; context.globalAlpha = .82; context.stroke(); context.globalAlpha = 1;
  const lastX = x(values.length - 1), lastY = y(values.at(-1));
  context.beginPath(); context.arc(lastX, lastY, 2.6, 0, Math.PI * 2); context.fillStyle = color; context.fill();
}

function drawMetricCharts() {
  document.querySelectorAll('.metric-chart').forEach(drawMetricChart);
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
  const profitFactor = metrics.profit_factor_unbounded ? '∞' : Number.isFinite(metrics.profit_factor) ? metrics.profit_factor.toFixed(2) : '—';
  set('#profit-factor', profitFactor, classFor(metrics.profit_factor_unbounded ? 1 : (metrics.profit_factor ?? 1) - 1));
  set('#win-rate', Number.isFinite(metrics.win_rate) ? `${(metrics.win_rate * 100).toFixed(1)}%` : '—', classFor((metrics.win_rate ?? .5) - .5));
  set('#average-daily-return', Number.isFinite(metrics.average_daily_return) ? percent(metrics.average_daily_return) : '—', classFor(metrics.average_daily_return));
  requestAnimationFrame(drawMetricCharts);
}

function renderPositions(rows, executions, equity) {
  latestPositions = rows;
  const latestSlippage = new Map();
  executions.forEach(row => {
    const isFill = row.status === 'filled' || row.status === 'partial';
    if (isFill && Number.isFinite(row.slippage_bps) && !latestSlippage.has(row.symbol)) latestSlippage.set(row.symbol, row.slippage_bps);
  });
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
    if (curve) { latestEquity = curve.points; latestPeriods = curve.periods; latestMetrics = curve.metrics; totalEquityPoints = curve.total_points; nextCurveRefresh = now + 15000; }
    const account = session.account;
    setTerminalMode('normal');
    document.querySelector('#equity').textContent = money(account.equity); document.querySelector('#fees').textContent = money(account.fee_paid); document.querySelector('#model-info').textContent = `Frozen ${session.model.backend.toUpperCase()} ensemble · h${session.model.horizon_days} · ${session.model.seed_count} seeds · cut-off ${session.model.cutoff_date}`;
    sessionInitialEquity = session.session_start_equity_usd; sessionFees = account.fee_paid; currentEquity = account.equity; renderPerformance(currentEquity, sessionInitialEquity, latestMetrics); renderTrades(executions); renderPositions(positions, executions, currentEquity); if (document.querySelector('#chart-view').classList.contains('active')) drawSelectedChart();
  } catch (_) { setTerminalMode('error'); }
}
refresh(); setInterval(refresh, 3000); window.addEventListener('resize', () => { drawMetricCharts(); if (document.querySelector('#chart-view').classList.contains('active')) drawSelectedChart(); });
