const money = value => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value ?? 0);
const number = value => new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value ?? 0);
const positionSort = { key: 'symbol', direction: 'asc' };
let latestPositions = [];

function chartFrame(svg) {
  const { width, height } = svg.getBoundingClientRect();
  const W = Math.max(width, 1), H = Math.max(height, 1);
  const L = W * .072, R = W * .028, T = H * .075, B = H * .13125;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.innerHTML = '';
  return { W, H, L, R, T, B, plotW: W - L - R, plotH: H - T - B };
}

function drawBootstrapState(rows) {
  const svg = document.querySelector('#chart');
  const tip = document.querySelector('#chart-tooltip');
  const { W, H, L, R, T, B, plotH } = chartFrame(svg);
  const initial = rows[0].equity;
  const current = rows.at(-1).equity;
  const delta = current - initial;
  const change = initial ? delta / initial : 0;
  const leftX = L + 74;
  const rightX = W - R - 74;
  const baseY = T + plotH / 2;
  svg.innerHTML = `<line class="chart-grid" x1="${L}" y1="${baseY}" x2="${W - R}" y2="${baseY}"/>
    <text class="chart-axis-label" x="${L}" y="${T + 24}">BOOTSTRAP COMPLETE — FIRST DAILY MARK PENDING</text>
    <text class="chart-axis-label" x="${leftX}" y="${baseY - 26}" text-anchor="middle">STARTING EQUITY</text>
    <text fill="#f2f0ea" x="${leftX}" y="${baseY + 7}" text-anchor="middle" font-size="24" font-weight="700">${money(initial)}</text>
    <line class="chart-line" x1="${leftX + 105}" y1="${baseY}" x2="${rightX - 105}" y2="${baseY}"/>
    <circle class="chart-marker" cx="${leftX + 105}" cy="${baseY}" r="4"/>
    <circle class="chart-marker" cx="${rightX - 105}" cy="${baseY}" r="4"/>
    <text class="chart-axis-label" x="${rightX}" y="${baseY - 26}" text-anchor="middle">CURRENT AFTER ENTRY FEES</text>
    <text fill="#f3bd61" x="${rightX}" y="${baseY + 7}" text-anchor="middle" font-size="24" font-weight="700">${money(current)}</text>
    <text class="chart-axis-label" x="${W / 2}" y="${baseY + 38}" text-anchor="middle">${delta >= 0 ? '+' : ''}${money(delta)} · ${(change * 100).toFixed(3)}%</text>
    <text class="chart-axis-label" x="${L}" y="${H - 15}">${rows[0].captured_at.slice(0, 10)}</text>
    <text class="chart-axis-label" x="${W - R}" y="${H - 15}" text-anchor="end">NEXT CONFIRMED 1D CLOSE ADDS THE FIRST CURVE POINT</text>`;
  tip.style.opacity = '0';
  svg.onmousemove = null;
  svg.onmouseleave = null;
}

function drawChart(rows) {
  if (!rows.length) {
    chartFrame(document.querySelector('#chart'));
    return;
  }
  const distinctDates = new Set(rows.map(row => row.captured_at.slice(0, 10)));
  if (rows.length < 2 || distinctDates.size < 2) {
    drawBootstrapState(rows);
    return;
  }
  const svg = document.querySelector('#chart');
  const tip = document.querySelector('#chart-tooltip');
  const { W, H, L, R, T, B, plotW, plotH } = chartFrame(svg);
  const values = rows.map(row => row.equity);
  const low = Math.min(...values), high = Math.max(...values);
  const pad = Math.max((high - low) * .12, Math.abs(values[0]) * .0025);
  const min = low - pad, max = high + pad, range = max - min;
  const x = index => L + index / (rows.length - 1) * plotW;
  const y = value => T + (max - value) / range * plotH;
  let markup = '<defs><linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#e3a54b" stop-opacity=".28"/><stop offset="100%" stop-color="#e3a54b" stop-opacity="0"/></linearGradient></defs>';
  for (let index = 0; index < 5; index += 1) {
    const value = min + range * index / 4;
    const yy = y(value);
    markup += `<line class="chart-grid" x1="${L}" y1="${yy}" x2="${W - R}" y2="${yy}"/><text class="chart-axis-label" x="${L - 12}" y="${yy + 4}" text-anchor="end">${money(value)}</text>`;
  }
  const dateLabels = [0, Math.floor((rows.length - 1) / 2), rows.length - 1];
  for (const index of [...new Set(dateLabels)]) {
    const anchor = index === 0 ? 'start' : index === rows.length - 1 ? 'end' : 'middle';
    markup += `<text class="chart-axis-label" x="${x(index)}" y="${H - 15}" text-anchor="${anchor}">${rows[index].captured_at.slice(0, 10)}</text>`;
  }
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(' ');
  markup += `<polygon class="chart-area" points="${L},${T + plotH} ${points} ${W - R},${T + plotH}"/><polyline class="chart-line" points="${points}"/><circle class="chart-marker" cx="${x(0)}" cy="${y(values[0])}" r="4"/><circle class="chart-marker" cx="${x(values.length - 1)}" cy="${y(values.at(-1))}" r="4"/><line id="chart-crosshair" class="chart-crosshair" x1="${L}" y1="${T}" x2="${L}" y2="${T + plotH}" opacity="0"/><circle id="chart-hover-dot" class="chart-marker" cx="${L}" cy="${T}" r="4" opacity="0"/>`;
  svg.innerHTML = markup;
  svg.onmousemove = event => {
    const rect = svg.getBoundingClientRect();
    const raw = (event.clientX - rect.left) / rect.width;
    const index = Math.max(0, Math.min(rows.length - 1, Math.round((raw * W - L) / plotW)));
    const row = rows[index], cx = x(index), cy = y(row.equity);
    const crosshair = svg.querySelector('#chart-crosshair');
    crosshair.setAttribute('x1', cx); crosshair.setAttribute('x2', cx); crosshair.setAttribute('opacity', '1');
    const dot = svg.querySelector('#chart-hover-dot');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('opacity', '1');
    tip.innerHTML = `${row.captured_at.slice(0, 10)}<br><strong>${money(row.equity)}</strong>`;
    tip.style.left = `${Math.max(72, Math.min(928, cx)) / W * 100}%`;
    tip.style.top = `${Math.max(42, cy) / H * 100}%`;
    tip.style.opacity = '1';
  };
  svg.onmouseleave = () => {
    svg.querySelector('#chart-crosshair').setAttribute('opacity', '0');
    svg.querySelector('#chart-hover-dot').setAttribute('opacity', '0');
    tip.style.opacity = '0';
  };
}

function renderPositions(rows) {
  latestPositions = rows;
  const body = document.querySelector('#positions-body');
  document.querySelector('#position-count').textContent = `${rows.length} instruments`;
  const direction = positionSort.direction === 'asc' ? 1 : -1;
  const sortedRows = [...rows].sort((left, right) => {
    const a = left[positionSort.key], b = right[positionSort.key];
    if (typeof a === 'string') return direction * a.localeCompare(b);
    return direction * ((a ?? 0) - (b ?? 0));
  });
  body.innerHTML = sortedRows.length ? sortedRows.map(row => {
    const pnlClass = row.unrealized_pnl > 0 ? 'positive' : row.unrealized_pnl < 0 ? 'negative' : 'neutral';
    return `<tr><td><b>${row.symbol}</b></td><td class="side-${row.side}">${row.side}</td><td>${money(row.notional)}</td><td>${number(row.entry_price)}</td><td>${number(row.mark_price)}</td><td class="${pnlClass}">${money(row.unrealized_pnl)}</td></tr>`;
  }).join('') : '<tr><td colspan="6" class="empty">No positions yet.</td></tr>';
  document.querySelectorAll('.sort-button').forEach(button => {
    const active = button.dataset.sortKey === positionSort.key;
    button.dataset.direction = active ? positionSort.direction : '';
    button.setAttribute('aria-sort', active ? (positionSort.direction === 'asc' ? 'ascending' : 'descending') : 'none');
  });
}

document.querySelectorAll('.sort-button').forEach(button => {
  button.addEventListener('click', () => {
    const key = button.dataset.sortKey;
    positionSort.direction = positionSort.key === key && positionSort.direction === 'asc' ? 'desc' : 'asc';
    positionSort.key = key;
    renderPositions(latestPositions);
  });
});

function renderExecutions(rows) {
  document.querySelector('#events').innerHTML = rows.length ? rows.map(row => `<li><b>${row.symbol}</b> ${row.side} ${row.filled_quantity ? number(row.filled_quantity) : ''}<br><small>${row.status} · ${row.slippage_bps == null ? '—' : number(row.slippage_bps) + ' bps'}</small></li>`).join('') : '<li class="empty">Waiting for the first daily rebalance.</li>';
}

async function refresh() {
  try {
    const [session, positions, equity, executions] = await Promise.all(['/api/session', '/api/positions', '/api/equity', '/api/executions'].map(url => fetch(url).then(response => {
      if (!response.ok) throw new Error(`${url} unavailable`);
      return response.json();
    })));
    const account = session.account, replay = session.status === 'historical_replay';
    document.querySelector('#status').textContent = session.status.replaceAll('_', ' ').toUpperCase();
    document.querySelector('#mode-label').textContent = replay ? 'HISTORICAL OOS REPLAY' : 'LIVE PAPER SESSION';
    document.querySelector('#session-mode').innerHTML = replay ? 'REPLAY / LOCAL OHLCV<br><span>NO ORDER BOOK OR SLIPPAGE</span>' : 'PAPER / BYBIT LINEAR<br><span>1D CLOSE ONLY</span>';
    document.querySelector('#equity').textContent = money(account.equity);
    document.querySelector('#cash').textContent = money(account.cash);
    document.querySelector('#gross').textContent = money(account.gross_notional);
    document.querySelector('#cutoff').textContent = session.model.cutoff_date;
    document.querySelector('#last-date').textContent = session.last_decision_date || 'Awaiting first close';
    document.querySelector('#model-info').textContent = `${session.model.backend.toUpperCase()} · h${session.model.horizon_days} · ${session.model.seed_count} seeds · cut-off ${session.model.cutoff_date}`;
    renderPositions(positions); renderExecutions(executions); drawChart(equity);
  } catch (_) {
    document.querySelector('#status').textContent = 'API UNAVAILABLE';
  }
}

refresh();
setInterval(refresh, 3000);
