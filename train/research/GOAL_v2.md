# Goal: improve the airōk strategy on both the full and the Copy Trading universe

You are a quantitative researcher. Iteratively improve the risk-adjusted
profitability and stability of the airōk cross-sectional long/short crypto
strategy. Every iteration: state a hypothesis with an economic rationale,
implement it behind a config flag, run it, evaluate it like a quant, accept or
reject it against the criteria below, log it, and move to the next.

## The core requirement: two evaluation tracks, one training run

The strategy will be traded through Bybit Copy Trading so that people can copy
the trades. Only Copy Trading contracts can be traded there. Every experiment
is therefore evaluated in two tracks that share the **same trained models**:

| Track | Models trained on | Portfolio may trade |
|---|---|---|
| **A – FULL** | full universe (all USDT perpetuals in `data/1d`) | full universe |
| **B – COPY** | full universe (same models as A) | only Copy Trading contracts |

Features, cross-sectional ranks, targets and training always use the full
universe. Track B only restricts which symbols the portfolio may select as
longs or shorts (and hedge with). The liquidity-ranked trading universe
(`portfolio.universe_size`) is formed **after** the Copy Trading filter.

**An improvement counts only if it holds in both tracks.** If a change helps one
track and hurts the other beyond noise, it is rejected (or reworked).

### First task: verify "train on full, trade on Copy Trading"

Earlier runs suggested that training on the full universe and trading only Copy
Trading pairs beats training on the Copy Trading universe alone. Re-test this on
the current champion, evaluated on track B:

1. train on full universe, trade Copy Trading (track B);
2. train only on the Copy Trading universe (`data.copy_trading_only: true`), trade Copy Trading.

If (1) is at least as good as (2) by the acceptance criteria, all further work
trains on the full universe only. If not, report the evidence and stop to ask.

### Copy Trading universe definition

- Primary: the current Bybit `instruments-info` set with `status=Trading`,
  `contractType=LinearPerpetual`, `quoteCoin=settleCoin=USDT`,
  `copyTrading=both`. Freeze it once into `.cache/` (as the harness already does
  for instrument rules) so runs are deterministic.
- Stress check: the stricter August-2026 Copy Trading list (416 symbols) in
  `train/outputs/runs/20260901_164714_lightgbm_h7_copytrading_multiseed/training_universe.json`.
  The current list (~757) is much wider than August's. Report track B on the
  strict list for every accepted change.
- Limitation to state in every report: Bybit gives no Copy Trading eligibility
  history. Applying today's list to the past is a mild look-ahead. Treat
  track B as an approximation.

## Context (read before starting)

- Repo: `/Users/quarkacc/Desktop/DEV/airok`. Work in `train/` on branch
  `research/funding-sharpe`, or a new branch from it. Commit after each accepted
  iteration and at milestones. Never push. Never commit data or bulky results.
- Python: conda env `nn` (`/opt/homebrew/Caskroom/miniconda/base/envs/nn/bin/python`).
- Read first: `train/research/GOAL.md`, `train/research/LOG.md` (14 earlier
  iterations), `train/research/harness.py`, `train/research/compare.py`,
  `train/src/{research,portfolio,pipeline,funding}.py`.
- Data: `data/1d` has 686 symbols from 2021-02-12 to 2026-10-13, including
  delisted ones up to their delisting date. `data/funding` has every funding
  settlement for the same symbols. Refresh both with
  `python src/scraper_bybit.py --funding ...` (see `train/README.md`).
- Everything is evaluated **with funding** and 10 bps one-way costs. The main
  series is ideal target weights; the control series is the Bybit execution
  simulation with frozen instrument rules.
- Current state:
  - The DEV champion `i_size` (h7+h21 ensemble, funding features, net-of-funding
    target, 0.5 alt-index neutralization, size neutralization) has DEV Sharpe
    2.93 against a 2.08 baseline.
  - On the HOLDOUT, opened once, the baseline scored 3.02 and `i_size` 2.55.
    That difference is not significant: with 122 days the Sharpe standard error
    is about 4.
  - The robust finding is portfolio neutralization: the baseline model plus
    alt 0.5 and size neutralization gives DEV 2.43 and HOLDOUT 2.94.
  - The funding-in-model step is unconfirmed outside DEV.
  - Start by evaluating both of these stacks, `i_size` and baseline model +
    neutralization, in both tracks. Pick the starting champion by the criteria
    below. Do not assume the DEV champion wins.
- Known findings:
  - Almost all residual alpha comes from shorts: +48%/yr against +5%/yr for
    longs after BTC beta. Funding is the price of that alpha.
  - Funding filters or penalties destroy the alpha.
  - Model longs are a necessary hedge of the alt factor.
  - Tail count 6, invvol weighting, 21-day boxcar smoothing and the LightGBM
    hyperparameters are at a local optimum.

## Evaluation protocol

- **DEV** 2023-08-05 … 2026-06-12, used for selection. Segments: 23H2, 2024,
  2025, 26H1.
- **HOLDOUT** 2026-06-13 … 2026-10-12 has been seen once. Report it, but do not
  select on it. It is only a veto: reject a change whose HOLDOUT paired t
  against the champion is below −1.5 in either track.
- **FORWARD**: if new daily data becomes available after 2026-10-13, refresh it
  and report it as the only truly untouched out-of-sample. Never tune on it.
- Report per track and per period: Sharpe, Sortino, CAGR, MaxDD, Calmar, share
  of profitable months and days, worst month, turnover, funding per year,
  execution-simulation Sharpe, and long-side versus short-side residual alpha.
- Noise control: a single 5-seed run moves Sharpe by about ±0.2–0.3. Confirm
  model, target or feature changes on both seed sets (`[11,23,37,53,71]` and
  `[101,202,303,404,505]`) and decide on the 10-seed ensemble. Portfolio-only
  changes reuse cached predictions and are cheap: check them on both 5-seed
  subsets too.

## Acceptance criteria (must hold in BOTH tracks A and B)

1. DEV Sharpe improves by ≥ +0.15, **and** either the paired daily-return
   t ≥ 1.5 or, for risk-shaping changes that leave mean returns unchanged,
   the monthly block-bootstrap P(ΔSharpe > 0) ≥ 0.85.
2. Sharpe is not worse in ≥ 3 of 4 DEV segments, and no segment is worse by
   more than 0.3.
3. DEV MaxDD is not worse by more than 2 pp, and the share of profitable months
   is not lower by more than 3 pp.
4. Effect sign and size are confirmed on both seed sets.
5. HOLDOUT veto is not triggered.
6. Track B also holds on the strict August-2026 Copy Trading list: same sign,
   no segment collapse.

A change that falls just short of one numeric threshold may be accepted only
with an explicit holistic argument, written in the log, that weighs Sharpe,
CAGR, drawdown, stability and execution. Never quietly move the goalposts.

## Hypothesis backlog (re-prioritize as you learn)

1. **Infrastructure:** track-B trading mask in the portfolio, harness reporting
   both tracks, and the train-full-versus-train-copy test above.
2. **New data sources**, the most promising route to new alpha. Verify the
   available history first and extend `scraper_bybit.py` behind flags:
   - open interest (`/v5/market/open-interest`, daily);
   - long/short account ratio (`/v5/market/account-ratio`);
   - premium index and mark-price klines (basis);
   - spot versus perp basis where spot exists;
   - 1h/4h candles for intraday features (realized skew, range, close-location
     dynamics, overnight versus intraday).

   Build causal, cross-sectionally ranked features from them.
3. Funding-in-model re-validation in both tracks: features only versus net
   target, horizons, rank-only. Decide whether it stays.
4. Feature universe breadth (`feature_universe_max` 150 → 250) and the trading
   universe size per track. Track B has fewer names, so the optimal
   `universe_size` and `tail_count` may differ.
5. Alternative factor neutralization: alt-index definition (equal-weighted
   versus liquidity-weighted, top-N), sector or cluster exposures from return
   correlation clustering, strength λ.
6. Cost and turnover: no-trade bands, partial rebalancing, maker-fee
   assumptions as a sensitivity only.
7. Copy Trading practicalities for track B: minimum order notional, leverage
   caps and position-count limits, if Bybit Copy Trading imposes them. Check
   the docs and API rules and model only real constraints.
8. Anything else your analysis suggests. Diagnose before you tune: use
   attribution by side, segment, factor and funding to find where the strategy
   loses money.

## Working rules

- One hypothesis per iteration, behind a config flag, with an economic
  rationale written before the result is seen.
- Use `research/harness.py <name> key=value …`. In zsh, pass multiple
  overrides as separate words or an array; a single unquoted variable is not
  word-split. Check `research/configs/<name>.yaml` after each run.
- Keep the prediction cache valid: a model's key must depend on its own seed,
  horizon, side, target, features and training schedule, never on the rest of
  the ensemble.
- Log every iteration in `research/LOG.md`: hypothesis, rationale, both-track
  tables, decision and reason. Reject failures in the log, and leave the code
  clean. Remove dead config options that will never be used.
- Do not use the HOLDOUT or FORWARD periods to choose parameters.
- Any accepted feature or portfolio rule must later be implementable in the
  Rust live service (`live/crates/rt-panel`) with bit-for-bit parity. Note what
  each accepted change requires there. Data sources must be available live.
- Run tests (`python -m pytest -q` in `train/`) before each commit.

## Reporting

After every 3–4 iterations, and at the end, give a short status:

- the current champion in both tracks against the starting baseline: Sharpe,
  CAGR, MaxDD and profitable months on DEV, HOLDOUT and FORWARD if available;
- what was accepted and what was rejected, with one-line reasons;
- what you plan next.

Stop and ask the user when a decision changes the trading setup (universe,
leverage, live changes), or when the train-full-versus-train-copy hypothesis
fails.
