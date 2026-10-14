"""Compare candidate result files against base result files: segment Sharpe, bootstrap, risk."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from harness import DEV, SEGMENTS, paired_t, stats


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "results" / f"{name}.csv", index_col=0, parse_dates=True)


def bootstrap(a: pd.Series, b: pd.Series, draws: int = 2000) -> tuple[float, float]:
    frame = pd.DataFrame({"a": a, "b": b})[DEV[0] : DEV[1]]
    blocks = [group for _, group in frame.groupby(frame.index.to_period("M"))]
    rng = np.random.default_rng(1)
    out = []
    for _ in range(draws):
        sample = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        out.append((sample.b.mean() / sample.b.std() - sample.a.mean() / sample.a.std()) * np.sqrt(365))
    out = np.array(out)
    return float((out > 0).mean()), float(np.quantile(out, 0.05))


for pair in sys.argv[1:]:
    base, cand = pair.split(":")
    A, B = load(base), load(cand)
    a, b = A.ideal, B.ideal
    dev = slice(*DEV)
    row = " ".join(
        f"{n}:{stats(a[x:y])['sharpe']:.2f}->{stats(b[x:y])['sharpe']:.2f}" for n, (x, y) in {"DEV": DEV, **SEGMENTS}.items()
    )
    p, q = bootstrap(a, b)
    ma, mb = stats(a[dev]), stats(b[dev])
    print(
        f"{cand:16s} {row} | t={paired_t(b[dev], a[dev]):+.2f} P={p:.2f} q05={q:+.2f} "
        f"cagr {ma['cagr']:.0%}->{mb['cagr']:.0%} dd {ma['max_dd']:.1%}->{mb['max_dd']:.1%} "
        f"+mon {ma['pos_months']:.0%}->{mb['pos_months']:.0%} exec {stats(A.exec[dev])['sharpe']:.2f}->{stats(B.exec[dev])['sharpe']:.2f}"
    )
