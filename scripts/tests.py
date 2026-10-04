"""Offline checks: indicator math on synthetic series + a full run with the network mocked."""
import json, os, shutil, sys, tempfile, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fetch  # noqa: E402

# 1. RSI: steady rise -> 100, steady fall -> 0, alternating -> ~50
up = pd.Series(np.arange(1, 60, dtype=float))
assert fetch.rsi(up).iloc[-1] == 100, fetch.rsi(up).iloc[-1]
assert fetch.rsi(up[::-1].reset_index(drop=True)).iloc[-1] < 1
alt = pd.Series([10 + (i % 2) for i in range(80)], dtype=float)
assert 40 < fetch.rsi(alt).iloc[-1] < 60

# 2. Wilder RSI against a hand-rolled reference
rng = np.random.default_rng(1)
s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300))))
d = s.diff().to_numpy()
g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
ag, al = g[1:15].mean(), l[1:15].mean()
for i in range(15, len(s)):
    ag = (ag * 13 + g[i]) / 14; al = (al * 13 + l[i]) / 14
ref = 100 - 100 / (1 + ag / al)
assert abs(fetch.rsi(s).iloc[-1] - ref) < 0.5, (fetch.rsi(s).iloc[-1], ref)  # ewm seed differs slightly, converges

# 3. EMA stack labels
assert fetch.ema_stack(10, 9, 8, 7) == "bull"
assert fetch.ema_stack(5, 6, 7, 8) == "bear"

# 4. Full run with mocked sources
def candles(base, n=400, drift=0.001):
    t0 = int(time.time() // 86400 - n) * 86400 * 1000
    px = base * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))
    return [[t0 + i * 86400000, p, p * 1.02, p * 0.98, p, 10, t0 + (i + 1) * 86400000 - 1, p * 1e6, 1, 0, 0, 0]
            for i, p in enumerate(px)]

def fake_get(url, params=None, tries=3):
    if "coins/markets" in url:
        return [{"symbol": s, "id": s.lower(), "name": s, "current_price": p, "market_cap_rank": i + 1, "market_cap": 1e9,
                 "total_volume": 1e8} for i, (s, p) in enumerate([("BTC", 85000), ("ETH", 2700), ("USDT", 1.0), ("WBTC", 85000),
                                                                   ("SOL", 120), ("DOGE", 0.09)])]
    if "klines" in url:
        sym = params["symbol"].replace("USDT", "")
        return candles({"BTC": 85000, "ETH": 2700, "SOL": 120, "DOGE": 0.09}[sym])
    if "fng" in url:
        return {"data": [{"value": "67", "value_classification": "Greed"}] * 30}
    if "global" in url:
        return {"data": {"market_cap_percentage": {"btc": 58.9, "eth": 11.4}, "total_market_cap": {"usd": 2.9e12},
                         "market_cap_change_percentage_24h_usd": 0.4}}
    if "okx" in url:
        return {"data": [{"fundingRate": "0.0001"}]}
    raise RuntimeError(url)

tmp = tempfile.mkdtemp()
fetch.ROOT = tmp
fetch.get = fake_get
fetch.main()
snap = json.load(open(os.path.join(tmp, "indicators.json")))
syms = [c["symbol"] for c in snap["coins"]]
assert syms == ["BTC", "ETH", "SOL", "DOGE"], syms              # stable + wrapped dropped, meme kept but flagged
assert snap["coins"][3]["kind"] == "meme"
eth = snap["coins"][1]
for k in ("ema200", "rsi14", "vs_btc_trend", "chg_30d", "atr14_pct"):
    assert eth[k] is not None, k
assert snap["sentiment"]["fear_greed"]["value"] == 67
assert os.path.exists(os.path.join(tmp, "ohlc", "ETH.csv")) and os.path.exists(os.path.join(tmp, "summary.md"))
print(open(os.path.join(tmp, "summary.md")).read()[:900])
shutil.rmtree(tmp)
print("ALL TESTS PASSED")
