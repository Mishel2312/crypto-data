"""Daily market data snapshot for the paper crypto portfolio.

Runs in GitHub Actions every morning. Writes:
  data/universe.json      top coins by market cap (CoinGecko), stables/memes flagged
  data/ohlc/<SYM>.csv     daily candles (closed days only), USD
  data/macro/<NAME>.csv   DXY, S&P 500, NASDAQ, US10Y daily closes
  data/indicators.json    per-coin indicators + macro + sentiment, ready for scoring
  data/summary.md         the same, as readable tables
Every source is optional: a failure is recorded in `errors` and the run continues.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
UA = {"User-Agent": "crypto-data-snapshot/1.0"}
TOP_N = 40  # fetch a bit more than top-30 so held coins that slip a few ranks stay covered

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "USD1", "PYUSD", "USDD",
           "BUSD", "FRAX", "GHO", "RLUSD", "USDTB", "USDF", "SUSDE", "EUSDE", "USDG", "BFUSD", "USYC"}
MEMES = {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "TRUMP", "FARTCOIN", "PENGU", "SPX",
         "BRETT", "POPCAT", "MOG", "PUMP"}
WRAPPED = {"WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "METH", "LBTC", "SOLVBTC",
           "BSC-USD", "JITOSOL", "MSOL", "WBETH", "EZETH", "RSETH", "CLBTC", "BNSOL", "WBT"}

MACRO = {"DXY": "DX-Y.NYB", "SPX": "^GSPC", "NDX": "^IXIC", "US10Y": "^TNX"}

errors: list[str] = []


def get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(15 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"rate limited: {url}")


# ---------- indicators (pure functions, unit-tested in tests.py) ----------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def pct(a, b):
    return None if a is None or b is None or b == 0 or pd.isna(a) or pd.isna(b) else round((a / b - 1) * 100, 2)


def r(x, nd=6):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def ema_stack(c, e20, e50, e200) -> str:
    if any(pd.isna(v) for v in (e20, e50)):
        return "n/a"
    if not pd.isna(e200):
        if c > e20 > e50 > e200:
            return "bull"   # full bullish alignment
        if c < e20 < e50 < e200:
            return "bear"
        if c > e200:
            return "above200"
        return "below200"
    return "bull_short" if c > e20 > e50 else "bear_short" if c < e20 < e50 else "mixed"


def bearish_divergence(close: pd.Series, rs: pd.Series, w: int = 14) -> bool | None:
    """Price made a higher high in the last w days than in the w before, while RSI made a lower high."""
    if len(close) < 2 * w + 1 or rs.iloc[-2 * w:].isna().any():
        return None
    p1, p2 = close.iloc[-2 * w:-w].max(), close.iloc[-w:].max()
    r1, r2 = rs.iloc[-2 * w:-w].max(), rs.iloc[-w:].max()
    return bool(p2 > p1 and r2 < r1 - 2)


def coin_indicators(df: pd.DataFrame, btc: pd.DataFrame | None) -> dict:
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    rs = rsi(c)
    last = c.iloc[-1]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    rets = c.pct_change().iloc[-30:]
    out = {
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "close": r(last, 8),
        "chg_1d": pct(last, c.iloc[-2]) if len(c) > 1 else None,
        "chg_7d": pct(last, c.iloc[-8]) if len(c) > 7 else None,
        "chg_30d": pct(last, c.iloc[-31]) if len(c) > 30 else None,
        "ema20": r(e20.iloc[-1], 8), "ema50": r(e50.iloc[-1], 8), "ema200": r(e200.iloc[-1], 8),
        "ema_stack": ema_stack(last, e20.iloc[-1], e50.iloc[-1], e200.iloc[-1]),
        "rsi14": r(rs.iloc[-1], 1),
        "rsi_bear_div": bearish_divergence(c, rs),
        "atr14_pct": r(atr / last * 100, 2) if not pd.isna(atr) else None,
        "vol30_ann_pct": r(rets.std() * math.sqrt(365) * 100, 1) if len(rets.dropna()) > 10 else None,
        "low_30d": r(c.iloc[-30:].min(), 8), "high_30d": r(c.iloc[-30:].max(), 8),
        "dist_from_30d_low_pct": pct(last, c.iloc[-30:].min()),
        "dist_from_90d_high_pct": pct(last, c.iloc[-90:].max()),
        "avg_quote_vol_7d_usd": r(df["quote_volume"].iloc[-7:].mean(), 0) if "quote_volume" in df else None,
        "days": int(len(c)),
    }
    if btc is not None:
        ratio = (c / btc["close"]).dropna()
        if len(ratio) > 50:
            r20, r50 = ema(ratio, 20).iloc[-1], ema(ratio, 50).iloc[-1]
            out.update({
                "vs_btc_chg_7d": pct(ratio.iloc[-1], ratio.iloc[-8]),
                "vs_btc_chg_30d": pct(ratio.iloc[-1], ratio.iloc[-31]) if len(ratio) > 30 else None,
                "vs_btc_trend": "up" if ratio.iloc[-1] > r20 > r50 else "down" if ratio.iloc[-1] < r20 < r50 else "mixed",
            })
    return out


# ---------- sources ----------

def universe():
    data = get("https://api.coingecko.com/api/v3/coins/markets",
               {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 80, "page": 1,
                "price_change_percentage": "24h,7d,30d"})
    rows = []
    for d in data:
        sym = d["symbol"].upper()
        kind = ("stable" if sym in STABLES or (d.get("current_price") and abs(d["current_price"] - 1) < 0.02 and sym.startswith("USD"))
                else "meme" if sym in MEMES else "wrapped" if sym in WRAPPED else "ok")
        rows.append({"rank": d.get("market_cap_rank"), "symbol": sym, "id": d["id"], "name": d["name"], "kind": kind,
                     "price": d.get("current_price"), "market_cap": d.get("market_cap"), "volume_24h": d.get("total_volume"),
                     "chg_24h": r(d.get("price_change_percentage_24h_in_currency"), 2),
                     "chg_7d": r(d.get("price_change_percentage_7d_in_currency"), 2),
                     "chg_30d": r(d.get("price_change_percentage_30d_in_currency"), 2)})
    # rank among real assets (wrapped tokens duplicate their base asset)
    rows = [x for x in rows if x["kind"] != "wrapped"]
    for i, x in enumerate(rows, 1):
        x["rank_clean"] = i
    return rows[:TOP_N]


def binance_daily(sym: str) -> pd.DataFrame | None:
    try:
        k = get("https://data-api.binance.vision/api/v3/klines", {"symbol": f"{sym}USDT", "interval": "1d", "limit": 400})
    except Exception:  # noqa: BLE001
        return None
    if not k:
        return None
    now_ms = time.time() * 1000
    k = [x for x in k if x[6] < now_ms]  # closed candles only
    df = pd.DataFrame(k, columns=["t", "open", "high", "low", "close", "volume", "ct", "quote_volume", "n", "tb", "tq", "i"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    return df[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)


def coinbase_daily(sym: str) -> pd.DataFrame | None:
    try:
        k = get(f"https://api.exchange.coinbase.com/products/{sym}-USD/candles", {"granularity": 86400})
    except Exception:  # noqa: BLE001
        return None
    if not k:
        return None
    df = pd.DataFrame(k, columns=["t", "low", "high", "open", "close", "volume"]).sort_values("t")
    df.index = pd.to_datetime(df["t"], unit="s")
    df = df[df.index < pd.Timestamp.utcnow().tz_localize(None).normalize()]
    df["quote_volume"] = df["volume"] * df["close"]
    return df[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)


def macro():
    out = {}
    try:
        import yfinance as yf
    except ImportError as e:
        errors.append(f"macro: {e}")
        return out
    for name, tk in MACRO.items():
        try:
            h = yf.Ticker(tk).history(period="1y", interval="1d", auto_adjust=False)
            if h.empty:
                raise ValueError("empty")
            c = h["Close"].dropna()
            c.index = c.index.tz_localize(None).normalize()
            pd.DataFrame({"close": c}).to_csv(os.path.join(ROOT, "macro", f"{name}.csv"), index_label="date")
            e50 = ema(c, 50).iloc[-1]
            out[name] = {"last_date": c.index[-1].strftime("%Y-%m-%d"), "close": r(c.iloc[-1], 3),
                         "chg_1d": pct(c.iloc[-1], c.iloc[-2]), "chg_5d": pct(c.iloc[-1], c.iloc[-6]),
                         "chg_20d": pct(c.iloc[-1], c.iloc[-21]), "ema50": r(e50, 3),
                         "trend": "up" if c.iloc[-1] > e50 else "down"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"macro {name}: {e}")
    return out


def sentiment():
    out = {}
    try:
        d = get("https://api.alternative.me/fng/", {"limit": 30})["data"]
        out["fear_greed"] = {"value": int(d[0]["value"]), "label": d[0]["value_classification"],
                             "value_7d_ago": int(d[7]["value"]) if len(d) > 7 else None,
                             "avg_30d": round(sum(int(x["value"]) for x in d) / len(d), 1)}
    except Exception as e:  # noqa: BLE001
        errors.append(f"fear_greed: {e}")
    try:
        g = get("https://api.coingecko.com/api/v3/global")["data"]
        out["btc_dominance"] = r(g["market_cap_percentage"].get("btc"), 2)
        out["eth_dominance"] = r(g["market_cap_percentage"].get("eth"), 2)
        out["total_mcap_usd"] = r(g["total_market_cap"].get("usd"), 0)
        out["total_mcap_chg_24h"] = r(g.get("market_cap_change_percentage_24h_usd"), 2)
    except Exception as e:  # noqa: BLE001
        errors.append(f"global: {e}")
    funding = {}
    for sym in ("BTC", "ETH", "SOL"):
        try:
            f = get("https://www.okx.com/api/v5/public/funding-rate", {"instId": f"{sym}-USDT-SWAP"})["data"][0]
            funding[sym] = {"rate_pct": r(float(f["fundingRate"]) * 100, 4), "source": "OKX, per 8h"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"funding {sym}: {e}")
    if funding:
        out["funding"] = funding
    return out


def main():
    os.makedirs(os.path.join(ROOT, "ohlc"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "macro"), exist_ok=True)
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        uni = universe()
    except Exception as e:  # noqa: BLE001
        errors.append(f"universe: {e}")
        uni = [{"symbol": s, "kind": "ok", "rank_clean": i} for i, s in enumerate(
            ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ADA", "LINK", "AVAX", "SUI", "XLM", "BCH", "LTC", "DOT", "NEAR", "UNI"], 1)]
    json.dump({"generated_at": gen, "coins": uni}, open(os.path.join(ROOT, "universe.json"), "w"), indent=1)

    frames: dict[str, pd.DataFrame] = {}
    for x in uni:
        if x["kind"] == "stable":
            continue
        sym = x["symbol"]
        df = binance_daily(sym)
        src = "binance"
        if df is None or len(df) < 30:
            df, src = coinbase_daily(sym), "coinbase"
        if df is None or len(df) < 30:
            errors.append(f"ohlc {sym}: no source")
            continue
        df.attrs["source"] = src
        frames[sym] = df
        df.round(10).to_csv(os.path.join(ROOT, "ohlc", f"{sym}.csv"), index_label="date")
        time.sleep(0.2)

    btc = frames.get("BTC")
    coins = []
    for x in uni:
        sym = x["symbol"]
        if sym not in frames:
            continue
        ind = coin_indicators(frames[sym], btc if sym != "BTC" else None)
        coins.append({"symbol": sym, "name": x.get("name"), "rank": x.get("rank_clean"), "kind": x["kind"],
                      "market_cap": x.get("market_cap"), "volume_24h": x.get("volume_24h"),
                      "source": frames[sym].attrs.get("source"), **ind})

    snap = {"generated_at": gen, "note": "Daily candles are closed UTC days; 'close' is the last fully closed day.",
            "sentiment": sentiment(), "macro": macro(), "coins": coins, "errors": errors}
    json.dump(snap, open(os.path.join(ROOT, "indicators.json"), "w"), indent=1, ensure_ascii=False)
    write_summary(snap)
    print(f"ok: {len(coins)} coins, {len(errors)} errors")
    for e in errors:
        print("  -", e)


def write_summary(s):
    L = [f"# Market snapshot {s['generated_at']}", ""]
    se = s.get("sentiment", {})
    fg = se.get("fear_greed", {})
    L.append(f"Fear&Greed: {fg.get('value')} ({fg.get('label')}), 7d ago {fg.get('value_7d_ago')} · "
             f"BTC dom {se.get('btc_dominance')}% · mcap chg 24h {se.get('total_mcap_chg_24h')}%")
    if se.get("funding"):
        L.append("Funding (8h): " + ", ".join(f"{k} {v['rate_pct']}%" for k, v in se["funding"].items()))
    L += ["", "| Macro | close | 1d | 5d | 20d | trend vs EMA50 |", "|---|---|---|---|---|---|"]
    for k, v in s.get("macro", {}).items():
        L.append(f"| {k} | {v['close']} | {v['chg_1d']} | {v['chg_5d']} | {v['chg_20d']} | {v['trend']} |")
    L += ["", "| # | Coin | kind | close | 1d% | 7d% | 30d% | EMA stack | RSI | bear div | vs BTC 30d% | vs BTC trend | vol7d $M |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in s["coins"]:
        v = c.get("avg_quote_vol_7d_usd")
        L.append(f"| {c['rank']} | {c['symbol']} | {c['kind']} | {c['close']} | {c['chg_1d']} | {c['chg_7d']} | {c['chg_30d']} | "
                 f"{c['ema_stack']} | {c['rsi14']} | {c['rsi_bear_div']} | {c.get('vs_btc_chg_30d')} | {c.get('vs_btc_trend')} | "
                 f"{round(v / 1e6) if v else None} |")
    if s["errors"]:
        L += ["", "Errors:"] + [f"- {e}" for e in s["errors"]]
    open(os.path.join(ROOT, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
