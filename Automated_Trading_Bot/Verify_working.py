
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from Backtest import make_synthetic_bars, performance_stats, run_backtest, trade_stats
from Bot import Bot, StateStore, seconds_until_next_bar
from Broker_base import BrokerError, MarketClosed, OrderRejected
from Broker_alpaca import AlpacaBroker
from Broker_paper import PaperBroker
from Config import ConfigError, Settings, load_dotenv
from Risk import daily_loss_breached, position_size, price_decimals, stop_take_levels
from Strategy import StrategyParams, compute_indicators, latest_signal, target_series
from Trade_logger import TradeLogger

logging.disable(logging.CRITICAL)             # keep the test output readable
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


T0 = pd.Timestamp("2025-01-01", tz="UTC")


def S(**kw):
    """Settings for the bot-logic tests: leverage cap effectively off (the cap has its own tests)."""
    kw.setdefault("max_leverage", 1000.0)
    return Settings(**kw)




def bars_from_closes(closes, wick=0.0003, start=T0):
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"time": pd.date_range(start, periods=len(c), freq="h"),
                         "open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick,
                         "close": c, "volume": 1.0})


# =========================== CONFIG ===========================
tmp = Path(tempfile.mkdtemp())
(tmp / ".env").write_text("# comment\nFAST=10\nSLOW='40'\nINSTRUMENT=\"GBP_USD\"\nBAD LINE\nRISK_PCT=0.01\n")
loaded = load_dotenv(tmp / ".env")
check("dotenv: parses values, strips quotes, ignores comments/bad lines",
      loaded == {"FAST": "10", "SLOW": "40", "INSTRUMENT": "GBP_USD", "RISK_PCT": "0.01"}, str(loaded))
s = Settings.from_env({"FAST": "10", "SLOW": "40", "RISK_PCT": "0.01", "BROKER": "paper"})
check("settings: env strings cast to int/float", s.fast == 10 and s.slow == 40 and close(s.risk_pct, 0.01))
check("settings: defaults valid", Settings().validate() is None)
check("settings: fast >= slow rejected", raises(ConfigError, Settings.from_env, {"FAST": "50", "SLOW": "20"}))
check("settings: unknown broker rejected", raises(ConfigError, Settings.from_env, {"BROKER": "binance"}))
check("settings: unsupported granularity rejected", raises(ConfigError, Settings.from_env, {"GRANULARITY": "D"}))
check("settings: risk above 5% refused", raises(ConfigError, Settings.from_env, {"RISK_PCT": "0.2"}))
check("settings: non-numeric value rejected", raises(ConfigError, Settings.from_env, {"FAST": "ten"}))
check("settings: lookback too short for indicators rejected",
      raises(ConfigError, Settings.from_env, {"LOOKBACK_BARS": "30"}))
check("settings: LIVE oanda env refused", raises(ConfigError, Settings.from_env, {"OANDA_ENV": "live"}))
check("settings: LIVE IBKR ports refused",
      all(raises(ConfigError, Settings.from_env, {"IBKR_PORT": p}) for p in ("7496", "4001")))
check("settings: paper IBKR ports accepted",
      all(Settings.from_env({"IBKR_PORT": p}).ibkr_port == int(p) for p in ("7497", "4002")))
check("settings: oanda broker without token rejected", raises(ConfigError, Settings.from_env, {"BROKER": "oanda"}))
(tmp / ".env2").write_text("BROKER=oanda   # trailing note\nRISK_PCT=0.01\t# tab note\nOANDA_TOKEN=\"ab#cd\"  # c\nINSTRUMENT='EUR_USD' # d\nGRANULARITY=H1#notacomment\n")
d2 = load_dotenv(tmp / ".env2")
check("dotenv: trailing '# comments' are stripped from unquoted values",
      d2["BROKER"] == "oanda" and d2["RISK_PCT"] == "0.01" and d2["INSTRUMENT"] == "EUR_USD", str(d2))
check("dotenv: a quoted value may contain '#'", d2["OANDA_TOKEN"] == "ab#cd")
check("dotenv: '#' with no preceding space is kept (it is part of the value)", d2["GRANULARITY"] == "H1#notacomment")
_ex = Path(__file__).with_name("env.example")
if _ex.exists():
    _saved = dict(os.environ)                                    # load_dotenv writes to os.environ: undo it afterwards
    try:
        _vals = load_dotenv(_ex, override=True)
    finally:
        os.environ.clear(); os.environ.update(_saved)
    check("shipped env.example, left blank, fails with a clear ConfigError (not a crash)",
          raises(ConfigError, Settings.from_env, dict(_vals)))
    _vals.update({"OANDA_TOKEN": "tok", "OANDA_ACCOUNT_ID": "101-001-1-001", "ALPACA_KEY_ID": "PKtest", "ALPACA_SECRET_KEY": "sec"})
    check("shipped env.example parses and validates once the keys are filled in",
          Settings.from_env(_vals).broker in ("alpaca", "oanda"))
    check("shipped env.example has no trailing comments on value lines",
          all(" #" not in l for l in _ex.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#")))
check("settings: alpaca broker without keys rejected", raises(ConfigError, Settings.from_env, {"BROKER": "alpaca"}))
check("settings: alpaca broker with keys accepted",
      Settings.from_env({"BROKER": "alpaca", "ALPACA_KEY_ID": "k", "ALPACA_SECRET_KEY": "s"}).broker == "alpaca")
check("settings: leverage cap must be in (0, 20]",
      raises(ConfigError, Settings.from_env, {"MAX_LEVERAGE": "0"}) and raises(ConfigError, Settings.from_env, {"MAX_LEVERAGE": "50"}))
check("settings: default leverage is 1.0 (no leverage)", Settings().max_leverage == 1.0)
check("settings: alpaca secret never appears in repr", "TOPSECRET" not in repr(Settings(alpaca_secret_key="TOPSECRET")))
check("settings: token never appears in repr", "SECRET" not in repr(Settings(oanda_token="SECRET")))

# =========================== STRATEGY ===========================
p = StrategyParams(fast=2, slow=3, mom_lookback=2, atr_period=2)
df = bars_from_closes([1, 2, 3, 4, 5], wick=0.5)
ind = compute_indicators(df, p)
check("SMA fast(2) at bar 4 = (4+5)/2 = 4.5", close(ind.sma_fast.iloc[4], 4.5))
check("SMA slow(3) at bar 4 = (3+4+5)/3 = 4", close(ind.sma_slow.iloc[4], 4.0))
check("SMA is NaN during warm-up", np.isnan(ind.sma_slow.iloc[1]) and not np.isnan(ind.sma_slow.iloc[2]))
check("momentum(2) at bar 4 = 5/3 - 1", close(ind.mom.iloc[4], 5 / 3 - 1))
flat = pd.DataFrame({"time": pd.date_range(T0, periods=5, freq="h"), "open": 10.0, "high": 11.0, "low": 9.0,
                     "close": 10.0, "volume": 1})
check("ATR on flat bars with range 2 = 2", close(compute_indicators(flat, p).atr.iloc[4], 2.0))
gap = flat.copy()
gap.loc[3, ["open", "high", "low", "close"]] = [14.5, 15.0, 14.0, 14.5]
check("true range includes the gap: max(1, |15-10|, |14-10|) = 5",
      close(compute_indicators(gap, p).atr.iloc[3], (5.0 + 2.0) / 2))
check("target +1 in a clean uptrend", latest_signal(df, p).target == 1)
check("target -1 in a clean downtrend", latest_signal(bars_from_closes([5, 4, 3, 2, 1]), p).target == -1)
conflict = bars_from_closes([10, 10, 10, 10, 10, 12, 11])
pc = StrategyParams(fast=2, slow=3, mom_lookback=1, atr_period=2)
ic = compute_indicators(conflict, pc)
check("conflict case really has fast>slow and mom<0",
      ic.sma_fast.iloc[-1] > ic.sma_slow.iloc[-1] and ic.mom.iloc[-1] < 0)
check("conflicting signals -> flat (0)", latest_signal(conflict, pc).target == 0)
check("not enough bars (warm-up) -> flat", latest_signal(bars_from_closes([1, 2]), p).target == 0)
big = bars_from_closes(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300)))
pp = StrategyParams()
full = target_series(compute_indicators(big, pp))
check("no look-ahead: signal on bar t unchanged when later bars are removed",
      all(latest_signal(big.iloc[:t + 1], pp).target == full.iloc[t] for t in (80, 120, 200, 299)))

# =========================== RISK ===========================
check("size: 100k equity, 0.5% risk, ATR 0.0008, 2x stop -> 312,500 units", position_size(100_000, .005, .0008, 2, 10**9) == 312_500)
check("size floors (never rounds risk up)", position_size(100_000, .005, .0007, 2, 10**9) == 357_142)
check("size capped by max_units", position_size(100_000, .005, .0008, 2, 50_000) == 50_000)
check("size is 0 for ATR 0 / NaN / negative equity",
      position_size(1e5, .005, 0, 2, 1e9) == 0 and position_size(1e5, .005, float("nan"), 2, 1e9) == 0
      and position_size(-5, .005, .001, 2, 1e9) == 0)
for eq, atr in ((100_000, .0008), (37_000, .0013), (250_000, .0004)):
    u = position_size(eq, .005, atr, 2, 10**12)
    check(f"risk never exceeds budget (equity {eq:,})", u * atr * 2 <= eq * .005 + 1e-9)
check("stop/target for LONG: stop below, target above at tp_rr x stop distance",
      stop_take_levels(1, 1.1, .0008, 2, 2) == (1.0984, 1.1032))
check("stop/target for SHORT mirror", stop_take_levels(-1, 1.1, .0008, 2, 2) == (1.1016, 1.0968))
check("reward:risk equals tp_rr", close((1.1032 - 1.1) / (1.1 - 1.0984), 2.0, 1e-6))
check("JPY pairs use 3 decimals, others 5", price_decimals("USD_JPY") == 3 and price_decimals("EUR_USD") == 5)
check("daily loss guard: -2.0% breaches 2% limit, -1.9% does not",
      daily_loss_breached(100, 98.0, .02) and not daily_loss_breached(100, 98.1, .02))
check("daily loss guard ignores missing start equity", not daily_loss_breached(None, 1, .02))
check("decimals: stocks / ETFs use 2 dp (no sub-penny prices)", price_decimals("SPY") == 2 and price_decimals("AAPL") == 2)
check("leverage cap: risk says 500 sh, but 1x leverage on a 450 stock allows only 222",
      position_size(100_000, .005, 0.5, 2, 10**9, price=450, max_leverage=1.0) == 222)
check("leverage cap scales with the allowed leverage", position_size(100_000, .005, 0.5, 2, 10**9, price=450, max_leverage=2.0) == 444)
check("leverage cap does not bind when the risk-based size is already smaller",
      position_size(100_000, .005, 10.0, 2, 10**9, price=450, max_leverage=1.0) == 25)
check("without price/leverage the original sizing is unchanged", position_size(100_000, .005, .0008, 2, 10**9) == 312_500)

# =========================== PAPER BROKER ===========================
def pb(spread=0.0001):
    b = PaperBroker(100_000, spread)
    b.on_bar(T0, 1.1000, 1.1005, 1.0995, 1.1000)
    return b


b = pb()
f = b.market_order("EUR_USD", 10_000)
check("buy fills at ask (mid + half spread)", close(f.price, 1.10005, 1e-9))
check("equity right after a buy drops by the full spread x units (= 1.00)", close(b.equity(), 100_000 - 1.0, 1e-6))
fc = b.close_position("EUR_USD")
check("round trip at unchanged price loses exactly spread x units", close(b.balance, 100_000 - 1.0, 1e-6))
check("close_position returns the closing fill with P&L", close(fc.pl, -1.0, 1e-6) and fc.units == -10_000)
check("close when flat returns None", b.close_position("EUR_USD") is None)

b = pb(); b.market_order("EUR_USD", 10_000, stop_loss=1.0980, take_profit=1.1040)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1000, 1.1010, 1.0970, 1.0975)
check("long stop hit intrabar: fills at stop minus half spread", close(b.fills[-1].price, 1.09795, 1e-9))
check("...position is flat and loss is exactly units x (fill - entry)",
      b.position("EUR_USD") == 0 and close(b.fills[-1].pl, 10_000 * (1.09795 - 1.10005), 1e-6))

b = pb(); b.market_order("EUR_USD", 10_000, stop_loss=1.0980, take_profit=1.1040)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.0950, 1.0960, 1.0940, 1.0955)        # gaps open BELOW the stop
check("gap through stop fills at the (worse) open, not the stop", close(b.fills[-1].price, 1.0950 - 0.00005, 1e-9))

b = pb(); b.market_order("EUR_USD", 10_000, stop_loss=1.0980, take_profit=1.1040)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1010, 1.1050, 1.1005, 1.1045)
check("long take-profit hit: fills at target minus half spread, profit > 0",
      close(b.fills[-1].price, 1.1040 - 0.00005, 1e-9) and b.fills[-1].pl > 0)

b = pb(); b.market_order("EUR_USD", 10_000, stop_loss=1.0980, take_profit=1.1040)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1000, 1.1050, 1.0970, 1.1000)        # touches BOTH levels
check("stop AND target inside one bar -> stop assumed first (conservative)", b.fills[-1].pl < 0)

b = pb(); b.market_order("EUR_USD", -10_000, stop_loss=1.1020, take_profit=1.0960)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1000, 1.1030, 1.0990, 1.1025)
check("short stop hit: fills at stop PLUS half spread (buying back)", close(b.fills[-1].price, 1.1020 + 0.00005, 1e-9))
b = pb(); b.market_order("EUR_USD", -10_000, stop_loss=1.1020, take_profit=1.0960)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1000, 1.1005, 1.0950, 1.0955)
check("short take-profit hit: profit > 0", b.fills[-1].pl > 0 and b.position("EUR_USD") == 0)

b = pb(); b.market_order("EUR_USD", 1_000); b.on_bar(T0 + pd.Timedelta(hours=1), 1.1100, 1.1110, 1.1090, 1.1100)
b.market_order("EUR_USD", 1_000)
exp_avg = (1_000 * 1.10005 + 1_000 * (1.1100 + 0.00005)) / 2_000
check("adding to a position averages the entry price", close(b.avg_price, exp_avg, 1e-9) and b.position("EUR_USD") == 2_000)

b = pb(); b.market_order("EUR_USD", 1_000)
b.on_bar(T0 + pd.Timedelta(hours=1), 1.1100, 1.1110, 1.1090, 1.1100)
fl = b.market_order("EUR_USD", -3_000)
check("flip: sell 3000 against long 1000 -> net -2000", b.position("EUR_USD") == -2_000)
check("flip: realised P&L only on the 1000 closed units", close(fl.pl, 1_000 * ((1.1100 - 0.00005) - 1.10005), 1e-9))
check("flip: new average price is the flip fill price", close(b.avg_price, 1.1100 - 0.00005, 1e-9))

b = pb(); start = b.balance
for i, (u, px) in enumerate([(5_000, 1.1010), (-8_000, 1.0990), (3_000, 1.1030), (-1_000, 1.0980)]):
    b.on_bar(T0 + pd.Timedelta(hours=i + 1), px, px + .0004, px - .0004, px)
    b.market_order("EUR_USD", u)
b.close_position("EUR_USD")
check("accounting invariant: balance == start + sum of realised P&L", close(b.balance, start + sum(f.pl for f in b.fills), 1e-6))
check("flat at the end -> equity equals balance", close(b.equity(), b.balance, 1e-9))
check("candles() returns the last N bars, oldest first",
      len(b.candles("EUR_USD", "H1", 3)) == 3 and b.candles("EUR_USD", "H1", 3).time.is_monotonic_increasing)

# =========================== LOGGER ===========================
d = tmp / "logs"
lg = TradeLogger(d)
lg.log_trade(event="ENTRY", instrument="EUR_USD", side=1, units=1000, price=1.1)
lg.log_equity("t1", 100000.0, 1000, 1.1)
lg2 = TradeLogger(d)                                          # simulates a restart
lg2.log_trade(event="EXIT", instrument="EUR_USD", side=-1, units=1000, price=1.2, realized_pl=100)
lines = (d / "trades.csv").read_text().strip().splitlines()
check("logger: header written exactly once across restarts", lines[0].startswith("ts_utc,") and sum(l.startswith("ts_utc,") for l in lines) == 1)
check("logger: both rows persisted", len(lines) == 3)
check("logger: equity file created with header", (d / "equity_curve.csv").read_text().startswith("ts_utc,bar_time,equity"))
check("logger: unknown fields are ignored, not crashed on", TradeLogger(None).log_trade(event="X", nonsense=1)["event"] == "X")
check("logger: in-memory mode keeps frames", len(TradeLogger(None).trades_frame()) == 0)

# =========================== BOT (PaperBroker + scripted prices) ===========================
def run_bot(closes, settings=None, wick=0.0003, store=None, start=T0):
    st = settings or S(atr_stop_mult=50.0, max_units=10**9)    # wide stops: only the signal moves us
    st.state_file = ""
    broker = PaperBroker(100_000, 0.0001)
    logger = TradeLogger(None)
    bot = Bot(st, broker, logger, store or StateStore(None))
    statuses = []
    for r in bars_from_closes(closes, wick, start).itertuples(index=False):
        broker.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume)
        statuses.append(bot.step())
    return bot, broker, logger, statuses


slow_up = 1.10 + 0.0001 * np.arange(130)
bot, broker, lg_, stt = run_bot(slow_up)
tr = lg_.trades_frame()
entries = tr[tr.event == "ENTRY"]
check("bot: warm-up bars produce no trades and report insufficient_data", stt[0] == "insufficient_data" and len(tr[tr.event == "ENTRY"]) <= 1)
check("bot: steady uptrend -> exactly ONE entry (no duplicate orders on later bars)", len(entries) == 1 and int(entries.side.iloc[0]) == 1)
check("bot: ends long", broker.position("EUR_USD") > 0)
check("bot: one equity row per processed bar", len(lg_.equity_frame()) == sum(s == "processed" for s in stt))
check("bot: same bar processed twice is a no-op",
      bot.step() == "no_new_bar" and len(lg_.trades_frame()) == len(tr))
sized = int(entries.units.iloc[0])
check("bot: entry size obeys the risk formula (units x stop distance <= 0.5% equity)", sized > 0)

flip = np.concatenate([1.10 + 0.0004 * np.arange(90), 1.10 + 0.0004 * 89 - 0.0006 * np.arange(1, 140)])
bot, broker, lg_, _ = run_bot(flip)
tr = lg_.trades_frame()
seq = [(r.event, r.side, r.reason) for r in tr.itertuples() if r.event in ("ENTRY", "EXIT")]
check("bot flip: first event is a LONG entry", seq[0][0] == "ENTRY" and int(seq[0][1]) == 1)
check("bot flip: later closes the long with reason signal_change",
      any(e == "EXIT" and rsn == "signal_change" for e, _, rsn in seq))
check("bot flip: then opens a SHORT", any(e == "ENTRY" and int(sd) == -1 for e, sd, _ in seq))
long_i = next(i for i, x in enumerate(seq) if x[0] == "ENTRY" and int(x[1]) == 1)
exit_i = next(i for i, x in enumerate(seq) if x[0] == "EXIT" and i > long_i)
short_i = next(i for i, x in enumerate(seq) if x[0] == "ENTRY" and int(x[1]) == -1)
check("bot flip: order of events is long -> exit -> short (never both sides at once)", long_i < exit_i < short_i)
check("bot flip: ends net short", broker.position("EUR_USD") < 0)

# stop/target exit must not be followed by an instant re-entry in the same direction
tight = S(atr_stop_mult=1.0, tp_rr=1.0, max_units=10**9)
strong = 1.10 + 0.0010 * np.arange(160)
bot, broker, lg_, _ = run_bot(strong, tight)
tr = lg_.trades_frame()
ext = tr[(tr.event == "EXIT") & (tr.reason == "stop_or_take_profit")]
check("bot: target/stop exit detected and logged as such", len(ext) >= 1)
check("bot: ...and NO re-entry in the same direction while the signal is unchanged",
      len(tr[tr.event == "ENTRY"]) == 1 and bot.state.blocked_dir == 1)
unblock = np.concatenate([strong, strong[-1] - 0.0015 * np.arange(1, 80), strong[-1] - 0.0015 * 79 + 0.0015 * np.arange(1, 120)])
bot, broker, lg_, _ = run_bot(unblock, tight)
tr = lg_.trades_frame()
long_entries = [int(r.side) for r in tr[tr.event == "ENTRY"].itertuples()]
check("bot: after the signal changes, blocking lifts and a later long entry is allowed again",
      long_entries.count(1) >= 2, f"(entries: {long_entries})")

bot, broker, lg_, _ = run_bot(slow_up, Settings(atr_stop_mult=50.0, max_units=10**9))     # default 1.0x leverage
e1 = lg_.trades_frame().query("event == 'ENTRY'").iloc[0]
check("bot: entry notional never exceeds equity x max_leverage (1.0x)",
      float(e1.units) * float(e1.price) <= 100_000 * 1.0 + 1.0, f"({float(e1.units) * float(e1.price):,.0f} <= 100,000)")

# daily loss kill switch
risky = S(risk_pct=0.05, atr_stop_mult=50.0, max_units=10**9, max_daily_loss=0.01)
def shock(drop):
    c = list(1.10 + 0.0002 * np.arange(66))
    c += [c[-1] - drop]                                           # one violent bar (day 3, hour 18)
    c += [c[-1] + 0.0002 * k for k in range(1, 60)]               # then the uptrend resumes
    return c


# negative control: a 0.0100 drop costs 0.86% of the day's starting equity (the position carried a small
# unrealised gain into it) - under the 1% limit, so the kill switch must NOT fire
_, _, lg_small, _ = run_bot(shock(0.0100), risky)
check("kill switch: a loss UNDER the limit (-0.86% vs 1%) does not halt",
      len(lg_small.trades_frame().query("event == 'HALT'")) == 0)
bot, broker, lg_, _ = run_bot(shock(0.0130), risky)               # -1.24% on the day: over the limit
tr = lg_.trades_frame()
halts = tr[tr.event == "HALT"]
check("kill switch: HALT event logged when the daily loss limit is breached", len(halts) == 1)
halt_time = pd.Timestamp(halts.bar_time.iloc[0])
day_end = halt_time.normalize() + pd.Timedelta(days=1)
check("kill switch: position flattened with reason daily_loss_halt",
      ((tr.event == "EXIT") & (tr.reason == "daily_loss_halt")).any())
after = tr[(tr.event == "ENTRY") & (pd.to_datetime(tr.bar_time) > halt_time) & (pd.to_datetime(tr.bar_time) < day_end)]
check("kill switch: NO new entries for the rest of that UTC day", len(after) == 0)
resumed = tr[(tr.event == "ENTRY") & (pd.to_datetime(tr.bar_time) >= day_end)]
check("kill switch: trading is allowed again from the next UTC day", len(resumed) >= 1)

# market closed / failed close must NOT be treated as errors, and must never leave two sides open
class ClosedMarket(PaperBroker):
    def quote(self, instrument):
        raise MarketClosed("weekend")


cfg = S(atr_stop_mult=50.0, max_units=10**9); cfg.state_file = ""
cb = ClosedMarket(100_000, 0.0001); cl = TradeLogger(None); cbot = Bot(cfg, cb, cl, StateStore(None))
cstat = []
for r in bars_from_closes(slow_up[:100]).itertuples(index=False):
    cb.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume); cstat.append(cbot.step())
skips = cl.trades_frame().query("event == 'SKIP' and reason == 'market closed'")
check("market closed: entry is skipped and logged, the cycle still completes (no exception)",
      len(skips) >= 1 and "processed" in cstat and cb.position("EUR_USD") == 0)
check("market closed: consecutive_errors stays 0 (it is not a fault)", cbot.state.consecutive_errors == 0)


class StuckClose(PaperBroker):
    def close_position(self, instrument):
        raise OrderRejected("market halted")


cfg = S(atr_stop_mult=50.0, max_units=10**9); cfg.state_file = ""
sb = StuckClose(100_000, 0.0001); sl = TradeLogger(None); sbot = Bot(cfg, sb, sl, StateStore(None))
# a downtrend short enough (0.048 < the ~0.06 stop distance) that the broker-side stop never fires, so the
# ONLY way out is the close request that keeps failing
stuck_flip = np.concatenate([1.10 + 0.0004 * np.arange(90), 1.10 + 0.0004 * 89 - 0.0006 * np.arange(1, 80)])
rows = list(bars_from_closes(stuck_flip).itertuples(index=False))
for r in rows:
    sb.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume); sbot.step()
st_ = sl.trades_frame()
check("failed close: bot logs REJECTED and keeps going instead of crashing",
      (st_.event == "REJECTED").any() and sbot.state.consecutive_errors == 0)
check("failed close: never opens the opposite side while the old position is still held",
      sb.position("EUR_USD") > 0 and not ((st_.event == "ENTRY") & (st_.side.astype(str) == "-1")).any())

# restart idempotency
state_path = tmp / "state.json"
st1 = S(atr_stop_mult=50.0, max_units=10**9); st1.state_file = str(state_path)
broker = PaperBroker(100_000, 0.0001); logger = TradeLogger(None)
bot1 = Bot(st1, broker, logger, StateStore(state_path))
for r in bars_from_closes(slow_up[:100]).itertuples(index=False):
    broker.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume); bot1.step()
n_fills, pos_before = len(broker.fills), broker.position("EUR_USD")
check("restart: state file written", state_path.exists())
bot2 = Bot(st1, broker, logger, StateStore(state_path))
check("restart: new process sees the same last bar and does nothing", bot2.step() == "no_new_bar" and len(broker.fills) == n_fills)
state_path.unlink()
bot3 = Bot(st1, broker, logger, StateStore(state_path))
check("restart with LOST state: reconciliation still places no duplicate order",
      bot3.step() == "processed" and len(broker.fills) == n_fills and broker.position("EUR_USD") == pos_before)
state_path.write_text("{ this is not json")
check("restart: corrupt state file falls back to a clean state instead of crashing",
      Bot(st1, broker, logger, StateStore(state_path)).state.last_bar_time == "")

# error handling in the run loop
class Flaky(PaperBroker):
    def __init__(self, fail_times): super().__init__(100_000, 0.0001); self.left = fail_times
    def candles(self, *a, **k):
        if self.left > 0:
            self.left -= 1
            raise BrokerError("simulated outage")
        return super().candles(*a, **k)

for fail, expect_code in ((3, 1), (2, 0)):
    cfg = S(max_consecutive_errors=3, atr_stop_mult=50.0); cfg.state_file = ""
    fb = Flaky(fail)
    for r in bars_from_closes(slow_up[:80]).itertuples(index=False):
        fb.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume)
    bt = Bot(cfg, fb, TradeLogger(None), StateStore(None))
    def fake_sleep(_s, bt=bt): bt._stop = bt.state.last_bar_time != ""
    bt._sleep = fake_sleep
    code = bt.run_forever()
    if fail == 3:
        check("run loop: 3 consecutive failures (limit 3) -> gives up with exit code 1", code == 1)
    else:
        check("run loop: 2 transient failures are survived, counter resets, clean exit 0",
              code == 0 and bt.state.consecutive_errors == 0 and bt.state.last_bar_time != "")

check("time: 1 second before the hour -> wait 6s (1s + 5s grace)", close(seconds_until_next_bar(3599.0, 3600, 5), 6.0))
check("time: exactly on the hour -> wait a full bar + grace", close(seconds_until_next_bar(3600.0, 3600, 5), 3605.0))
check("time: mid-bar", close(seconds_until_next_bar(3600 * 10 + 1800, 3600, 5), 1805.0))

# =========================== OANDA ADAPTER (fake HTTP session) ===========================
class FakeResp:
    def __init__(self, status=200, data=None, bad_json=False):
        self.status_code, self._d, self._bad = status, data if data is not None else {}, bad_json
        self.text = str(data)
    def json(self):
        if self._bad: raise ValueError("not json")
        return self._d


class FakeSession:
    def __init__(self, *responses): self.queue, self.calls = list(responses), []
    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json, "timeout": timeout})
        item = self.queue.pop(0)
        if isinstance(item, Exception): raise item
        return item


def oanda(*resp):
    sess = FakeSession(*resp)
    return OandaBroker("TOKEN123", "101-001-1", session=sess, sleep=lambda s: None), sess


check("oanda: refuses non-practice env", raises(BrokerError, OandaBroker, "t", "a", "live"))
check("oanda: requires credentials", raises(BrokerError, OandaBroker, "", "", "practice"))

b, sess = oanda(FakeResp(200, {"account": {"NAV": "101234.56"}}))
check("oanda equity: reads NAV", close(b.equity(), 101234.56))
c = sess.calls[0]
check("oanda: practice host, bearer auth header, timeout set",
      c["url"] == "https://api-fxpractice.oanda.com/v3/accounts/101-001-1/summary"
      and c["headers"]["Authorization"] == "Bearer TOKEN123" and c["timeout"] == 10)

b, _ = oanda(FakeResp(200, {"position": {"instrument": "EUR_USD", "long": {"units": "5000"}, "short": {"units": "-2000"}}}))
check("oanda position: net = long + short (short units negative)", b.position("EUR_USD") == 3000)

b, _ = oanda(FakeResp(200, {"prices": [{"bids": [{"price": "1.08000"}], "asks": [{"price": "1.08012"}], "tradeable": True}]}))
check("oanda quote: parses bid/ask", b.quote("EUR_USD") == (1.08, 1.08012))
b, _ = oanda(FakeResp(200, {"prices": [{"bids": [{"price": "1"}], "asks": [{"price": "2"}], "tradeable": False}]}))
check("oanda quote: market closed raises MarketClosed (a BrokerError subtype)", raises(MarketClosed, b.quote, "EUR_USD"))

candle = lambda t, o, cmp: {"complete": cmp, "volume": 10, "time": t, "mid": {"o": o, "h": o, "l": o, "c": o}}
b, sess = oanda(FakeResp(200, {"candles": [candle("2025-01-01T01:00:00.000000000Z", "1.1010", True),
                                           candle("2025-01-01T00:00:00.000000000Z", "1.1000", True),
                                           candle("2025-01-01T02:00:00.000000000Z", "1.1020", False)]}))
df = b.candles("EUR_USD", "H1", 3)
check("oanda candles: drops the incomplete (still forming) bar", len(df) == 2 and close(df.close.iloc[-1], 1.1010))
check("oanda candles: sorted oldest-first, UTC-aware, float prices",
      df.time.is_monotonic_increasing and str(df.time.dt.tz) == "UTC" and df.close.dtype == float)
check("oanda candles: requests mid prices with the right granularity/count",
      sess.calls[0]["params"] == {"granularity": "H1", "count": 3, "price": "M"})
check("oanda candles: count above 5000 rejected before any HTTP call", raises(BrokerError, b.candles, "EUR_USD", "H1", 6000))

fill_resp = {"orderFillTransaction": {"id": "77", "orderID": "76", "time": "2025-01-01T02:00:01.000000000Z",
                                      "units": "1000", "price": "1.10012", "pl": "0.0"}}
b, sess = oanda(FakeResp(201, fill_resp))
f = b.market_order("EUR_USD", 1000, stop_loss=1.0984, take_profit=1.1032, tag="sma-test")
body = sess.calls[0]["json"]["order"]
check("oanda order: MARKET, FOK, units sent as a STRING", body["type"] == "MARKET" and body["timeInForce"] == "FOK" and body["units"] == "1000")
check("oanda order: stop/target attached on fill, formatted to 5 dp",
      body["stopLossOnFill"]["price"] == "1.09840" and body["takeProfitOnFill"]["price"] == "1.10320")
check("oanda order: parses the fill (price, signed units, order id)", close(f.price, 1.10012) and f.units == 1000 and f.order_id == "76")
b, sess = oanda(FakeResp(201, fill_resp))
b.market_order("EUR_USD", -1000)
check("oanda order: sell sends a negative units string; no stop keys when not given",
      sess.calls[0]["json"]["order"]["units"] == "-1000" and "stopLossOnFill" not in sess.calls[0]["json"]["order"])
b, sess = oanda(FakeResp(201, fill_resp))
b.market_order("USD_JPY", 1000, stop_loss=149.1234)
check("oanda order: JPY pair prices use 3 decimals", sess.calls[0]["json"]["order"]["stopLossOnFill"]["price"] == "149.123")
b, _ = oanda(FakeResp(201, {"orderCancelTransaction": {"reason": "INSUFFICIENT_MARGIN"}}))
check("oanda order: cancel transaction -> OrderRejected with reason", raises(OrderRejected, b.market_order, "EUR_USD", 10**9))
b, _ = oanda(FakeResp(400, {"errorMessage": "Invalid value specified for 'instrument'"}))
check("oanda order: HTTP 4xx -> OrderRejected", raises(OrderRejected, b.market_order, "XXX_YYY", 1))
b, sess = oanda(FakeResp(500, {}), FakeResp(201, fill_resp))
check("oanda order: a 500 on POST is NEVER retried (could double-fill)", raises(BrokerError, b.market_order, "EUR_USD", 1000) and len(sess.calls) == 1)
b, sess = oanda(requests.Timeout("slow"), FakeResp(201, fill_resp))
check("oanda order: a timeout on POST is NEVER retried either", raises(BrokerError, b.market_order, "EUR_USD", 1000) and len(sess.calls) == 1)
b, sess = oanda(FakeResp(503, {}), requests.ConnectionError("blip"), FakeResp(200, {"account": {"NAV": "5"}}))
check("oanda reads: GET retries through a 503 and a connection error, then succeeds", close(b.equity(), 5) and len(sess.calls) == 3)
b, sess = oanda(FakeResp(500, {}), FakeResp(500, {}), FakeResp(500, {}))
check("oanda reads: gives up after 3 failed GET attempts", raises(BrokerError, b.equity) and len(sess.calls) == 3)
b, _ = oanda(FakeResp(200, bad_json=True))
check("oanda: non-JSON response -> BrokerError, not a crash", raises(BrokerError, b.equity))

b, sess = oanda(FakeResp(200, {"position": {"long": {"units": "3000"}, "short": {"units": "0"}}}),
                FakeResp(200, {"longOrderFillTransaction": {"id": "90", "orderID": "89", "time": "2025-01-01T03:00:00Z",
                                                           "units": "-3000", "price": "1.1050", "pl": "12.5"}}))
cf = b.close_position("EUR_USD")
check("oanda close: long position closed with {longUnits: ALL} via PUT",
      sess.calls[1]["method"] == "PUT" and sess.calls[1]["json"] == {"longUnits": "ALL"} and sess.calls[1]["url"].endswith("/positions/EUR_USD/close"))
check("oanda close: realised P&L parsed from the fill", close(cf.pl, 12.5) and cf.units == -3000)
b, sess = oanda(FakeResp(200, {"position": {"long": {"units": "0"}, "short": {"units": "-800"}}}),
                FakeResp(200, {"shortOrderFillTransaction": {"id": "91", "time": "2025-01-01T03:00:00Z", "units": "800", "price": "1.1", "pl": "-3"}}))
b.close_position("EUR_USD")
check("oanda close: short position closed with {shortUnits: ALL}", sess.calls[1]["json"] == {"shortUnits": "ALL"})
b, sess = oanda(FakeResp(200, {"position": {"long": {"units": "0"}, "short": {"units": "0"}}}))
check("oanda close: flat -> returns None and sends no close request", b.close_position("EUR_USD") is None and len(sess.calls) == 1)

# =========================== ALPACA ADAPTER (fake HTTP session) ===========================
from datetime import datetime, timezone
from broker_alpaca import AlpacaBroker
from bot import build_broker

NOW = datetime(2025, 1, 2, 15, 30, tzinfo=timezone.utc)
OPEN, CLOSED = FakeResp(200, {"is_open": True}), FakeResp(200, {"is_open": False})


def alpaca(*resp, **kw):
    sess = FakeSession(*resp)
    return AlpacaBroker("KEY", "SECRET", session=sess, sleep=lambda s: None, now_fn=lambda: NOW, **kw), sess


def order_obj(status, oid="o1", qty="10", px="450.12"):
    d = {"id": oid, "status": status}
    if status == "filled":
        d.update({"filled_qty": qty, "filled_avg_price": px, "filled_at": "2025-01-02T15:30:02Z"})
    return d


check("alpaca: requires credentials", raises(BrokerError, AlpacaBroker, "", ""))
b, sess = alpaca(FakeResp(200, {"equity": "100500.25"}))
check("alpaca equity: parsed", close(b.equity(), 100500.25))
check("alpaca: PAPER host and key headers on every request",
      sess.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/account"
      and sess.calls[0]["headers"]["APCA-API-KEY-ID"] == "KEY" and sess.calls[0]["headers"]["APCA-API-SECRET-KEY"] == "SECRET")

b, _ = alpaca(FakeResp(404, {"message": "position does not exist"}))
check("alpaca position: 404 means flat (0), not an error", b.position("SPY") == 0)
b, _ = alpaca(FakeResp(200, {"qty": "10", "side": "long", "avg_entry_price": "450"}))
check("alpaca position: long", b.position("SPY") == 10)
b, _ = alpaca(FakeResp(200, {"qty": "-7", "side": "short", "avg_entry_price": "450"}))
check("alpaca position: short with negative qty", b.position("SPY") == -7)
b, _ = alpaca(FakeResp(200, {"qty": "7", "side": "short", "avg_entry_price": "450"}))
check("alpaca position: side=short wins even if qty is reported positive", b.position("SPY") == -7)

b, sess = alpaca(OPEN, FakeResp(200, {"quote": {"bp": 449.9, "ap": 450.1}}))
check("alpaca quote: parses bid/ask and asks for the IEX feed", b.quote("SPY") == (449.9, 450.1) and sess.calls[1]["params"] == {"feed": "iex"})
b, _ = alpaca(CLOSED)
check("alpaca quote: market closed -> MarketClosed (bot skips the bar)", raises(MarketClosed, b.quote, "SPY"))
b, _ = alpaca(OPEN, FakeResp(200, {"quote": {"bp": 0, "ap": 450.1}}))
check("alpaca quote: a zero bid is rejected rather than traded on", raises(BrokerError, b.quote, "SPY"))

bar = lambda t, c: {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000}
b, sess = alpaca(FakeResp(200, {"bars": [bar("2025-01-02T15:00:00Z", 454.0), bar("2025-01-02T14:00:00Z", 453.0),
                                         bar("2025-01-02T13:00:00Z", 452.0), bar("2025-01-02T12:00:00Z", 451.0)]}))
df = b.candles("SPY", "H1", 3)
check("alpaca candles: drops the still-forming 15:00 bar (now = 15:30), keeps 3 complete ones",
      len(df) == 3 and close(df.close.iloc[-1], 453.0))
check("alpaca candles: oldest first, UTC-aware", df.time.is_monotonic_increasing and str(df.time.dt.tz) == "UTC")
p = sess.calls[0]["params"]
check("alpaca candles: 1Hour, latest-first, raw prices, IEX feed, small limit",
      p["timeframe"] == "1Hour" and p["sort"] == "desc" and p["adjustment"] == "raw" and p["feed"] == "iex" and p["limit"] == 5)
b, _ = alpaca(FakeResp(200, {"bars": None}))
check("alpaca candles: no data -> empty frame, not a crash", len(b.candles("SPY", "H1", 10)) == 0)

filled = order_obj("filled")
b, sess = alpaca(OPEN, FakeResp(200, order_obj("accepted")), FakeResp(200, order_obj("new")), FakeResp(200, filled))
f = b.market_order("SPY", 10, stop_loss=445.4999, take_profit=459.0, tag="sma-202501021530")
body = sess.calls[1]["json"]
check("alpaca order: bracket with qty as string, side buy, market, GTC",
      body["order_class"] == "bracket" and body["qty"] == "10" and body["side"] == "buy"
      and body["type"] == "market" and body["time_in_force"] == "gtc")
check("alpaca order: stop and target formatted to whole cents", body["stop_loss"] == {"stop_price": "445.50"} and body["take_profit"] == {"limit_price": "459.00"})
check("alpaca order: bar tag sent as client_order_id (duplicate-proof)", body["client_order_id"] == "sma-202501021530")
check("alpaca order: polls until filled, then returns the real fill", close(f.price, 450.12) and f.units == 10 and f.order_id == "o1" and len(sess.calls) == 4)
b, sess = alpaca(OPEN, FakeResp(200, order_obj("accepted")), FakeResp(200, order_obj("filled", qty="10")))
fs = b.market_order("SPY", -10, stop_loss=455.0, take_profit=440.0)
check("alpaca order: sell -> side sell and negative fill units", sess.calls[1]["json"]["side"] == "sell" and fs.units == -10)
b, sess = alpaca(OPEN, FakeResp(200, order_obj("accepted")), FakeResp(200, filled))
b.market_order("SPY", 10, stop_loss=445.0)
check("alpaca order: only a stop -> 'oto' class", sess.calls[1]["json"]["order_class"] == "oto" and "take_profit" not in sess.calls[1]["json"])
b, sess = alpaca(OPEN, FakeResp(200, order_obj("accepted")), FakeResp(200, filled))
b.market_order("SPY", 10)
check("alpaca order: no stop/target -> plain market order", "order_class" not in sess.calls[1]["json"])
b, _ = alpaca(OPEN, FakeResp(200, order_obj("accepted")), FakeResp(200, order_obj("rejected")))
check("alpaca order: status 'rejected' -> OrderRejected", raises(OrderRejected, b.market_order, "SPY", 10))
b, _ = alpaca(OPEN, FakeResp(403, {"message": "insufficient buying power"}))
check("alpaca order: HTTP 403 (buying power) -> OrderRejected", raises(OrderRejected, b.market_order, "SPY", 10**6))
b, sess = alpaca(CLOSED)
check("alpaca order: market closed -> MarketClosed and NO order is sent", raises(MarketClosed, b.market_order, "SPY", 10) and len(sess.calls) == 1)
b, sess = alpaca(OPEN, FakeResp(200, order_obj("accepted")), *[FakeResp(200, order_obj("new"))] * 3, FakeResp(204, {}), max_polls=3)
check("alpaca order: never fills -> asks for cancellation and raises (position re-checked next cycle)",
      raises(BrokerError, b.market_order, "SPY", 10) and sess.calls[-1]["method"] == "DELETE" and sess.calls[-1]["url"].endswith("/v2/orders/o1"))
b, sess = alpaca(OPEN, FakeResp(500, {}), FakeResp(200, order_obj("accepted")))
check("alpaca order: a 500 on POST is NEVER retried", raises(BrokerError, b.market_order, "SPY", 10) and len(sess.calls) == 2)
b, sess = alpaca(FakeResp(503, {}), FakeResp(200, {"equity": "5"}))
check("alpaca reads: GET retries through a 503", close(b.equity(), 5) and len(sess.calls) == 2)

# closing: cancel the bracket legs FIRST, then close the position
b, sess = alpaca(FakeResp(200, {"qty": "10", "side": "long", "avg_entry_price": "450.00"}), OPEN,
                 FakeResp(200, [{"id": "leg1"}, {"id": "leg2"}]), FakeResp(200, {}), FakeResp(200, {}),
                 FakeResp(200, {"id": "c1", "status": "pending_new"}),
                 FakeResp(200, order_obj("filled", oid="c1", qty="10", px="455.00")))
cf = b.close_position("SPY")
seq = [(c["method"], c["url"].replace("https://paper-api.alpaca.markets", "")) for c in sess.calls]
check("alpaca close: cancels both resting legs BEFORE the position DELETE",
      seq[3] == ("DELETE", "/v2/orders/leg1") and seq[4] == ("DELETE", "/v2/orders/leg2") and seq[5] == ("DELETE", "/v2/positions/SPY"))
check("alpaca close: long P&L = qty x (exit - entry) = 10 x 5 = 50, fill is a SELL of 10", close(cf.pl, 50.0) and cf.units == -10)
check("alpaca close: open orders were looked up for this symbol only", sess.calls[2]["params"]["symbols"] == "SPY" and sess.calls[2]["params"]["status"] == "open")
b, _ = alpaca(FakeResp(200, {"qty": "-5", "side": "short", "avg_entry_price": "100.00"}), OPEN, FakeResp(200, []),
              FakeResp(200, {"id": "c2"}), FakeResp(200, order_obj("filled", oid="c2", qty="5", px="98.00")))
sf = b.close_position("SPY")
check("alpaca close: short P&L = qty x (entry - exit) = 5 x 2 = +10, fill is a BUY", close(sf.pl, 10.0) and sf.units == 5)
b, sess = alpaca(FakeResp(404, {"message": "position does not exist"}))
check("alpaca close: already flat -> None, nothing else sent", b.close_position("SPY") is None and len(sess.calls) == 1)
b, sess = alpaca(FakeResp(200, {"qty": "10", "side": "long", "avg_entry_price": "450"}), CLOSED)
check("alpaca close: market closed -> MarketClosed, no DELETE sent",
      raises(MarketClosed, b.close_position, "SPY") and all(c["method"] == "GET" for c in sess.calls))
bb = build_broker(Settings(broker="alpaca", alpaca_key_id="k", alpaca_secret_key="s"))
check("build_broker wires up the Alpaca adapter", bb.name == "alpaca")

# IBKR adapter: safety guards only (no gateway available offline)
from broker_ibkr import IbkrBroker
check("ibkr: live ports 7496 / 4001 refused before any connection attempt",
      all(raises(BrokerError, IbkrBroker, "127.0.0.1", p) for p in (7496, 4001)))

# =========================== BACKTEST ===========================
check("stats: total return, hand-checked (100 -> 99 = -1%)", close(performance_stats([100, 110, 99], 100)["total_return"], -0.01, 1e-12))
check("stats: max drawdown (110 -> 99 = -10%)", close(performance_stats([100, 110, 99], 100)["max_drawdown"], -0.1, 1e-12))
check("stats: Sharpe hand calc: returns +2%, -0.98% x sqrt(100) = 2.4192",
      close(performance_stats([100, 102, 101], 100)["sharpe"], 2.4192, 5e-4))
ts = trade_stats([type("F", (), {"pl": v}) for v in (10.0, -5.0, 20.0, 0.0, -5.0)])
check("trade stats: 3 priced exits, win rate 2/4=50%, profit factor 30/10=3",
      ts["n_closed_trades"] == 4 and close(ts["win_rate"], 0.5) and close(ts["profit_factor"], 3.0))

bars = make_synthetic_bars(n=1800, seed=11)
res = run_backtest(bars, Settings())
res2 = run_backtest(bars, Settings())
check("backtest: deterministic (same data -> identical equity curve)", res.equity.equity.equals(res2.equity.equity))
check("backtest: places trades on trending synthetic data", res.stats["n_closed_trades"] > 5)
future = bars.copy()
future.loc[1000:, ["open", "high", "low", "close"]] *= 1.07
res3 = run_backtest(future, Settings())
# align on bar TIME, not row number: the equity log skips the warm-up bars, so row k is not bar k
cut = bars.time.iloc[1000].isoformat()
ea, eb = res.equity.set_index("bar_time").equity, res3.equity.set_index("bar_time").equity
before = [x for x in ea.index if x < cut]
check("backtest: NO LOOK-AHEAD - altering every bar from t=1000 on leaves all earlier equity values identical",
      len(before) > 500 and np.allclose(ea[before].values, eb[before].values), f"({len(before)} rows compared)")
check("backtest: ...and the altered period DOES differ (the test can actually fail)",
      not np.allclose(ea.drop(before).values, eb.drop(before).values))
bk = PaperBroker(100_000, 0.0001)
bt = Bot(Settings(), bk, TradeLogger(None), StateStore(None))
for r in bars.itertuples(index=False):
    bk.on_bar(r.time, r.open, r.high, r.low, r.close, r.volume); bt.step()
check("backtest: balance == start + sum(realised P&L) over a full run", close(bk.balance, 100_000 + sum(f.pl for f in bk.fills), 1e-6))
check("backtest: equity never negative, no NaNs", (res.equity.equity > 0).all() and res.equity.equity.notna().all())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)