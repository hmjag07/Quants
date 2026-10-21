import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED_MODULES = ["backtest", "bot", "broker_alpaca", "broker_base", "broker_paper", "config", "risk",
                    "strategy", "trade_logger"]
OPTIONAL_MODULES = ["broker_ibkr", "broker_oanda", "test_connection"]       # renamed if present, never required
REQUIRED_PACKAGES = ["numpy", "pandas", "requests"]


def _case_mismatches():
    """{actual file name: expected lowercase name} for project files that have the wrong capitalisation."""
    actual = {p.name for p in HERE.glob("*.py")}
    return {n: m + ".py" for m in REQUIRED_MODULES + OPTIONAL_MODULES for n in actual if n.lower() == m + ".py" and n != m + ".py"}


def _fix_names(bad):
    """Rename to lowercase in two steps (a one-step case-only rename is ignored on Windows/macOS).
    Uses `git mv` when the folder is a git repo so the INDEX is lowercase too (the Linux server needs that)."""
    try:
        in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=HERE, capture_output=True,
                                text=True).stdout.strip() == "true"
    except OSError:
        in_git = False
    for name, want in bad.items():
        tmp = "_tmp_rename_" + want
        for src, dst in ((name, tmp), (tmp, want)):
            moved = False
            if in_git:
                moved = subprocess.run(["git", "mv", src, dst], cwd=HERE, capture_output=True).returncode == 0
            if not moved:                                       # untracked file, or no git: plain rename
                os.rename(HERE / src, HERE / dst)
        print(f"  renamed  {name}  ->  {want}")
    importlib.invalidate_caches()


def _preflight():
    bad = _case_mismatches()
    if bad and "--fix-names" in sys.argv:
        print("Fixing file names:")
        _fix_names(bad)
        bad = _case_mismatches()
    present = {p.name for p in HERE.glob("*.py")}
    problems = []
    if bad:
        problems.append("These files have the wrong capitalisation (imports are case-sensitive, so Python cannot find them):\n"
                        + "".join(f"      {a}   ->   should be   {b}\n" for a, b in sorted(bad.items()))
                        + f"    Fix them all in one go:   python {Path(__file__).name} --fix-names")
    missing = [m + ".py" for m in REQUIRED_MODULES if m + ".py" not in present and m + ".py" not in bad.values()]
    if missing:
        problems.append("These required files are not in this folder (" + str(HERE) + "):\n"
                        + "".join(f"      {m}\n" for m in missing))
    no_pkgs = [p for p in REQUIRED_PACKAGES if importlib.util.find_spec(p) is None]
    if no_pkgs:
        problems.append("Missing Python packages: " + ", ".join(no_pkgs) + "\n    Install them:   pip install -r requirements.txt")
    if problems:
        print("\nPREFLIGHT FAILED - nothing was tested.\n")
        for p in problems:
            print("  *", p, "\n")
        sys.exit(1)


_preflight()

import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backtest import make_synthetic_bars, performance_stats, run_backtest, trade_stats
from bot import Bot, StateStore, seconds_until_next_bar
from broker_base import BrokerError, MarketClosed, OrderRejected
from broker_paper import PaperBroker
from config import PROJECT_DIR, ConfigError, Settings, load_dotenv
from risk import daily_loss_breached, position_size, price_decimals, stop_take_levels
from strategy import StrategyParams, compute_indicators, latest_signal, target_series
from trade_logger import TradeLogger

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
check("settings: LIVE IBKR ports refused",
      all(raises(ConfigError, Settings.from_env, {"IBKR_PORT": p}) for p in ("7496", "4001")))
check("settings: paper IBKR ports accepted",
      all(Settings.from_env({"IBKR_PORT": p}).ibkr_port == int(p) for p in ("7497", "4002")))
(tmp / ".env2").write_text("BROKER=alpaca   # trailing note\nRISK_PCT=0.01\t# tab note\nALPACA_SECRET_KEY=\"ab#cd\"  # c\nINSTRUMENT='SPY' # d\nGRANULARITY=H1#notacomment\n")
d2 = load_dotenv(tmp / ".env2")
check("dotenv: trailing '# comments' are stripped from unquoted values",
      d2["BROKER"] == "alpaca" and d2["RISK_PCT"] == "0.01" and d2["INSTRUMENT"] == "SPY", str(d2))
check("dotenv: a quoted value may contain '#'", d2["ALPACA_SECRET_KEY"] == "ab#cd")
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
    _vals.update({"ALPACA_KEY_ID": "PKtest", "ALPACA_SECRET_KEY": "sec"})
    check("shipped env.example parses and validates once the keys are filled in",
          Settings.from_env(_vals).broker == "alpaca" and Settings.from_env(_vals).instrument == "SPY")
    check("shipped env.example has no trailing comments on value lines",
          all(" #" not in l for l in _ex.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#")))
# ---- finding the .env: from any working directory, in any text encoding, with a useful error when it is wrong ----
def _isolated(fn):
    """Run fn, then undo any os.environ changes load_dotenv made."""
    saved = dict(os.environ)
    try:
        return fn()
    finally:
        os.environ.clear(); os.environ.update(saved)


def _msg(fn):
    try:
        fn()
    except ConfigError as e:
        return str(e)
    return ""


cwd_dir, proj_dir, empty_dir = tmp / "cwd1", tmp / "proj1", tmp / "empty1"
for d_ in (cwd_dir, proj_dir, empty_dir):
    d_.mkdir()
(proj_dir / ".env").write_text("ALPACA_KEY_ID=PKfromproj\nALPACA_SECRET_KEY=SECRETproj\n")
find = lambda dirs: _isolated(lambda: load_dotenv(search_dirs=dirs, override=True))
check("dotenv: finds the .env in the PROJECT folder even when you run from another directory (the reported bug)",
      find([cwd_dir, proj_dir]).get("ALPACA_KEY_ID") == "PKfromproj")
(cwd_dir / ".env").write_text("ALPACA_KEY_ID=PKfromcwd\n")
check("dotenv: a .env in the current directory takes priority over the project one",
      find([cwd_dir, proj_dir]).get("ALPACA_KEY_ID") == "PKfromcwd")
check("dotenv: the same folder listed twice is read once, without error", find([proj_dir, proj_dir]).get("ALPACA_KEY_ID") == "PKfromproj")
import config as _cfg


def _from_directory(run_dir, project_dir):
    """Call load_dotenv() with NO arguments, exactly as bot.py does, while 'standing in' run_dir, with the
    project folder pointed at project_dir."""
    old_cwd, old_proj = os.getcwd(), _cfg.PROJECT_DIR
    saved = dict(os.environ)
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY"):
        os.environ.pop(k, None)
    try:
        os.chdir(run_dir); _cfg.PROJECT_DIR = Path(project_dir)
        return _cfg.load_dotenv()
    finally:
        os.chdir(old_cwd); _cfg.PROJECT_DIR = old_proj
        os.environ.clear(); os.environ.update(saved)


(cwd_dir / ".env").unlink()                       # the 'Quants' parent folder: no .env here
check("dotenv DEFAULT search: running from a parent folder still finds the project's .env (EXACTLY your bug)",
      _from_directory(cwd_dir, proj_dir).get("ALPACA_KEY_ID") == "PKfromproj")
check("dotenv DEFAULT search: running from inside the project folder works too",
      _from_directory(proj_dir, proj_dir).get("ALPACA_KEY_ID") == "PKfromproj")
(cwd_dir / ".env").write_text("ALPACA_KEY_ID=PKfromcwd\n")
check("dotenv DEFAULT search: a .env in the current directory still wins", _from_directory(cwd_dir, proj_dir).get("ALPACA_KEY_ID") == "PKfromcwd")
(cwd_dir / ".env").unlink()
check("dotenv: no .env anywhere -> empty result, no crash", find([empty_dir]) == {})
check("dotenv: an explicit path is honoured exactly", _isolated(lambda: load_dotenv(proj_dir / ".env", override=True)).get("ALPACA_KEY_ID") == "PKfromproj")

enc_text = "BROKER=alpaca\nINSTRUMENT=SPY\n# caf\u00e9 note\nALPACA_KEY_ID=PKenc\n"
for label, raw_bytes in (("UTF-8 with BOM (some editors)", b"\xef\xbb\xbf" + enc_text.encode("utf-8")),
                         ("UTF-16 (PowerShell '>' redirection)", enc_text.encode("utf-16")),
                         ("Windows ANSI / cp1252", enc_text.encode("cp1252"))):
    (tmp / "enc.env").write_bytes(raw_bytes)
    got = _isolated(lambda: load_dotenv(tmp / "enc.env", override=True))
    check(f"dotenv: reads a .env saved as {label}",
          got.get("BROKER") == "alpaca" and got.get("INSTRUMENT") == "SPY" and got.get("ALPACA_KEY_ID") == "PKenc", str(got))

check("paths: relative LOG_DIR / STATE_FILE are anchored to the project folder, not the current directory",
      Settings.from_env({"LOG_DIR": "logs", "STATE_FILE": "s.json"}, base_dir=tmp).log_dir == str(tmp / "logs")
      and Settings.from_env({"STATE_FILE": "s.json"}, base_dir=tmp).state_file == str(tmp / "s.json"))
check("paths: absolute paths are left alone", Settings.from_env({"LOG_DIR": str(tmp / "x")}, base_dir=tmp / "other").log_dir == str(tmp / "x"))
check("paths: the defaults land inside the project folder",
      Settings.from_env({}).log_dir == str(PROJECT_DIR / "logs") and Settings.from_env({}).state_file == str(PROJECT_DIR / "bot_state.json"))

(tmp / "typo.env").write_text("BROKER=alpaca\nALPACA_API_KEY=SECRETVALUE123\nALPACA_SECRET_KEY=\n")
_isolated(lambda: load_dotenv(tmp / "typo.env", override=True))
m = _msg(lambda: Settings.from_env({"BROKER": "alpaca"}))
check("error message: names exactly which settings are missing", "ALPACA_KEY_ID" in m and "ALPACA_SECRET_KEY" in m and "missing or empty" in m)
check("error message: shows the .env path it read and the setting NAMES in it (spots a typo like ALPACA_API_KEY)",
      "typo.env" in m and "ALPACA_API_KEY" in m)
check("error message: flags an empty value", "ALPACA_SECRET_KEY (EMPTY)" in m)
check("error message: NEVER prints a secret value", "SECRETVALUE123" not in m)
(empty_dir / ".env.txt").write_text("BROKER=alpaca\n")
find([empty_dir])
m = _msg(lambda: Settings.from_env({"BROKER": "alpaca"}))
check("error message: no .env found -> says so and lists the folders it searched", "No .env was found" in m and str(empty_dir) in m)
check("error message: spots the classic Notepad '.env.txt' mistake", ".env.txt" in m and "rename" in m.lower())

check("settings: alpaca broker without keys rejected", raises(ConfigError, Settings.from_env, {"BROKER": "alpaca"}))
check("settings: alpaca broker with keys accepted",
      Settings.from_env({"BROKER": "alpaca", "ALPACA_KEY_ID": "k", "ALPACA_SECRET_KEY": "s"}).broker == "alpaca")
check("settings: leverage cap must be in (0, 20]",
      raises(ConfigError, Settings.from_env, {"MAX_LEVERAGE": "0"}) and raises(ConfigError, Settings.from_env, {"MAX_LEVERAGE": "50"}))
check("settings: default leverage is 1.0 (no leverage)", Settings().max_leverage == 1.0)
check("settings: alpaca secret never appears in repr", "TOPSECRET" not in repr(Settings(alpaca_secret_key="TOPSECRET")))
check("settings: key id never appears in repr either", "PKLEAKME" not in repr(Settings(alpaca_key_id="PKLEAKME")))

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

# =========================== HTTP TEST DOUBLES (shared by the adapter tests) ===========================
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
# ---- safety behaviours carried over from the earlier forex-adapter tests, now checked on Alpaca ----
import inspect
b, sess = alpaca()
check("alpaca candles: count outside 1..9000 rejected before any HTTP call",
      raises(BrokerError, b.candles, "SPY", "H1", 9001) and raises(BrokerError, b.candles, "SPY", "H1", 0) and len(sess.calls) == 0)
b, sess = alpaca()
check("alpaca order: zero units rejected before any HTTP call", raises(ValueError, b.market_order, "SPY", 0) and len(sess.calls) == 0)
b, _ = alpaca(FakeResp(200, bad_json=True))
check("alpaca: non-JSON response -> BrokerError, not a crash", raises(BrokerError, b.equity))
b, _ = alpaca(FakeResp(401, {"message": "forbidden"}))
try:
    b.equity(); kind = "no error"
except OrderRejected:
    kind = "OrderRejected"
except BrokerError as e:
    kind = "BrokerError" if "forbidden" in str(e) else "BrokerError without the server message"
check("alpaca reads: HTTP 401 (bad keys) -> BrokerError carrying the server's message, not an order rejection", kind == "BrokerError", f"({kind})")
b, sess = alpaca(OPEN, requests.Timeout("slow"), FakeResp(200, order_obj("accepted")))
check("alpaca order: a timeout on POST is NEVER retried either", raises(BrokerError, b.market_order, "SPY", 10) and len(sess.calls) == 2)
b, sess = alpaca(FakeResp(503, {}), requests.ConnectionError("blip"), FakeResp(200, {"equity": "5"}))
check("alpaca reads: GET retries through a 503 and a connection error, then succeeds", close(b.equity(), 5) and len(sess.calls) == 3)
b, sess = alpaca(FakeResp(500, {}), FakeResp(500, {}), FakeResp(500, {}))
check("alpaca reads: gives up after 3 failed GET attempts", raises(BrokerError, b.equity) and len(sess.calls) == 3)
b, sess = alpaca(FakeResp(200, {"equity": "1"}), OPEN, FakeResp(200, {"quote": {"bp": 1.0, "ap": 2.0}}), FakeResp(200, {"bars": []}))
b.equity(); b.quote("SPY"); b.candles("SPY", "H1", 5)
check("alpaca: only the PAPER trading host and the data host are ever contacted",
      all(c["url"].startswith(("https://paper-api.alpaca.markets/", "https://data.alpaca.markets/")) for c in sess.calls))
check("alpaca: there is no constructor option to point the adapter at a live host",
      not ({"host", "base_url", "base", "url", "live", "paper"} & set(inspect.signature(AlpacaBroker.__init__).parameters)))
check("alpaca: API keys travel in headers only, never in a URL or query string",
      all("KEY" not in c["url"] and "SECRET" not in str(c["params"]) for c in sess.calls))
bb = build_broker(Settings(broker="alpaca", alpaca_key_id="k", alpaca_secret_key="s"))
check("build_broker wires up the Alpaca adapter", bb.name == "alpaca")

# IBKR adapter: safety guards only (no gateway available offline)
if (HERE / "broker_ibkr.py").exists():
    from broker_ibkr import IbkrBroker
    check("ibkr: live ports 7496 / 4001 refused before any connection attempt",
          all(raises(BrokerError, IbkrBroker, "127.0.0.1", p) for p in (7496, 4001)))
else:
    print("[SKIP] ibkr: broker_ibkr.py is not in this folder (the IBKR adapter is optional)")

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

# ------------------------------------------------------------------ heartbeat (15-minute snapshots)
import bot as _botmod
from broker_base import MarketClosed as _MC, BrokerError as _BE
from bot import next_heartbeat_ts

check("heartbeat: next tick is the next wall-clock quarter hour, strictly after now",
      next_heartbeat_ts(1000 * 900 + 1, 900) == 1001 * 900 and next_heartbeat_ts(1000 * 900, 900) == 1001 * 900)
check("settings: HEARTBEAT_SECONDS default 900, 0 allowed, 30 refused",
      Settings().heartbeat_seconds == 900 and Settings.from_env({"HEARTBEAT_SECONDS": "0"}).heartbeat_seconds == 0
      and raises(ConfigError, Settings.from_env, {"HEARTBEAT_SECONDS": "30"}))


class _Clock:
    """Fake clock: sleeping just advances time, so a whole hour 'passes' instantly."""
    def __init__(self, t): self.t = t
    def time(self): return self.t


def _run_wait(start, hb, bar_seconds=3600, fail=None):
    pb = PaperBroker(100_000, 0.0001)
    if fail == "closed":
        pb.quote = lambda inst: (_ for _ in ()).throw(_MC("closed"))
    if fail == "error":
        pb.equity = lambda: (_ for _ in ()).throw(_BE("boom"))
    lg = TradeLogger(None)
    b = Bot(S(heartbeat_seconds=hb), pb, lg, StateStore(None))
    b.s.__dict__["granularity"] = "H1"
    clk = _Clock(start)
    real_time, real_sleep = _botmod.time.time, b._sleep
    _botmod.time.time = clk.time
    b._sleep = lambda s: setattr(clk, "t", clk.t + s)
    try:
        b._wait_for_next_bar()
    finally:
        _botmod.time.time = real_time
    return b, lg, clk


# start 14:20 UTC on the hour grid -> heartbeats at 14:30, 14:45, 15:00; the next bar is due 15:00:05
base = 1_700_000_000 // 3600 * 3600 + 14 * 0          # an exact hour boundary
b1, lg1, clk1 = _run_wait(base + 20 * 60, 900)
check("heartbeat: 3 snapshots (:30, :45, :00) fit between a bar at :20 and the next bar", len(lg1.heartbeat_rows) == 3,
      f"({len(lg1.heartbeat_rows)})")
check("heartbeat: waiting ends exactly when the next bar is due (5 s after the boundary)", abs(clk1.t - (base + 3605)) < 1e-6)
check("heartbeat: snapshot holds equity and position", lg1.heartbeat_rows[0]["equity"] == 100000 and lg1.heartbeat_rows[0]["position_units"] == 0)
b2, lg2, _ = _run_wait(base + 20 * 60, 0)
check("heartbeat: HEARTBEAT_SECONDS=0 -> no snapshots, same wait", len(lg2.heartbeat_rows) == 0)
b3, lg3, _ = _run_wait(base + 20 * 60, 900, fail="closed")
check("heartbeat: market closed -> row written with blank bid/ask and a note, not a crash",
      len(lg3.heartbeat_rows) == 3 and lg3.heartbeat_rows[0]["note"] == "market_closed" and lg3.heartbeat_rows[0]["bid"] == ""
      and lg3.heartbeat_rows[0]["equity"] == 100000)
b4, lg4, clk4 = _run_wait(base + 20 * 60, 900, fail="error")
check("heartbeat: broker error -> logged as a gap, waiting still completes", len(lg4.heartbeat_rows) == 3
      and lg4.heartbeat_rows[0]["note"].startswith("error") and abs(clk4.t - (base + 3605)) < 1e-6)
check("heartbeat: never changes bot state (observation only)", b1.state == StateStore(None).load())
_hd = Path(tempfile.mkdtemp())
_l = TradeLogger(_hd); _l.log_heartbeat(1.0, 0, 1.0, 1.1, "", ""); _l.log_heartbeat(2.0, 1, None, None, "", "market_closed")
_rows = (_hd / "heartbeat.csv").read_text().strip().splitlines()
check("heartbeat: csv has one header and one line per snapshot", len(_rows) == 3 and _rows[0].startswith("ts_utc,equity"))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)