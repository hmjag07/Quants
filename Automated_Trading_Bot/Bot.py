import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from Broker_base import Broker, BrokerError, OrderRejected
from Config import ConfigError, Settings, load_dotenv
from Risk import position_size, stop_take_levels, daily_loss_breached, price_decimals
from Strategy import StrategyParams, latest_signal
from Trade_logger import TradeLogger

log = logging.getLogger("bot")


def seconds_until_next_bar(now_ts, bar_seconds, delay=5):
    """Seconds from `now_ts` (unix) until `delay` seconds after the next bar boundary."""
    next_boundary = (int(now_ts) // bar_seconds + 1) * bar_seconds
    return max(0.0, next_boundary + delay - now_ts)


@dataclass
class BotState:
    last_bar_time: str = ""        # ISO time of the last bar we acted on
    last_position: int = 0         # sign of the position at the end of the last cycle
    blocked_dir: int = 0           # direction we may not re-enter after a stop/target exit
    day: str = ""                  # UTC date the day_start_equity belongs to
    day_start_equity: float = 0.0
    halted_day: str = ""           # UTC date on which the daily-loss halt is active
    consecutive_errors: int = 0


class StateStore:
    """JSON state file with atomic writes. path=None -> in memory only."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def load(self):
        if self.path and self.path.exists():
            try:
                return BotState(**json.loads(self.path.read_text()))
            except (ValueError, TypeError):
                log.error("state file unreadable, starting from a clean state: %s", self.path)
        return BotState()

    def save(self, state):
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2))
        os.replace(tmp, self.path)                      # atomic: never a half-written file


def _sign(x):
    return (x > 0) - (x < 0)


class Bot:
    def __init__(self, settings: Settings, broker: Broker, logger: TradeLogger, store: StateStore = None):
        self.s = settings
        self.broker = broker
        self.tlog = logger
        self.store = store or StateStore(settings.state_file)
        self.state = self.store.load()
        self.params = StrategyParams(settings.fast, settings.slow, settings.mom_lookback, settings.atr_period)
        self.decimals = price_decimals(settings.instrument)
        self._stop = False

    # ------------------------------------------------------------------ one cycle
    def step(self):
        """Process the newest completed bar. Returns a short status string (useful in tests/logs)."""
        s, st = self.s, self.state
        df = self.broker.candles(s.instrument, s.granularity, s.lookback_bars)
        min_bars = max(s.slow, s.mom_lookback + 1, s.atr_period)
        if len(df) < min_bars:
            log.info("only %d candles available (need %d) - waiting for more data", len(df), min_bars)
            return "insufficient_data"

        bar_time = pd.Timestamp(df["time"].iloc[-1])
        bar_iso = bar_time.isoformat()
        if bar_iso == st.last_bar_time:
            return "no_new_bar"

        equity = self.broker.equity()
        pos = int(self.broker.position(s.instrument))
        self._roll_day(bar_time, equity)

        # (2) did a protective stop/target close the position since the last cycle?
        if st.last_position != 0 and pos == 0:
            st.blocked_dir = st.last_position
            self.tlog.log_trade(bar_time=bar_iso, event="EXIT", instrument=s.instrument,
                                side=_sign(-st.last_position), equity=round(equity, 2),
                                reason="stop_or_take_profit", broker=self.broker.name)
            log.info("position closed by stop/target; blocking re-entry in direction %+d", st.blocked_dir)

        # (3) strategy
        sig = latest_signal(df, self.params)
        if st.blocked_dir != 0 and sig.target != st.blocked_dir:
            st.blocked_dir = 0                                   # signal changed: unblock
        target = sig.target

        # (4) safety rules
        today = bar_time.strftime("%Y-%m-%d")
        if daily_loss_breached(st.day_start_equity, equity, s.max_daily_loss) and st.halted_day != today:
            st.halted_day = today
            self.tlog.log_trade(bar_time=bar_iso, event="HALT", instrument=s.instrument, equity=round(equity, 2),
                                reason=f"daily loss limit {s.max_daily_loss:.1%} breached", broker=self.broker.name)
            log.error("DAILY LOSS LIMIT breached - flattening and halting until next UTC day")
        if st.halted_day == today:
            target = 0
        elif target == st.blocked_dir and target != 0:
            target = 0

        # (5) reconcile
        cur = _sign(pos)
        if target != cur:
            if cur != 0:
                self._exit(bar_iso, "signal_change" if st.halted_day != today else "daily_loss_halt")
            if target != 0:
                self._enter(bar_iso, target, sig)

        # (6) bookkeeping
        pos_after = int(self.broker.position(s.instrument))
        st.last_position = _sign(pos_after)
        st.last_bar_time = bar_iso
        st.consecutive_errors = 0
        self.tlog.log_equity(bar_iso, self.broker.equity(), pos_after, sig.close)
        self.store.save(st)
        log.info("bar %s | close %.5f | target %+d | position %+d | equity %.2f",
                 bar_iso, sig.close, target, pos_after, self.broker.equity())
        return "processed"

    # ------------------------------------------------------------------ actions
    def _roll_day(self, bar_time, equity):
        day = bar_time.strftime("%Y-%m-%d")
        if self.state.day != day:
            self.state.day, self.state.day_start_equity = day, float(equity)

    def _exit(self, bar_iso, reason):
        fill = self.broker.close_position(self.s.instrument)
        if fill is None:
            return
        self.tlog.log_trade(bar_time=bar_iso, event="EXIT", instrument=self.s.instrument,
                            side=_sign(fill.units), units=abs(fill.units), price=fill.price,
                            realized_pl=round(fill.pl, 2), equity=round(self.broker.equity(), 2),
                            reason=reason, order_id=fill.order_id, broker=self.broker.name)

    def _enter(self, bar_iso, side, sig):
        s = self.s
        bid, ask = self.broker.quote(s.instrument)
        ref = ask if side > 0 else bid
        units = position_size(self.broker.equity(), s.risk_pct, sig.atr, s.atr_stop_mult, s.max_units)
        if units == 0:
            self.tlog.log_trade(bar_time=bar_iso, event="SKIP", instrument=s.instrument, side=side,
                                reason="position size is zero", broker=self.broker.name)
            log.warning("entry skipped: position size is zero")
            return
        stop, take = stop_take_levels(side, ref, sig.atr, s.atr_stop_mult, s.tp_rr, self.decimals)
        tag = f"sma-{pd.Timestamp(sig.bar_time):%Y%m%d%H%M}"
        try:
            fill = self.broker.market_order(s.instrument, side * units, stop, take, tag)
        except OrderRejected as e:
            self.tlog.log_trade(bar_time=bar_iso, event="REJECTED", instrument=s.instrument, side=side,
                                units=units, stop_loss=stop, take_profit=take, reason=str(e)[:200],
                                broker=self.broker.name)
            log.error("order rejected: %s", e)
            return
        self.tlog.log_trade(bar_time=bar_iso, event="ENTRY", instrument=s.instrument, side=side,
                            units=abs(fill.units), price=fill.price, stop_loss=stop, take_profit=take,
                            equity=round(self.broker.equity(), 2), reason="signal",
                            order_id=fill.order_id, broker=self.broker.name)

    # ------------------------------------------------------------------ run loop
    def request_stop(self, *_):
        log.info("shutdown requested")
        self._stop = True

    def _sleep(self, seconds):
        end = time.time() + seconds
        while not self._stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def run_forever(self):
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        log.info("bot started | %s %s | broker=%s", self.s.instrument, self.s.granularity, self.broker.name)
        while not self._stop:
            try:
                self.step()
                self._sleep(seconds_until_next_bar(time.time(), self.s.bar_seconds))
            except (BrokerError, OSError, ValueError) as e:
                self.state.consecutive_errors += 1
                self.store.save(self.state)
                log.exception("cycle failed (%d/%d): %s", self.state.consecutive_errors,
                              self.s.max_consecutive_errors, e)
                if self.state.consecutive_errors >= self.s.max_consecutive_errors:
                    log.critical("too many consecutive errors - attempting to flatten and stopping")
                    try:
                        self._exit("", "error_halt")
                    except Exception:
                        log.exception("could not flatten - CHECK THE ACCOUNT MANUALLY")
                    return 1
                self._sleep(30)
        log.info("bot stopped cleanly")
        return 0


# ---------------------------------------------------------------------- wiring
def build_broker(s: Settings) -> Broker:
    if s.broker == "oanda":
        from broker_oanda import OandaBroker
        return OandaBroker(s.oanda_token, s.oanda_account_id, s.oanda_env)
    if s.broker == "ibkr":
        from broker_ibkr import IbkrBroker
        return IbkrBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id)
    raise SystemExit("BROKER=paper has no live data feed. Use backtest.py for the paper broker, "
                     "or set BROKER=oanda / ibkr.")


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=2_000_000, backupCount=5)):
        h.setFormatter(fmt)
        root.addHandler(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paper-trading SMA/momentum bot")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = ap.parse_args(argv)
    load_dotenv()
    try:
        s = Settings.from_env()
    except ConfigError as e:
        raise SystemExit(f"configuration error: {e}")
    setup_logging(s.log_dir)
    bot = Bot(s, build_broker(s), TradeLogger(s.log_dir))
    if args.once:
        print(bot.step())
        return 0
    return bot.run_forever()


if __name__ == "__main__":
    sys.exit(main())