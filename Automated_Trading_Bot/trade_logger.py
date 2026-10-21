import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TRADE_FIELDS = ["ts_utc", "bar_time", "event", "instrument", "side", "units", "price", "stop_loss",
                "take_profit", "realized_pl", "equity", "reason", "order_id", "broker"]
EQUITY_FIELDS = ["ts_utc", "bar_time", "equity", "position_units", "close"]
HEARTBEAT_FIELDS = ["ts_utc", "equity", "position_units", "bid", "ask", "last_bar_time", "note"]


class TradeLogger:
    def __init__(self, log_dir="logs"):
        self.dir = Path(log_dir) if log_dir else None
        self.trades, self.equity_rows, self.heartbeat_rows = [], [], []
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _append(self, name, fields, row):
        path = self.dir / name
        new = (not path.exists()) or path.stat().st_size == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def log_trade(self, **kw):
        row = {k: "" for k in TRADE_FIELDS}
        row.update({k: v for k, v in kw.items() if k in TRADE_FIELDS})
        row["ts_utc"] = row["ts_utc"] or self._now()
        self.trades.append(row)
        if self.dir:
            self._append("trades.csv", TRADE_FIELDS, row)
        return row

    def log_equity(self, bar_time, equity, position_units, close):
        row = {"ts_utc": self._now(), "bar_time": bar_time, "equity": round(float(equity), 4),
               "position_units": int(position_units), "close": close}
        self.equity_rows.append(row)
        if self.dir:
            self._append("equity_curve.csv", EQUITY_FIELDS, row)
        return row

    def log_heartbeat(self, equity, position_units, bid, ask, last_bar_time, note=""):
        """Read-only snapshot between bars. Blank cells mean 'not available' (see `note`)."""
        def num(x, nd):
            return "" if x is None else round(float(x), nd)
        row = {"ts_utc": self._now(), "equity": num(equity, 4),
               "position_units": "" if position_units is None else int(position_units),
               "bid": num(bid, 6), "ask": num(ask, 6), "last_bar_time": last_bar_time, "note": note}
        self.heartbeat_rows.append(row)
        if self.dir:
            self._append("heartbeat.csv", HEARTBEAT_FIELDS, row)
        return row

    def trades_frame(self):
        return pd.DataFrame(self.trades, columns=TRADE_FIELDS)

    def equity_frame(self):
        return pd.DataFrame(self.equity_rows, columns=EQUITY_FIELDS)