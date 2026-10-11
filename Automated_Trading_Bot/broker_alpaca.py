import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from Broker_base import Broker, BrokerError, Fill, MarketClosed, OrderRejected, CANDLE_COLUMNS, empty_candles
from Config import GRANULARITY_SECONDS
from Risk import price_decimals

TRADING_HOST = "https://paper-api.alpaca.markets"
DATA_HOST = "https://data.alpaca.markets"
TIMEFRAME = {"M1": "1Min", "M5": "5Min", "M15": "15Min", "M30": "30Min", "H1": "1Hour"}
BARS_PER_DAY = {"M1": 390, "M5": 78, "M15": 26, "M30": 13, "H1": 7}       # regular session, rounded up
BAD_ORDER_STATES = {"canceled", "expired", "rejected", "suspended", "done_for_day", "stopped"}


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, key_id, secret_key, feed="iex", session=None, timeout=10, sleep=time.sleep,
                 now_fn=None, max_polls=40):
        if not key_id or not secret_key:
            raise BrokerError("Alpaca key id and secret key are required")
        self.feed = feed
        self.timeout = timeout
        self._sleep = sleep
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.max_polls = max_polls
        self.session = session or requests.Session()
        self.headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key,
                        "Accept": "application/json"}

    # ------------------------------------------------------------------ HTTP
    def _request(self, method, host, path, params=None, body=None):
        attempts = 3 if method == "GET" else 1                       # never auto-retry order writes
        last = None
        for i in range(attempts):
            try:
                r = self.session.request(method, host + path, headers=self.headers, params=params,
                                         json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = BrokerError(f"{method} {path} failed: {e}")
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = BrokerError(f"{method} {path} -> HTTP {r.status_code}")
                else:
                    return r
            if i < attempts - 1:
                self._sleep(1.5 ** i)
        raise last

    @staticmethod
    def _json(r, path, write=False):
        try:
            data = r.json() if r.status_code != 204 else {}
        except ValueError:
            raise BrokerError(f"{path}: response was not JSON (HTTP {r.status_code})")
        if r.status_code >= 400:
            msg = data.get("message", str(data)[:200]) if isinstance(data, dict) else str(data)[:200]
            raise (OrderRejected if write else BrokerError)(f"{path} -> HTTP {r.status_code}: {msg}")
        return data

    def _get(self, path, params=None, host=TRADING_HOST):
        return self._json(self._request("GET", host, path, params=params), path)

    # ------------------------------------------------------------------ reads
    def is_open(self):
        return bool(self._get("/v2/clock")["is_open"])

    def equity(self):
        return float(self._get("/v2/account")["equity"])

    def _position_detail(self, instrument):
        """(net signed qty, average entry price). A 404 means 'no position', not an error."""
        path = f"/v2/positions/{instrument}"
        r = self._request("GET", TRADING_HOST, path)
        if r.status_code == 404:
            return 0, None
        p = self._json(r, path)
        qty = int(float(p["qty"]))
        if p.get("side") == "short":
            qty = -abs(qty)
        return qty, float(p.get("avg_entry_price", 0) or 0)

    def position(self, instrument):
        return self._position_detail(instrument)[0]

    def quote(self, instrument):
        if not self.is_open():
            raise MarketClosed(f"{instrument}: US market is closed")
        path = f"/v2/stocks/{instrument}/quotes/latest"
        q = self._get(path, params={"feed": self.feed}, host=DATA_HOST).get("quote") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        if bid <= 0 or ask <= 0:
            raise BrokerError(f"{instrument}: no valid bid/ask right now (thin IEX quote?)")
        return bid, ask

    def candles(self, instrument, granularity, count):
        if not 1 <= count <= 9000:
            raise BrokerError("count must be between 1 and 9000")
        bar_s = GRANULARITY_SECONDS[granularity]
        now = self._now()
        days = math.ceil(count / BARS_PER_DAY[granularity] * 1.4 * 1.3) + 5
        path = f"/v2/stocks/{instrument}/bars"
        data = self._get(path, host=DATA_HOST, params={
            "timeframe": TIMEFRAME[granularity], "start": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": min(count + 2, 10000), "adjustment": "raw", "feed": self.feed, "sort": "desc"})
        rows = []
        for b in data.get("bars") or []:
            t = pd.Timestamp(b["t"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            if t.to_pydatetime() + timedelta(seconds=bar_s) > now:      # bar still forming: never act on it
                continue
            rows.append((t, float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]), float(b.get("v", 0))))
        if not rows:
            return empty_candles()
        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS).sort_values("time").reset_index(drop=True)
        return df.tail(count).reset_index(drop=True)

    # ------------------------------------------------------------------ writes
    def _await_fill(self, order_id, instrument):
        """Poll until the order fills. Returns the order dict. Raises on rejection / timeout."""
        path = f"/v2/orders/{order_id}"
        for _ in range(self.max_polls):
            o = self._get(path)
            status = o.get("status", "")
            if status == "filled":
                return o
            if status in BAD_ORDER_STATES:
                raise OrderRejected(f"order {status}: {o.get('failed_at') or o.get('reject_reason') or ''}".strip())
            self._sleep(0.5)
        try:                                                           # ambiguous: ask for cancellation, then bail
            self._request("DELETE", TRADING_HOST, path)
        except BrokerError:
            pass
        raise BrokerError(f"order {order_id} not filled in time; cancel requested - re-check the position")

    @staticmethod
    def _fill(o, instrument, sign, pl=0.0):
        t = pd.Timestamp(o["filled_at"]).to_pydatetime() if o.get("filled_at") else datetime.now(timezone.utc)
        return Fill(instrument, sign * int(float(o["filled_qty"])), float(o["filled_avg_price"]), t,
                    str(o["id"]), pl, "alpaca")

    def market_order(self, instrument, units, stop_loss=None, take_profit=None, tag=""):
        units = int(units)
        if units == 0:
            raise ValueError("units must be non-zero")
        if not self.is_open():
            raise MarketClosed(f"{instrument}: US market is closed - order not sent")
        dec = price_decimals(instrument)
        order = {"symbol": instrument, "qty": str(abs(units)), "side": "buy" if units > 0 else "sell",
                 "type": "market", "time_in_force": "gtc"}
        if stop_loss is not None and take_profit is not None:
            order["order_class"] = "bracket"
        elif stop_loss is not None or take_profit is not None:
            order["order_class"] = "oto"
        if take_profit is not None:
            order["take_profit"] = {"limit_price": f"{take_profit:.{dec}f}"}
        if stop_loss is not None:
            order["stop_loss"] = {"stop_price": f"{stop_loss:.{dec}f}"}
        if tag:
            order["client_order_id"] = tag[:128]
        path = "/v2/orders"
        placed = self._json(self._request("POST", TRADING_HOST, path, body=order), path, write=True)
        done = self._await_fill(placed["id"], instrument)
        return self._fill(done, instrument, 1 if units > 0 else -1)

    def close_position(self, instrument):
        net, entry = self._position_detail(instrument)
        if net == 0:
            return None
        if not self.is_open():
            raise MarketClosed(f"{instrument}: US market is closed - cannot close")
        # shares tied up in a bracket's stop/target can't be sold separately: cancel those first
        for o in self._get("/v2/orders", params={"status": "open", "symbols": instrument, "limit": 100}):
            self._request("DELETE", TRADING_HOST, f"/v2/orders/{o['id']}")
        self._sleep(0.5)
        path = f"/v2/positions/{instrument}"
        placed = self._json(self._request("DELETE", TRADING_HOST, path), path, write=True)
        done = self._await_fill(placed["id"], instrument)
        exit_px, q = float(done["filled_avg_price"]), abs(net)
        pl = q * (exit_px - entry) if net > 0 else q * (entry - exit_px)
        return self._fill(done, instrument, -1 if net > 0 else 1, pl)