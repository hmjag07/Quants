import time
from datetime import datetime, timezone

import pandas as pd
import requests

from Broker_base import Broker, BrokerError, Fill, OrderRejected, CANDLE_COLUMNS, empty_candles
from Risk import price_decimals

PRACTICE_HOST = "https://api-fxpractice.oanda.com"


class OandaBroker(Broker):
    name = "oanda"

    def __init__(self, token, account_id, env="practice", session=None, timeout=10, sleep=time.sleep):
        if env != "practice":
            raise BrokerError("only OANDA practice accounts are allowed in this project")
        if not token or not account_id:
            raise BrokerError("OANDA token and account id are required")
        self.account_id = account_id
        self.base = PRACTICE_HOST
        self.timeout = timeout
        self._sleep = sleep
        self.session = session or requests.Session()
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                        "Accept-Datetime-Format": "RFC3339"}

    # ------------------------------------------------------------------ HTTP
    def _request(self, method, path, params=None, body=None):
        url = self.base + path
        attempts = 3 if method == "GET" else 1                       # never auto-retry order writes
        last_err = None
        for i in range(attempts):
            try:
                r = self.session.request(method, url, headers=self.headers, params=params,
                                         json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = BrokerError(f"{method} {path} failed: {e}")
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = BrokerError(f"{method} {path} -> HTTP {r.status_code}")
                else:
                    return r
            if i < attempts - 1:
                self._sleep(1.5 ** i)
        raise last_err

    def _json(self, r, path):
        try:
            data = r.json()
        except ValueError:
            raise BrokerError(f"{path}: response was not JSON (HTTP {r.status_code})")
        if r.status_code >= 400:
            msg = data.get("errorMessage") or data.get("rejectReason") or str(data)[:200]
            err = OrderRejected if "orders" in path or "close" in path else BrokerError
            raise err(f"{path} -> HTTP {r.status_code}: {msg}")
        return data

    # ------------------------------------------------------------------ reads
    def equity(self):
        path = f"/v3/accounts/{self.account_id}/summary"
        acc = self._json(self._request("GET", path), path)["account"]
        return float(acc["NAV"])

    def position(self, instrument):
        path = f"/v3/accounts/{self.account_id}/positions/{instrument}"
        pos = self._json(self._request("GET", path), path)["position"]
        return int(float(pos["long"]["units"])) + int(float(pos["short"]["units"]))   # short units are negative

    def quote(self, instrument):
        path = f"/v3/accounts/{self.account_id}/pricing"
        data = self._json(self._request("GET", path, params={"instruments": instrument}), path)
        try:
            p = data["prices"][0]
            if not p.get("tradeable", True):
                raise BrokerError(f"{instrument} is not tradeable right now (market closed?)")
            return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])
        except (KeyError, IndexError):
            raise BrokerError(f"unexpected pricing response: {str(data)[:200]}")

    def candles(self, instrument, granularity, count):
        if not 1 <= count <= 5000:
            raise BrokerError("OANDA returns at most 5000 candles per request")
        path = f"/v3/instruments/{instrument}/candles"
        data = self._json(self._request("GET", path, params={"granularity": granularity,
                                                              "count": count, "price": "M"}), path)
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", False):
                continue                                              # never act on a bar still forming
            m = c["mid"]
            rows.append((pd.Timestamp(c["time"]), float(m["o"]), float(m["h"]), float(m["l"]),
                         float(m["c"]), float(c.get("volume", 0))))
        if not rows:
            return empty_candles()
        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.sort_values("time").reset_index(drop=True)

    # ------------------------------------------------------------------ writes
    @staticmethod
    def _fill_from(tx, instrument, name="oanda"):
        t = pd.Timestamp(tx["time"]).to_pydatetime()
        return Fill(instrument, int(float(tx["units"])), float(tx["price"]), t,
                    str(tx.get("orderID") or tx.get("id")), float(tx.get("pl", 0.0)), name)

    def market_order(self, instrument, units, stop_loss=None, take_profit=None, tag=""):
        units = int(units)
        if units == 0:
            raise ValueError("units must be non-zero")
        dec = price_decimals(instrument)
        order = {"type": "MARKET", "instrument": instrument, "units": str(units),
                 "timeInForce": "FOK", "positionFill": "DEFAULT",
                 "clientExtensions": {"tag": "sma-bot", "comment": (tag or "bot")[:100]}}
        if stop_loss is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss:.{dec}f}", "timeInForce": "GTC"}
        if take_profit is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.{dec}f}", "timeInForce": "GTC"}
        path = f"/v3/accounts/{self.account_id}/orders"
        data = self._json(self._request("POST", path, body={"order": order}), path)
        if "orderFillTransaction" in data:
            return self._fill_from(data["orderFillTransaction"], instrument)
        if "orderCancelTransaction" in data:
            raise OrderRejected(f"order cancelled: {data['orderCancelTransaction'].get('reason', 'unknown')}")
        raise BrokerError(f"unexpected order response: {str(data)[:200]}")

    def close_position(self, instrument):
        net = self.position(instrument)
        if net == 0:
            return None
        body = {"longUnits": "ALL"} if net > 0 else {"shortUnits": "ALL"}
        path = f"/v3/accounts/{self.account_id}/positions/{instrument}/close"
        data = self._json(self._request("PUT", path, body=body), path)
        tx = data.get("longOrderFillTransaction") or data.get("shortOrderFillTransaction")
        if not tx:
            raise BrokerError(f"unexpected close response: {str(data)[:200]}")
        return self._fill_from(tx, instrument)