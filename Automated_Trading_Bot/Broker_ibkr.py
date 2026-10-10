import math
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from Broker_base import Broker, BrokerError, Fill, OrderRejected, CANDLE_COLUMNS, empty_candles
from Config import IBKR_BAR_SIZE, IBKR_LIVE_PORTS, GRANULARITY_SECONDS


def _import_ib():
    try:
        import ib_async as ib
    except ImportError:
        try:
            import ib_insync as ib
        except ImportError:
            raise ImportError("install an IBKR library:  pip install ib_async")
    return ib


class IbkrBroker(Broker):
    name = "ibkr"

    def __init__(self, host="127.0.0.1", port=7497, client_id=1, ib=None, connect_timeout=10):
        if port in IBKR_LIVE_PORTS:
            raise BrokerError(f"port {port} is a LIVE trading port: refused (use 7497 or 4002)")
        self._lib = _import_ib() if ib is None else None
        if ib is None:
            ib = self._lib.IB()
            try:
                ib.connect(host, port, clientId=client_id, timeout=connect_timeout)
            except Exception as e:
                raise BrokerError(f"could not connect to IB at {host}:{port} - is Gateway/TWS running "
                                  f"and the API enabled? ({e})")
        self.ib = ib

    # ---- helpers
    def _contract(self, instrument):
        lib = self._lib or _import_ib()
        base, _, quote = instrument.partition("_")
        c = lib.Forex(base + quote)                                  # e.g. EUR_USD -> Forex('EURUSD')
        self.ib.qualifyContracts(c)
        return c

    # ---- reads
    def equity(self):
        for v in self.ib.accountSummary():
            if v.tag == "NetLiquidation":
                return float(v.value)
        raise BrokerError("NetLiquidation not found in account summary")

    def position(self, instrument):
        base, _, quote = instrument.partition("_")
        total = 0
        for p in self.ib.positions():
            if p.contract.symbol == base and p.contract.currency == quote:
                total += int(p.position)
        return total

    def quote(self, instrument):
        t = self.ib.reqTickers(self._contract(instrument))[0]
        if not (t.bid and t.ask) or math.isnan(t.bid) or math.isnan(t.ask):
            raise BrokerError("no bid/ask (market data subscription missing or market closed)")
        return float(t.bid), float(t.ask)

    def candles(self, instrument, granularity, count):
        bar_s = GRANULARITY_SECONDS[granularity]
        days = max(2, math.ceil(count * bar_s / 86400 * 1.6) + 2)      # generous: weekends have no bars
        bars = self.ib.reqHistoricalData(self._contract(instrument), endDateTime="",
                                         durationStr=f"{days} D", barSizeSetting=IBKR_BAR_SIZE[granularity],
                                         whatToShow="MIDPOINT", useRTH=False, formatDate=2)
        if not bars:
            return empty_candles()
        now = datetime.now(timezone.utc)
        rows = []
        for b in bars:
            t = pd.Timestamp(b.date)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            if t.to_pydatetime() + timedelta(seconds=bar_s) <= now:    # drop the bar still forming
                rows.append((t, float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
        if not rows:
            return empty_candles()
        return pd.DataFrame(rows, columns=CANDLE_COLUMNS).tail(count).reset_index(drop=True)

    # ---- writes
    def _wait_filled(self, trade, timeout=20):
        end = time.time() + timeout
        while time.time() < end and not trade.isDone():
            self.ib.sleep(0.25)
        return trade

    def market_order(self, instrument, units, stop_loss=None, take_profit=None, tag=""):
        lib = self._lib or _import_ib()
        units = int(units)
        if units == 0:
            raise ValueError("units must be non-zero")
        contract = self._contract(instrument)
        action, opp = ("BUY", "SELL") if units > 0 else ("SELL", "BUY")
        trade = self._wait_filled(self.ib.placeOrder(contract, lib.MarketOrder(action, abs(units))))
        status = trade.orderStatus.status
        if status != "Filled":
            raise OrderRejected(f"IBKR order ended with status {status}")
        price = float(trade.orderStatus.avgFillPrice)
        if stop_loss is not None or take_profit is not None:
            oca = f"oca-{uuid.uuid4().hex[:10]}"
            for order in ([lib.StopOrder(opp, abs(units), stop_loss, tif="GTC", ocaGroup=oca, ocaType=1)] if stop_loss is not None else []) + \
                         ([lib.LimitOrder(opp, abs(units), take_profit, tif="GTC", ocaGroup=oca, ocaType=1)] if take_profit is not None else []):
                self.ib.placeOrder(contract, order)
        return Fill(instrument, units, price, datetime.now(timezone.utc), str(trade.order.orderId), 0.0, self.name)

    def close_position(self, instrument):
        lib = self._lib or _import_ib()
        net = self.position(instrument)
        if net == 0:
            return None
        contract = self._contract(instrument)
        for t in self.ib.openTrades():                                   # cancel resting stop / target first
            if t.contract.symbol == contract.symbol and t.contract.currency == contract.currency:
                self.ib.cancelOrder(t.order)
        self.ib.sleep(0.5)
        action = "SELL" if net > 0 else "BUY"
        trade = self._wait_filled(self.ib.placeOrder(contract, lib.MarketOrder(action, abs(net))))
        if trade.orderStatus.status != "Filled":
            raise OrderRejected(f"IBKR close ended with status {trade.orderStatus.status}")
        return Fill(instrument, -net, float(trade.orderStatus.avgFillPrice), datetime.now(timezone.utc),
                    str(trade.order.orderId), 0.0, self.name)