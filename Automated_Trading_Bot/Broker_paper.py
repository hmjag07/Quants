from collections import deque
from datetime import datetime, timezone
from itertools import islice

import pandas as pd

from Broker_base import Broker, Fill, CANDLE_COLUMNS, empty_candles


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, start_equity=100_000.0, spread=0.0001, max_history=6000):
        self.balance = float(start_equity)
        self.spread = float(spread)
        self.units = 0
        self.avg_price = 0.0
        self.stop = None
        self.take = None
        self.mid = None
        self.now = None
        self.fills = []                      # every Fill, including stop/target exits
        self._bars = deque(maxlen=max_history)
        self._oid = 0

    # ----- market feed -----
    def on_bar(self, time, open_, high, low, close, volume=0.0):
        """Feed one COMPLETE bar. Processes stops/targets inside it, then marks the market at close."""
        if self.units != 0:
            self._check_exits(time, open_, high, low)
        self._bars.append((time, open_, high, low, close, volume))
        self.mid, self.now = float(close), time

    def _half(self):
        return self.spread / 2.0

    def _check_exits(self, time, o, h, l):
        half, long_ = self._half(), self.units > 0
        price = None
        if long_:
            if self.stop is not None and o <= self.stop:        price = o - half            # gapped through stop
            elif self.stop is not None and l <= self.stop:      price = self.stop - half
            elif self.take is not None and o >= self.take:      price = o - half
            elif self.take is not None and h >= self.take:      price = self.take - half
        else:
            if self.stop is not None and o >= self.stop:        price = o + half
            elif self.stop is not None and h >= self.stop:      price = self.stop + half
            elif self.take is not None and o <= self.take:      price = o + half
            elif self.take is not None and l <= self.take:      price = self.take + half
        if price is not None:
            self._realise_close(time, price, order_tag="stop_or_target")

    # ----- accounting -----
    def _realise_close(self, time, price, order_tag):
        q = abs(self.units)
        pl = q * (price - self.avg_price) if self.units > 0 else q * (self.avg_price - price)
        self.balance += pl
        fill = Fill("PAPER", -self.units, round(price, 6), time, self._next_id(order_tag), pl, self.name)
        self.fills.append(fill)
        self.units, self.avg_price, self.stop, self.take = 0, 0.0, None, None
        return fill

    def _next_id(self, tag):
        self._oid += 1
        return f"paper-{self._oid}-{tag}"

    # ----- Broker interface -----
    def equity(self):
        if self.units == 0 or self.mid is None:
            return self.balance
        mark = self.mid - self._half() if self.units > 0 else self.mid + self._half()
        unreal = self.units * (mark - self.avg_price)
        return self.balance + unreal

    def position(self, instrument):
        return int(self.units)

    def quote(self, instrument):
        if self.mid is None:
            raise RuntimeError("no market data yet: call on_bar() first")
        return self.mid - self._half(), self.mid + self._half()

    def candles(self, instrument, granularity, count):
        if not self._bars:
            return empty_candles()
        rows = list(islice(self._bars, max(0, len(self._bars) - count), None))
        return pd.DataFrame(rows, columns=CANDLE_COLUMNS)

    def market_order(self, instrument, units, stop_loss=None, take_profit=None, tag=""):
        units = int(units)
        if units == 0:
            raise ValueError("units must be non-zero")
        bid, ask = self.quote(instrument)
        price = ask if units > 0 else bid
        pl = 0.0
        if self.units == 0:
            self.units, self.avg_price = units, price
            self.stop, self.take = stop_loss, take_profit
        elif (self.units > 0) == (units > 0):                      # adding to the position
            total = self.units + units
            self.avg_price = (self.units * self.avg_price + units * price) / total
            self.units = total
            if stop_loss is not None: self.stop = stop_loss
            if take_profit is not None: self.take = take_profit
        else:                                                      # reducing / flipping
            closed = min(abs(units), abs(self.units))
            pl = closed * (price - self.avg_price) if self.units > 0 else closed * (self.avg_price - price)
            self.balance += pl
            remaining = self.units + units
            if remaining == 0:
                self.units, self.avg_price, self.stop, self.take = 0, 0.0, None, None
            elif (remaining > 0) == (self.units > 0):              # partially reduced, same direction
                self.units = remaining
            else:                                                  # flipped through zero
                self.units, self.avg_price = remaining, price
                self.stop, self.take = stop_loss, take_profit
        fill = Fill(instrument, units, round(price, 6), self.now, self._next_id(tag or "mkt"), pl, self.name)
        self.fills.append(fill)
        return fill

    def close_position(self, instrument):
        if self.units == 0:
            return None
        bid, ask = self.quote(instrument)
        price = bid if self.units > 0 else ask
        fill = self._realise_close(self.now, price, "close")
        return Fill(instrument, fill.units, fill.price, fill.time, fill.order_id, fill.pl, self.name)