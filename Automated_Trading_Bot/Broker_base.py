
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

CANDLE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class BrokerError(Exception):
    """Anything went wrong talking to the broker (network, auth, bad response)."""


class OrderRejected(BrokerError):
    """The broker received the order and declined it (margin, market closed, FOK not met...)."""


@dataclass(frozen=True)
class Fill:
    instrument: str
    units: int            # signed: + bought, - sold
    price: float
    time: datetime        # UTC
    order_id: str
    pl: float = 0.0       # realised P&L from this fill (non-zero when it closes/reduces a position)
    broker: str = ""


def empty_candles():
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDLE_COLUMNS}).astype(
        {"time": "datetime64[ns, UTC]"})


class Broker(ABC):
    name = "base"

    @abstractmethod
    def equity(self) -> float:
        """Net liquidation value of the account, in account currency."""

    @abstractmethod
    def position(self, instrument) -> int:
        """Net signed units currently held (0 = flat)."""

    @abstractmethod
    def quote(self, instrument) -> tuple:
        """(bid, ask)."""

    @abstractmethod
    def candles(self, instrument, granularity, count) -> pd.DataFrame:
        """
        The most recent `count` COMPLETE candles, oldest first, with columns
        time (UTC, tz-aware), open, high, low, close, volume. Never includes a bar still forming.
        """

    @abstractmethod
    def market_order(self, instrument, units, stop_loss=None, take_profit=None, tag="") -> Fill:
        """Send a market order for signed `units`, optionally with protective stop / target prices."""

    @abstractmethod
    def close_position(self, instrument):
        """Flatten the position. Returns a Fill, or None if already flat."""