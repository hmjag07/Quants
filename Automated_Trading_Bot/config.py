import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

GRANULARITY_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}
IBKR_BAR_SIZE = {"M1": "1 min", "M5": "5 mins", "M15": "15 mins", "M30": "30 mins", "H1": "1 hour"}
IBKR_LIVE_PORTS = (7496, 4001)          # TWS live, Gateway live  -> refused
BROKERS = ("paper", "alpaca", "oanda", "ibkr")


class ConfigError(ValueError):
    pass


def _parse_env_value(raw):
    """'abc  # note' -> 'abc';   '"a#b"  # note' -> 'a#b' (quoted values may contain '#')."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
        return raw[1:]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_dotenv(path=".env", override=False):
    """Minimal .env reader (KEY=VALUE per line, # comments, optional quotes). No extra dependency."""
    p = Path(path)
    if not p.exists():
        return {}
    loaded = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), _parse_env_value(value)
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass
class Settings:
    # --- what to trade ---
    broker: str = "paper"                 # paper | alpaca | oanda | ibkr
    instrument: str = "EUR_USD"
    granularity: str = "H1"               # M1 M5 M15 M30 H1
    # --- strategy ---
    fast: int = 20
    slow: int = 50
    mom_lookback: int = 10
    atr_period: int = 14
    # --- risk ---
    atr_stop_mult: float = 2.0            # stop-loss distance = atr_stop_mult * ATR
    tp_rr: float = 2.0                    # take-profit distance = tp_rr * stop distance
    risk_pct: float = 0.005               # fraction of equity risked per trade (0.5%)
    max_units: int = 500_000
    max_leverage: float = 1.0             # notional position value <= equity x this (1.0 = no leverage)
    max_daily_loss: float = 0.02          # halt for the UTC day after losing 2% of day-start equity
    max_consecutive_errors: int = 5
    # --- data / paths ---
    lookback_bars: int = 300
    log_dir: str = "logs"
    state_file: str = "bot_state.json"
    # --- paper broker only ---
    paper_start_equity: float = 100_000.0
    paper_spread: float = 0.00010         # 1 pip on EUR_USD, in price units
    # --- Alpaca (paper only: the paper host is hard-coded in broker_alpaca.py) ---
    alpaca_key_id: str = field(default="", repr=False)
    alpaca_secret_key: str = field(default="", repr=False)
    alpaca_feed: str = "iex"              # 'iex' is the free feed
    # --- OANDA ---
    oanda_token: str = field(default="", repr=False)
    oanda_account_id: str = ""
    oanda_env: str = "practice"           # only 'practice' is accepted
    # --- IBKR ---
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497                 # 7497 = TWS paper, 4002 = Gateway paper
    ibkr_client_id: int = 1

    @classmethod
    def from_env(cls, env=None):
        """Build Settings from environment variables named after the fields, upper-cased."""
        env = os.environ if env is None else env
        kwargs = {}
        for f in fields(cls):
            raw = env.get(f.name.upper())
            if raw is None or raw == "":
                continue
            try:
                kwargs[f.name] = f.type(raw) if f.type in (int, float, str) else raw
            except (TypeError, ValueError):
                raise ConfigError(f"{f.name.upper()}={raw!r} is not a valid {f.type.__name__}")
        s = cls(**kwargs)
        s.validate()
        return s

    def validate(self):
        if self.broker not in BROKERS:
            raise ConfigError(f"BROKER must be one of {BROKERS}, got {self.broker!r}")
        if self.granularity not in GRANULARITY_SECONDS:
            raise ConfigError(f"GRANULARITY must be one of {tuple(GRANULARITY_SECONDS)}, got {self.granularity!r}")
        if not (1 <= self.fast < self.slow):
            raise ConfigError("need 1 <= FAST < SLOW")
        if self.mom_lookback < 1 or self.atr_period < 2:
            raise ConfigError("MOM_LOOKBACK must be >= 1 and ATR_PERIOD >= 2")
        need = self.slow + max(self.mom_lookback, self.atr_period) + 5
        if self.lookback_bars < need:
            raise ConfigError(f"LOOKBACK_BARS must be >= {need} for these indicator settings")
        if not (0 < self.risk_pct <= 0.05):
            raise ConfigError("RISK_PCT must be in (0, 0.05]: more than 5% risk per trade is refused")
        if self.atr_stop_mult <= 0 or self.tp_rr <= 0:
            raise ConfigError("ATR_STOP_MULT and TP_RR must be positive")
        if self.max_units <= 0:
            raise ConfigError("MAX_UNITS must be positive")
        if not (0 < self.max_daily_loss <= 0.2):
            raise ConfigError("MAX_DAILY_LOSS must be in (0, 0.2]")
        if self.oanda_env != "practice":
            raise ConfigError("OANDA_ENV must be 'practice': this project is paper-trading only")
        if self.ibkr_port in IBKR_LIVE_PORTS:
            raise ConfigError(f"IBKR_PORT {self.ibkr_port} is a LIVE port: refused (use 7497 or 4002)")
        if not (0 < self.max_leverage <= 20):
            raise ConfigError("MAX_LEVERAGE must be in (0, 20]")
        if self.alpaca_feed not in ("iex", "sip"):
            raise ConfigError("ALPACA_FEED must be 'iex' or 'sip'")
        if self.broker == "alpaca" and not (self.alpaca_key_id and self.alpaca_secret_key):
            raise ConfigError("BROKER=alpaca needs ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        if self.broker == "oanda" and not (self.oanda_token and self.oanda_account_id):
            raise ConfigError("BROKER=oanda needs OANDA_TOKEN and OANDA_ACCOUNT_ID")

    @property
    def bar_seconds(self):
        return GRANULARITY_SECONDS[self.granularity]