import math

def price_decimals(instrument):
    return 3 if "JPY" in instrument.upper() else 5


def position_size(equity, risk_pct, atr, atr_stop_mult, max_units):
    """Whole units to trade (>= 0). Returns 0 if inputs are unusable."""
    if not all(map(math.isfinite, (equity, atr))) or equity <= 0 or atr <= 0:
        return 0
    stop_distance = atr * atr_stop_mult
    units = int(equity * risk_pct / stop_distance)          # floor: never round risk UP
    return max(0, min(units, int(max_units)))


def stop_take_levels(side, entry, atr, atr_stop_mult, tp_rr, decimals=5):
    """(stop_loss, take_profit) prices for side = +1 (long) or -1 (short)."""
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    stop_dist = atr * atr_stop_mult
    stop = entry - side * stop_dist
    tp = entry + side * stop_dist * tp_rr
    return round(stop, decimals), round(tp, decimals)


def daily_loss_breached(day_start_equity, equity, max_daily_loss):
    """True if equity has fallen by >= max_daily_loss (fraction) since the day started."""
    if day_start_equity is None or day_start_equity <= 0:
        return False
    return (equity / day_start_equity - 1.0) <= -max_daily_loss