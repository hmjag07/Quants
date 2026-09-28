"""
Week 1 - Day 5/6: Discounting, bond pricing, forward pricing.
All rates are annual, continuously compounded unless the function name says otherwise.
"""
import numpy as np


def discount_factor_cc(r, T):
    """Continuously compounded discount factor e^{-rT}."""
    return np.exp(-r * T)


def discount_factor_annual(r, T):
    """Annually compounded discount factor (1+r)^{-T}."""
    return (1.0 + r) ** (-T)


def zero_coupon_price(face, r, T, compounding="continuous"):
    """Price of a zero-coupon bond paying `face` at time T."""
    df = discount_factor_cc(r, T) if compounding == "continuous" else discount_factor_annual(r, T)
    return face * df


def coupon_bond_price(face, coupon_rate, ytm, years, freq=1):
    """
    Price of a plain-vanilla coupon bond.
    ytm is the yield, compounded `freq` times per year.
    """
    n = (round(years * freq))
    c = face * coupon_rate / freq
    t = np.arange(1, n + 1)
    disc = (1.0 + ytm / freq) ** (-t)
    return float(np.sum(c * disc) + face * disc[-1])


def forward_price(S0, r, T, income_yield=0.0):
    """Model-independent forward price F = S0 * e^{(r - q)T}."""
    return S0 * np.exp((r - income_yield) * T)


def forward_value(S0, K, r, T, income_yield=0.0):
    """Value today of a long forward with delivery price K: S0 e^{-qT} - K e^{-rT}."""
    return S0 * np.exp(-income_yield * T) - K * np.exp(-r * T)


def put_call_parity_gap(call, put, S0, K, r, T):
    """Should be ~0 for European options on a non-dividend stock: C - P - (S0 - K e^{-rT})."""
    return call - put - (S0 - K * np.exp(-r * T))


if __name__ == "__main__":
    print("ZCB 100 @ 5% cc, 2y      :", round(zero_coupon_price(100, 0.05, 2), 4))
    print("Coupon bond 5%, 4% ytm 3y:", round(coupon_bond_price(100, 0.05, 0.04, 3), 4))
    print("Forward S=100,r=5%,T=1   :", round(forward_price(100, 0.05, 1), 4))
