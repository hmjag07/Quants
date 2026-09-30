
import math
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price(S, K, r, sigma, T, option="call"):
    if option == "call":
        return bs_call(S, K, r, sigma, T)
    if option == "put":
        return bs_put(S, K, r, sigma, T)
    raise ValueError("option must be 'call' or 'put'")


def norm_cdf_manual(x):
    """Standard normal CDF via the error function (used to cross-check scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_manual(S, K, r, sigma, T):
    """Same as bs_call but with NO scipy - proves you understand every piece."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf_manual(d1) - K * math.exp(-r * T) * norm_cdf_manual(d2)


if __name__ == "__main__":
    print("Call:", round(float(bs_call(100, 100, 0.05, 0.2, 1)), 4))
    print("Put :", round(float(bs_put(100, 100, 0.05, 0.2, 1)), 4))
