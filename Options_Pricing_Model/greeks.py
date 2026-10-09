import numpy as np
from scipy.stats import norm
from black_scholes import bs_price, _d1_d2


# ---------------- 1. Closed form ----------------
def bs_greeks(S, K, r, sigma, T, option="call"):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf = norm.pdf(d1)
    disc = np.exp(-r * T)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    decay = -S * pdf * sigma / (2.0 * np.sqrt(T))
    if option == "call":
        delta = norm.cdf(d1)
        theta = decay - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    elif option == "put":
        delta = norm.cdf(d1) - 1.0
        theta = decay + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def practitioner_units(g):
    """vega per 1 vol point, theta per calendar day, rho per 1% rate move."""
    return {"delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"] / 100.0,
            "theta": g["theta"] / 365.0, "rho": g["rho"] / 100.0}


# ---------------- 2. Finite differences ----------------
def fd_greeks(S, K, r, sigma, T, option="call", pricer=bs_price):
    """Central differences around the current point. `pricer(S,K,r,sigma,T,option)`."""
    hS, hv, hr = S * 1e-3, 1e-4, 1e-5
    hT = min(1e-4, T / 10.0)
    V = lambda s=S, v=sigma, rr=r, t=T: float(pricer(s, K, rr, v, t, option))
    v0 = V()
    return {
        "delta": (V(s=S + hS) - V(s=S - hS)) / (2 * hS),
        "gamma": (V(s=S + hS) - 2 * v0 + V(s=S - hS)) / hS**2,
        "vega": (V(v=sigma + hv) - V(v=sigma - hv)) / (2 * hv),
        "theta": -(V(t=T + hT) - V(t=T - hT)) / (2 * hT),   # theta = -dV/dT (time passing)
        "rho": (V(rr=r + hr) - V(rr=r - hr)) / (2 * hr),
    }


# ---------------- 3. Monte Carlo estimators ----------------
def _mean_se(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def mc_greeks(S0, K, r, sigma, T, n_paths=500_000, option="call", seed=None):
    """
    Returns {name: (estimate, standard_error)} for price, delta, vega, rho, gamma.

    delta / vega / rho: PATHWISE estimators - differentiate the payoff along each path.
        d S_T / d S0    = S_T / S0
        d S_T / d sigma = S_T (sqrt(T) Z - sigma T)
        d S_T / d r     = T S_T
    gamma: LIKELIHOOD-RATIO estimator. Pathwise fails for gamma because the payoff's
        second derivative is a spike at the strike (a Dirac delta) - you cannot
        differentiate max(S-K, 0) twice along a path.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    sqT = np.sqrt(T)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqT * Z)
    disc = np.exp(-r * T)
    sign = 1.0 if option == "call" else -1.0
    itm = (ST > K) if option == "call" else (ST < K)
    payoff = np.maximum(sign * (ST - K), 0.0)

    price = disc * payoff
    delta = disc * sign * itm * ST / S0
    vega = disc * sign * itm * ST * (sqT * Z - sigma * T)
    rho = disc * T * (sign * itm * ST - payoff)
    gamma = disc * payoff * ((Z**2 - 1.0) / (S0**2 * sigma**2 * T) - Z / (S0**2 * sigma * sqT))

    return {"price": _mean_se(price), "delta": _mean_se(delta), "vega": _mean_se(vega),
            "rho": _mean_se(rho), "gamma": _mean_se(gamma)}


# ---------------- Plot: Greeks vs spot ----------------
def plot_greeks(K=100, r=0.05, sigma=0.2, T=0.5, path="greeks_vs_spot.png"):
    import matplotlib.pyplot as plt
    S = np.linspace(60, 140, 300)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    for opt, ls in (("call", "-"), ("put", "--")):
        g = bs_greeks(S, K, r, sigma, T, opt)
        for i, name in enumerate(("delta", "gamma", "vega", "theta", "rho")):
            ax[i].plot(S, g[name], ls, label=opt)
            ax[i].set_title(name)
    for T_ in (1.0, 0.25, 0.05):                         # gamma sharpens as expiry nears
        ax[5].plot(S, bs_greeks(S, K, r, sigma, T_, "call")["gamma"], label=f"T={T_}")
    ax[5].set_title("gamma vs time to expiry")
    for a in ax:
        a.axvline(K, color="grey", ls=":")
        a.grid(alpha=0.3); a.legend(fontsize=8); a.set_xlabel("Spot S")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    for opt in ("call", "put"):
        a, f = bs_greeks(S, K, r, sigma, T, opt), fd_greeks(S, K, r, sigma, T, opt)
        m = mc_greeks(S, K, r, sigma, T, 1_000_000, opt, seed=0)
        print(f"\n{opt.upper()}   (S=K=100, r=5%, sigma=20%, T=1)")
        print(f"{'greek':<7}{'analytic':>12}{'finite-diff':>14}{'monte-carlo':>26}")
        for n in ("delta", "gamma", "vega", "theta", "rho"):
            est, se = m[n] if n != "theta" else (float("nan"), float("nan"))
            mc = f"{est:10.5f} +/- {se:.5f}" if n != "theta" else "      (bump & reprice)"
            print(f"{n:<7}{a[n]:12.5f}{f[n]:14.5f}{mc:>26}")
    print("\nSaved:", plot_greeks())