
import time
import numpy as np
from black_scholes import bs_price


# ---------------- Baseline: loop ----------------
def price_loop(S0, K, r, sigma, T, n_paths, option="call", seed=None):
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    total = 0.0
    for _ in range(n_paths):
        ST = S0 * np.exp(drift + vol * rng.standard_normal())
        total += max(ST - K, 0.0) if option == "call" else max(K - ST, 0.0)
    return np.exp(-r * T) * total / n_paths


# ---------------- Vectorised, with variance reduction ----------------
def price_vectorized(S0, K, r, sigma, T, n_paths, option="call", seed=None,
                     antithetic=False, control_variate=False):
    """
    Returns (price, standard_error).

    antithetic:      for every draw Z also use -Z and average the pair. The two payoffs are
                     negatively correlated, so the pair-average has lower variance.
                     n_paths counts TOTAL paths, so the comparison with plain MC is fair.
    control_variate: S_T has a KNOWN expectation (S0 e^{rT}). Subtract beta*(Y - E[Y]) from
                     the payoff, where Y is the discounted S_T; beta is estimated from the sample.
    """
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    sign = 1.0 if option == "call" else -1.0

    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        Zs = np.concatenate([Z[:, None], -Z[:, None]], axis=1)          # (n/2, 2)
    else:
        Zs = rng.standard_normal((n_paths, 1))

    ST = S0 * np.exp(drift + vol * Zs)
    X = (disc * np.maximum(sign * (ST - K), 0.0)).mean(axis=1)           # one value per pair/path
    if control_variate:
        Y = (disc * ST).mean(axis=1)                                      # E[Y] = S0 exactly
        beta = np.cov(X, Y, ddof=1)[0, 1] / Y.var(ddof=1)
        X = X - beta * (Y - S0)
    return float(X.mean()), float(X.std(ddof=1) / np.sqrt(len(X)))


# ---------------- Benchmark ----------------
def benchmark(n_paths=200_000, repeats=3):
    args = (100, 100, 0.05, 0.2, 1.0, n_paths)
    def best_time(f):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter(); f(); times.append(time.perf_counter() - t0)
        return min(times)
    t_loop = best_time(lambda: price_loop(*args, seed=0))
    t_vec = best_time(lambda: price_vectorized(*args, seed=0))
    return {"n_paths": n_paths, "loop_s": t_loop, "vectorized_s": t_vec,
            "speedup": t_loop / t_vec, "runtime_reduction_pct": 100 * (1 - t_vec / t_loop)}


def variance_reduction_table(n_paths=100_000, seed=0):
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs = float(bs_price(S0, K, r, sigma, T, "call"))
    print(f"Black-Scholes call = {bs:.4f}   ({n_paths:,} total paths each)")
    print(f"{'method':<28}{'price':>10}{'std err':>10}{'vs plain':>10}")
    base = None
    for name, kw in (("plain", {}), ("antithetic", {"antithetic": True}),
                     ("control variate", {"control_variate": True}),
                     ("antithetic + control", {"antithetic": True, "control_variate": True})):
        p, se = price_vectorized(S0, K, r, sigma, T, n_paths, "call", seed, **kw)
        base = base or se
        print(f"{name:<28}{p:10.4f}{se:10.4f}{se / base:9.2f}x")


def convergence_table():
    print("\nStandard error vs path count (should fall like 1/sqrt(N): 100x paths -> 10x smaller)")
    print(f"{'paths':>12}{'price':>10}{'std err':>10}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        p, se = price_vectorized(100, 100, 0.05, 0.2, 1.0, n, "call", seed=1)
        print(f"{n:>12,}{p:10.4f}{se:10.4f}")


if __name__ == "__main__":
    b = benchmark()
    print(f"Loop       : {b['loop_s']:.3f}s for {b['n_paths']:,} paths")
    print(f"Vectorised : {b['vectorized_s']:.4f}s   -> {b['speedup']:.0f}x faster "
          f"({b['runtime_reduction_pct']:.1f}% runtime reduction)\n")
    variance_reduction_table()
    convergence_table()