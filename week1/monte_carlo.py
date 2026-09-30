
import numpy as np
from black_scholes import bs_call, bs_put

def simulate_gbm_paths_loop(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Returns an array of shape (n_paths, n_steps+1): full price path for every simulation.
    Deliberately unvectorised - you can see exactly what happens at each time step.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1))
    for p in range(n_paths):                      # one path at a time
        paths[p, 0] = S0
        for t in range(n_steps):                  # one time step at a time
            Z = rng.standard_normal()
            paths[p, t + 1] = paths[p, t] * np.exp(drift + vol * Z)
    return paths



def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Same model as above, but all paths and all time steps generated in one shot.
    This is what lets Week 3 jump from a few hundred paths to 10,000+.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    increments = drift + vol * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])   # start every path at log(S0)
    return S0 * np.exp(log_paths)


def simulate_terminal_prices(S0, r, sigma, T, n_paths, seed=None):
    """
    Shortcut for pricing: a European option only needs S_T, not the whole path.
    One time step is enough, and it is exact (not an approximation).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def mc_price_from_payoffs(payoffs, r, T):
    """
    Discount the average payoff, and report the Monte Carlo standard error
    and a 95% confidence interval. Always report the error alongside the price -
    a Monte Carlo number without a standard error is not a finished result.
    """
    disc_payoffs = np.exp(-r * T) * payoffs
    price = disc_payoffs.mean()
    stderr = disc_payoffs.std(ddof=1) / np.sqrt(len(payoffs))
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)
    return price, stderr, ci95


def mc_price_call(S0, K, r, sigma, T, n_paths, seed=None):
    ST = simulate_terminal_prices(S0, r, sigma, T, n_paths, seed)
    payoffs = np.maximum(ST - K, 0.0)
    return mc_price_from_payoffs(payoffs, r, T)


def mc_price_put(S0, K, r, sigma, T, n_paths, seed=None):
    ST = simulate_terminal_prices(S0, r, sigma, T, n_paths, seed)
    payoffs = np.maximum(K - ST, 0.0)
    return mc_price_from_payoffs(payoffs, r, T)


def compare_to_bs(S0, K, r, sigma, T, n_paths, seed=0):
    """Prints MC price vs closed-form BS price for both call and put."""
    mc_c, se_c, ci_c = mc_price_call(S0, K, r, sigma, T, n_paths, seed)
    mc_p, se_p, ci_p = mc_price_put(S0, K, r, sigma, T, n_paths, seed)
    bs_c, bs_p = float(bs_call(S0, K, r, sigma, T)), float(bs_put(S0, K, r, sigma, T))
    print(f"n_paths = {n_paths:,}")
    print(f"  Call:  MC = {mc_c:.4f} +/- {se_c:.4f}   (95% CI {ci_c[0]:.4f}, {ci_c[1]:.4f})   BS = {bs_c:.4f}")
    print(f"  Put :  MC = {mc_p:.4f} +/- {se_p:.4f}   (95% CI {ci_p[0]:.4f}, {ci_p[1]:.4f})   BS = {bs_p:.4f}")


if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

    # Show a handful of full paths (this is the "aha, I can see GBM" moment)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=5, seed=1)
    print("Sample terminal prices from 5 simulated paths:", np.round(paths[:, -1], 2))

    # Price with increasing path counts - watch the MC price converge to BS
    for n in (1_000, 20_000, 500_000):
        compare_to_bs(S0, K, r, sigma, T, n, seed=0)
        print()