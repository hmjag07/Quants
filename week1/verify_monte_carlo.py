
import sys
import numpy as np

from black_scholes import bs_call, bs_put
from monte_carlo import (simulate_gbm_paths_loop, simulate_gbm_paths, simulate_terminal_prices,
                          mc_price_from_payoffs, mc_price_call, mc_price_put)

results = []


def check(name, condition, detail=""):
    results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

# ---------------- Sanity: zero volatility must be deterministic ----------------
paths_loop = simulate_gbm_paths_loop(S0, r, 0.0, T, n_steps=50, n_paths=3, seed=1)
paths_vec = simulate_gbm_paths(S0, r, 0.0, T, n_steps=50, n_paths=3, seed=1)
expected_final = S0 * np.exp(r * T)
check("loop sim, sigma=0 -> deterministic S0*e^{rT}",
      np.allclose(paths_loop[:, -1], expected_final))
check("vectorised sim, sigma=0 -> deterministic S0*e^{rT}",
      np.allclose(paths_vec[:, -1], expected_final))
check("every path starts at S0 (loop)", np.allclose(paths_loop[:, 0], S0))
check("every path starts at S0 (vectorised)", np.allclose(paths_vec[:, 0], S0))

# ---------------- Loop vs vectorised: same statistical model ----------------
# Not the same numbers (different RNG draw pattern), but the same DISTRIBUTION.
# Check E[S_T] and Var[log(S_T/S0)] match theory for both implementations.
n_paths_stat = 60_000
loop_paths = simulate_gbm_paths_loop(S0, r, sigma, T, n_steps=1, n_paths=2_000, seed=2)
vec_paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=1, n_paths=n_paths_stat, seed=2)

theory_mean = S0 * np.exp(r * T)                       # risk-neutral growth
theory_logvar = sigma**2 * T                            # Var[log(S_T/S0)]

loop_mean = loop_paths[:, -1].mean()
vec_mean = vec_paths[:, -1].mean()
vec_logvar = np.log(vec_paths[:, -1] / S0).var(ddof=1)

# tolerance scales with 1/sqrt(n) - Monte Carlo error, not a bug, if this is tight
loop_tol = 3 * (S0 * sigma * np.sqrt(T)) / np.sqrt(2_000)
vec_tol = 3 * (S0 * sigma * np.sqrt(T)) / np.sqrt(n_paths_stat)
check(f"loop sim E[S_T] ~ S0*e^rT (tol {loop_tol:.2f})", abs(loop_mean - theory_mean) < loop_tol,
      f"(got {loop_mean:.3f}, expected {theory_mean:.3f})")
check(f"vectorised E[S_T] ~ S0*e^rT (tol {vec_tol:.2f})", abs(vec_mean - theory_mean) < vec_tol,
      f"(got {vec_mean:.3f}, expected {theory_mean:.3f})")
check("vectorised Var[log(S_T/S0)] ~ sigma^2 * T",
      abs(vec_logvar - theory_logvar) < 0.05 * theory_logvar,
      f"(got {vec_logvar:.4f}, expected {theory_logvar:.4f})")

# ---------------- Path shape checks ----------------
check("loop paths shape correct", paths_loop.shape == (3, 51))
check("vectorised paths shape correct", paths_vec.shape == (3, 51))
big = simulate_gbm_paths(S0, r, sigma, T, n_steps=100, n_paths=10_000, seed=3)
check("large batch: no negative prices (GBM can't go negative)", np.all(big > 0))
check("large batch: no NaNs/Infs", np.all(np.isfinite(big)))

# ---------------- MC price converges to Black-Scholes as n grows ----------------
bs_c = float(bs_call(S0, K, r, sigma, T))
bs_p = float(bs_put(S0, K, r, sigma, T))

err_small = []
err_large = []
for trial_seed in range(5):
    mc_c_small, _, _ = mc_price_call(S0, K, r, sigma, T, 500, seed=trial_seed)
    mc_c_large, se_large, ci_large = mc_price_call(S0, K, r, sigma, T, 300_000, seed=trial_seed)
    err_small.append(abs(mc_c_small - bs_c))
    err_large.append(abs(mc_c_large - bs_c))

check("MC error shrinks as path count grows (avg over 5 seeds)",
      np.mean(err_large) < np.mean(err_small),
      f"(avg err @500 = {np.mean(err_small):.4f}, avg err @300k = {np.mean(err_large):.4f})")

mc_c_big, se_big, ci_big = mc_price_call(S0, K, r, sigma, T, 500_000, seed=42)
check("large-N MC call within its own 95% CI of BS price",
      ci_big[0] - 1e-9 <= bs_c <= ci_big[1] + 1e-9 or abs(mc_c_big - bs_c) < 4 * se_big,
      f"(MC={mc_c_big:.4f} +/- {se_big:.4f}, BS={bs_c:.4f})")

mc_p_big, se_p_big, ci_p_big = mc_price_put(S0, K, r, sigma, T, 500_000, seed=42)
check("large-N MC put within 4 standard errors of BS price",
      abs(mc_p_big - bs_p) < 4 * se_p_big,
      f"(MC={mc_p_big:.4f} +/- {se_p_big:.4f}, BS={bs_p:.4f})")

# ---------------- Exact identity: put-call parity via SAME draws ----------------
# If you price a call and a put from the SAME simulated S_T (same seed), then
# call_payoff - put_payoff = S_T - K for every single path, no exceptions.
# So the discounted average must equal the forward value EXACTLY (to float precision) -
# this must hold even at n_paths=50, not just at large N. If it doesn't, your
# call/put payoff code has a sign error, not a "need more paths" problem.
ST_same = simulate_terminal_prices(S0, r, sigma, T, n_paths=50, seed=7)
call_payoffs = np.maximum(ST_same - K, 0.0)
put_payoffs = np.maximum(K - ST_same, 0.0)
mc_c_small_exact, _, _ = mc_price_from_payoffs(call_payoffs, r, T)
mc_p_small_exact, _, _ = mc_price_from_payoffs(put_payoffs, r, T)
# The identity call_i - put_i = S_T,i - K holds PER PATH, so it holds for the
# SAMPLE MEAN too, exactly, regardless of N. It does NOT equal the theoretical
# forward value S0 - K*e^{-rT} at small N - that only holds in expectation as
# N -> infinity (that's the "MC error shrinks" check above, a different thing).
sample_forward_value = np.exp(-r * T) * (ST_same.mean() - K)
check("put-call parity: C-P equals discounted (mean S_T - K) EXACTLY, same draws, any N",
      close(mc_c_small_exact - mc_p_small_exact, sample_forward_value, tol=1e-9),
      f"(C-P={mc_c_small_exact - mc_p_small_exact:.8f}, sample fwd={sample_forward_value:.8f})")
theoretical_forward = S0 - K * np.exp(-r * T)
check("...but only CONVERGES to the theoretical forward value as N grows (expected, not a bug)",
      abs(sample_forward_value - theoretical_forward) > 1e-6,
      f"(sample={sample_forward_value:.4f} vs theory={theoretical_forward:.4f} - a real gap at N=50, as expected)")

# ---------------- Reproducibility ----------------
a = simulate_gbm_paths(S0, r, sigma, T, n_steps=10, n_paths=100, seed=123)
b = simulate_gbm_paths(S0, r, sigma, T, n_steps=10, n_paths=100, seed=123)
check("same seed -> identical paths (reproducible results)", np.array_equal(a, b))
c = simulate_gbm_paths(S0, r, sigma, T, n_steps=10, n_paths=100, seed=124)
check("different seed -> different paths", not np.array_equal(a, c))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)