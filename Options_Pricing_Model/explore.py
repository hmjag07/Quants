
import numpy as np
import matplotlib.pyplot as plt
from payoffs import call_profit, put_profit, straddle_profit
from black_scholes import bs_call, bs_put
from bonds import coupon_bond_price

# %% Payoff at expiry vs. BS value today (how much "time value" is left?)
S = np.linspace(60, 140, 200)
K, r, sigma = 100, 0.05, 0.2
plt.figure(figsize=(8, 5))
plt.plot(S, np.maximum(S - K, 0), label="Payoff at expiry")
for T in (1.0, 0.25, 0.05):
    plt.plot(S, bs_call(S, K, r, sigma, T), label=f"BS call, T={T}")
plt.axvline(K, color="grey", ls=":")
plt.xlabel("Spot S"); plt.ylabel("Call value"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Time value shrinks as expiry approaches")
plt.show()

# %% Sensitivity of call price to volatility
vols = np.linspace(0.05, 0.6, 50)
plt.figure(figsize=(8, 4))
plt.plot(vols, bs_call(100, 100, 0.05, vols, 1.0))
plt.xlabel("Volatility"); plt.ylabel("ATM call price"); plt.grid(alpha=0.3)
plt.show()

# %% Bond price vs. yield (the convex curve you saw in the Caltech bonds videos)
ytms = np.linspace(0.01, 0.12, 50)
prices = [coupon_bond_price(100, 0.05, y, 10) for y in ytms]
plt.figure(figsize=(8, 4))
plt.plot(ytms * 100, prices)
plt.xlabel("Yield (%)"); plt.ylabel("Price of 10y 5% bond"); plt.grid(alpha=0.3)
plt.show()
