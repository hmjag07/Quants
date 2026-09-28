"""
Week 1 - Day 3/4/5: Payoff functions and diagrams.

Payoff  = value at expiry, ignoring what you paid.
Profit  = payoff minus the premium you paid (or plus the premium you received).
Convention: position = +1 for long, -1 for short.
"""
import numpy as np


# ---------- Linear instruments (Day 3) ----------
def stock_payoff(S_T, S0, position=1):
    """P&L of holding a stock bought/sold at S0."""
    return position * (np.asarray(S_T, dtype=float) - S0)


def forward_payoff(S_T, K, position=1):
    """Forward/futures contract with delivery price K (costs nothing to enter)."""
    return position * (np.asarray(S_T, dtype=float) - K)


# ---------- Options (Day 4) ----------
def call_payoff(S_T, K, position=1):
    """Payoff of a call at expiry: max(S_T - K, 0)."""
    return position * np.maximum(np.asarray(S_T, dtype=float) - K, 0.0)


def put_payoff(S_T, K, position=1):
    """Payoff of a put at expiry: max(K - S_T, 0)."""
    return position * np.maximum(K - np.asarray(S_T, dtype=float), 0.0)


def call_profit(S_T, K, premium, position=1):
    """Long: payoff - premium paid. Short: premium received - payoff."""
    return call_payoff(S_T, K, position) - position * premium


def put_profit(S_T, K, premium, position=1):
    return put_payoff(S_T, K, position) - position * premium


# ---------- Combinations (Day 5) ----------
def straddle_profit(S_T, K, call_prem, put_prem, position=1):
    """Long straddle: long call + long put, same strike. Bets on big moves."""
    return call_profit(S_T, K, call_prem, position) + put_profit(S_T, K, put_prem, position)


def strangle_profit(S_T, K_put, K_call, put_prem, call_prem, position=1):
    """Long strangle: long OTM put (K_put) + long OTM call (K_call), K_put < K_call."""
    return call_profit(S_T, K_call, call_prem, position) + put_profit(S_T, K_put, put_prem, position)


def bull_call_spread_profit(S_T, K1, K2, prem1, prem2):
    """Long call at K1 (paid prem1) + short call at K2 (received prem2), K1 < K2."""
    return call_profit(S_T, K1, prem1, +1) + call_profit(S_T, K2, prem2, -1)


def butterfly_profit(S_T, K1, K2, K3, prem1, prem2, prem3):
    """Long call K1, short 2 calls K2, long call K3, with K2 the midpoint."""
    return (call_profit(S_T, K1, prem1, +1)
            + 2 * call_profit(S_T, K2, prem2, -1)
            + call_profit(S_T, K3, prem3, +1))


# ---------- Plotting ----------
def plot_all(save_dir="."):
    import os
    import matplotlib.pyplot as plt

    S = np.linspace(50, 150, 401)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()

    # 1. Stock and forward
    ax[0].plot(S, stock_payoff(S, 100, +1), label="Long stock (bought @100)")
    ax[0].plot(S, forward_payoff(S, 100, -1), label="Short forward (K=100)")
    ax[0].set_title("Stock & forward")

    # 2. Long/short call
    ax[1].plot(S, call_profit(S, 100, 5, +1), label="Long call K=100, prem 5")
    ax[1].plot(S, call_profit(S, 100, 5, -1), label="Short call")
    ax[1].set_title("Call profit at expiry")

    # 3. Long/short put
    ax[2].plot(S, put_profit(S, 100, 4, +1), label="Long put K=100, prem 4")
    ax[2].plot(S, put_profit(S, 100, 4, -1), label="Short put")
    ax[2].set_title("Put profit at expiry")

    # 4. Straddle
    ax[3].plot(S, straddle_profit(S, 100, 5, 4), label="Long straddle K=100")
    ax[3].set_title("Straddle")

    # 5. Bull call spread
    ax[4].plot(S, bull_call_spread_profit(S, 95, 105, 8, 3), label="Bull call spread 95/105")
    ax[4].set_title("Bull call spread")

    # 6. Butterfly
    ax[5].plot(S, butterfly_profit(S, 90, 100, 110, 12, 6, 2), label="Butterfly 90/100/110")
    ax[5].set_title("Butterfly")

    for a in ax:
        a.axhline(0, color="black", lw=0.8)
        a.set_xlabel("Stock price at expiry $S_T$")
        a.set_ylabel("Profit")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(save_dir, "payoff_diagrams.png")
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Saved:", plot_all())
    plt.show()
