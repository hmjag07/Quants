import contextlib
import csv
import io
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

try:  # the file may be called market_validation.py or Market_Validation.py (GitHub shows the capitals)
    from market_validation import bs_price_q, forward_from_parity
except ModuleNotFoundError:
    from Market_Validation import bs_price_q, forward_from_parity
import nse_validation as nv

passed = failed = 0


def check(name, ok):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text.lower() in str(e).lower()
    return False


# ------------------------------------------------------------------ synthetic data
S0, R0, T0 = 24000.0, 0.065, 8 / 365
EXPIRY = datetime(2026, 10, 27)
ASOF = datetime(2026, 10, 19, 15, 30)


def smile_chain(smile, S=S0, r=R0, T=T0, noise=0.0, seed=0, step=50, width=0.10):
    """Exact Black-Scholes prices from a known smile sigma(ln(K/F)); optional multiplicative noise."""
    rng = np.random.default_rng(seed)
    F = S * np.exp(r * T)
    rows = []
    for K in np.arange(round(S * (1 - width) / step) * step, S * (1 + width) + 1, step):
        sig = smile(np.log(K / F))
        for opt in ("call", "put"):
            p = float(bs_price_q(S, K, r, sig, T, opt)) * (
                1 + noise * rng.standard_normal()
            )
            rows.append(
                {
                    "strike": float(K),
                    "option": opt,
                    "market_price": round(p, 2),
                    "price_source": "x",
                }
            )
    return pd.DataFrame(rows)


def write_page(path, chain, stale_ltp=True, extra_rows=()):
    """NSE option-chain page layout: CALLS | STRIKE | PUTS, numbers with thousands commas, '-' for blanks."""
    head1 = ["", "CALLS"] + [""] * 10 + ["PUTS"] + [""] * 9
    head2 = [
        "",
        "OI",
        "CHNG IN OI",
        "VOLUME",
        "IV",
        "LTP",
        "CHNG",
        "BID QTY",
        "BID",
        "ASK",
        "ASK QTY",
        "STRIKE",
        "BID QTY",
        "BID",
        "ASK",
        "ASK QTY",
        "CHNG",
        "LTP",
        "IV",
        "VOLUME",
        "CHNG IN OI",
        "OI",
        "",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head1)
        w.writerow(head2)
        for K, g in chain.groupby("strike"):
            c = float(g[g.option == "call"].market_price.iloc[0])
            p = float(g[g.option == "put"].market_price.iloc[0])

            def side(x):
                half = max(0.05, 0.02 * x) / 2
                ltp = x * 1.03 if stale_ltp else x
                return [
                    f"{ltp:,.2f}",
                    "-",
                    f"{x - half:,.2f}",
                    f"{x + half:,.2f}",
                    "100",
                ]  # ltp, chng, bid, ask, qty

            cl, pl = side(c), side(p)
            w.writerow(
                [
                    "",
                    "1,000",
                    "-",
                    "5,000",
                    "12.5",
                    cl[0],
                    "-",
                    "75",
                    cl[2],
                    cl[3],
                    "75",
                    f"{K:,.2f}",
                    "75",
                    pl[2],
                    pl[3],
                    "75",
                    "-",
                    pl[0],
                    "12.9",
                    "4,000",
                    "-",
                    "900",
                    "",
                ]
            )
        for r in extra_rows:
            w.writerow(r)


def write_udiff(path, chain, expiry="2026-10-27"):
    rows = []
    for _, r in chain.iterrows():
        rows.append(
            {
                "TradDt": "2026-10-19",
                "TckrSymb": "NIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": expiry,
                "StrkPric": r.strike,
                "OptnTp": "CE" if r.option == "call" else "PE",
                "ClsPric": r.market_price,
                "UndrlygPric": 24010.5,
            }
        )
    rows.append(
        {
            "TradDt": "2026-10-19",
            "TckrSymb": "NIFTY",
            "FinInstrmTp": "IDF",
            "XpryDt": expiry,
            "StrkPric": "",
            "OptnTp": "",
            "ClsPric": 24020.0,
            "UndrlygPric": 24010.5,
        }
    )  # a future
    for _, r in chain.head(6).iterrows():  # other symbol
        rows.append(
            {
                "TradDt": "2026-10-19",
                "TckrSymb": "BANKNIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": expiry,
                "StrkPric": r.strike * 2,
                "OptnTp": "CE",
                "ClsPric": 999.99,
                "UndrlygPric": 52000.0,
            }
        )
    for _, r in chain.head(6).iterrows():  # other expiry
        rows.append(
            {
                "TradDt": "2026-10-19",
                "TckrSymb": "NIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": "2026-11-24",
                "StrkPric": r.strike,
                "OptnTp": "PE",
                "ClsPric": 888.88,
                "UndrlygPric": 24010.5,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_old_bhav(path, chain):
    rows = [
        {
            "INSTRUMENT": "OPTIDX",
            "SYMBOL": "NIFTY",
            "EXPIRY_DT": "27-Oct-2026",
            "STRIKE_PR": r.strike,
            "OPTION_TYP": "CE" if r.option == "call" else "PE",
            "CLOSE": r.market_price,
            "TIMESTAMP": "19-Oct-2026",
        }
        for _, r in chain.iterrows()
    ]
    rows.append(
        {
            "INSTRUMENT": "OPTSTK",
            "SYMBOL": "NIFTY",
            "EXPIRY_DT": "27-Oct-2026",
            "STRIKE_PR": 100,
            "OPTION_TYP": "CE",
            "CLOSE": 777.77,
            "TIMESTAMP": "19-Oct-2026",
        }
    )  # stock-option row
    pd.DataFrame(rows).to_csv(path, index=False)


def quiet_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = nv.main(argv)
    return code, buf.getvalue()


tmp = tempfile.mkdtemp(prefix="nse_verify_")
P = lambda name: os.path.join(tmp, name)

# ------------------------------------------------------------------ small helpers
print("=== Parsing helpers ===")
check("'24,950.00' parses to 24950.0", nv.parse_number("24,950.00") == 24950.0)
check(
    "'-', '' and 'NA' parse to nan",
    all(np.isnan(nv.parse_number(x)) for x in ("-", "", "NA", None)),
)
check(
    "dates: 2026-10-27, 27-Oct-2026, 27/10/2026 all read as the same day",
    nv.parse_date("2026-10-27")
    == nv.parse_date("27-Oct-2026")
    == nv.parse_date("27/10/2026")
    == datetime(2026, 10, 27),
)
check(
    "a bare --asof date means the 15:30 close",
    nv.parse_asof("2026-10-19") == datetime(2026, 10, 19, 15, 30),
)
check(
    "--asof with a time is read exactly",
    nv.parse_asof("2026-10-19 15:15") == datetime(2026, 10, 19, 15, 15),
)
check("a garbage --asof is rejected", raises(lambda: nv.parse_asof("tomorrow"), "asof"))
check(
    "time to expiry: 19 Oct 15:30 -> 27 Oct 15:30 is exactly 8/365",
    abs(nv.time_to_expiry(EXPIRY, ASOF) - 8 / 365) < 1e-12,
)
check(
    "quotes after expiry are rejected",
    raises(
        lambda: nv.time_to_expiry(EXPIRY, datetime(2026, 10, 28, 9, 15)), "after expiry"
    ),
)

print("\n=== Price rule ===")
check("valid bid/ask -> mid", nv.choose_price(110, 99, 101) == (100.0, "mid"))
check(
    "no bid -> falls back to last traded price",
    nv.choose_price(110, 0, 101) == (110, "ltp"),
)
check(
    "crossed market (ask < bid) -> falls back to last traded price",
    nv.choose_price(110, 105, 100) == (110, "ltp"),
)
check(
    "rule 'ltp' ignores bid/ask",
    nv.choose_price(110, 99, 101, rule="ltp") == (110, "ltp"),
)
check(
    "very wide market (bid 1, ask 100) is rejected",
    np.isnan(nv.choose_price(50, 1, 100)[0])
    and nv.choose_price(50, 1, 100)[1] == "wide",
)
check("nothing usable -> nan", np.isnan(nv.choose_price(0, 0, 0)[0]))

# ------------------------------------------------------------------ layouts
print("\n=== Reading the three file layouts ===")
smile = lambda m: 0.14 + 3.0 * m * m
base = smile_chain(smile)

write_page(P("page.csv"), base)
page, meta = nv.load_chain(P("page.csv"))
check("option-chain page layout is detected", meta["layout"] == "NSE option-chain page")
key = lambda d: d.sort_values(["strike", "option"]).reset_index(drop=True)
a, b = key(page), key(base)
check(
    "page file: every strike and both sides recovered",
    len(a) == len(b) and (a.strike == b.strike).all() and (a.option == b.option).all(),
)
real = (
    b.market_price >= 2
).values  # premiums below 2 points can fall back to LTP or be rejected as too wide, by design
check(
    "page file: for real premiums the bid/ask mid (not the stale +3% last price) is used",
    (a.price_source[real] == "mid").all()
    and np.allclose(a.market_price[real], b.market_price[real], atol=0.011),
)
check(
    "page file: tiny premiums are handled by the rules (mid, last price, 'wide' or 'none'), never invented",
    set(a.price_source) <= {"mid", "ltp", "wide", "none"}
    and a.market_price[a.price_source.isin(["wide", "none"])].isna().all(),
)
page_ltp, _ = nv.load_chain(P("page.csv"), rule="ltp")
check(
    "page file with rule 'ltp' uses the last traded price (for real premiums)",
    np.allclose(
        key(page_ltp).market_price[real],
        b.market_price[real] * 1.03,
        atol=0.011,
        rtol=1e-3,
    ),
)

write_page(
    P("page_gaps.csv"),
    base,
    extra_rows=[
        [
            "",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "30,000.00",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "",
        ]
    ],
)
gaps, _ = nv.load_chain(P("page_gaps.csv"))
check(
    "a strike whose cells are all '-' gives no usable price",
    gaps[gaps.strike == 30000.0].market_price.isna().all(),
)

write_udiff(P("udiff.csv"), base)
ud, meta_u = nv.load_chain(P("udiff.csv"), "NIFTY", EXPIRY)
check("UDiFF bhavcopy layout is detected", meta_u["layout"] == "NSE F&O bhavcopy")
n_priced = int(
    (base.market_price > 0).sum()
)  # the generator rounds a few far-wing prices to 0.00, which are dropped
check(
    "UDiFF: futures, other symbols and other expiries are filtered out",
    len(ud) == n_priced
    and set(ud.option) == {"call", "put"}
    and not ud.market_price.isin([999.99, 888.88]).any(),
)
check(
    "UDiFF: the file's underlying price and trade date are picked up",
    abs(meta_u["spot"] - 24010.5) < 1e-9
    and meta_u["trade_date"] == datetime(2026, 10, 19),
)
check(
    "UDiFF: two expiries without --expiry is an error that lists them",
    raises(lambda: nv.load_chain(P("udiff.csv"), "NIFTY", None), "2026-10-27"),
)

write_old_bhav(P("old.csv"), base)
old, meta_o = nv.load_chain(P("old.csv"), "NIFTY")
check(
    "old bhavcopy layout works and drops the stock-option row",
    len(old) == n_priced and not old.market_price.isin([777.77]).any(),
)
check(
    "old bhavcopy expiry and date read from 27-Oct-2026 style",
    meta_o["expiry"] == EXPIRY and meta_o["trade_date"] == datetime(2026, 10, 19),
)

simple = base[["strike", "option", "market_price"]]
simple.to_csv(P("simple.csv"), index=False)
sm, meta_s = nv.load_chain(P("simple.csv"))
check(
    "simple strike/option/market_price CSV works (zero prices dropped)",
    meta_s["layout"].startswith("simple") and len(sm) == n_priced,
)

with open(P("junk.csv"), "w") as f:
    f.write("foo,bar\n1,2\n")
check(
    "an unrecognised file gives a helpful error",
    raises(lambda: nv.load_chain(P("junk.csv")), "could not recognise"),
)
open(P("empty.csv"), "w").close()
check(
    "an empty file gives an error",
    raises(lambda: nv.load_chain(P("empty.csv")), "empty"),
)

# ------------------------------------------------------------------ the maths
print("\n=== Forward and filters ===")
noisy = smile_chain(smile, noise=0.01, seed=3)
F_true = S0 * np.exp(R0 * T0)
F_est = nv.forward_robust(noisy, R0, T0)
check(
    "forward from parity within 0.05% of the truth on a chain with 1% quote noise",
    abs(F_est / F_true - 1) < 5e-4,
)

bad = base.copy()
k0 = round(S0 * 0.97 / 50) * 50
bad.loc[bad.strike == k0, "market_price"] = (
    100.0  # a stale quote that makes C - P look like zero at a far strike
)
F_single = forward_from_parity(bad, R0, T0)
F_med = nv.forward_robust(bad, R0, T0)
check(
    "one stale far-strike quote ruins the single-strike forward (>1% off)",
    abs(F_single / F_true - 1) > 0.01,
)
check(
    "the median-of-5 forward shrugs it off (<0.05% off)", abs(F_med / F_true - 1) < 5e-4
)
check(
    "no strike with both a call and a put -> clear error",
    raises(
        lambda: nv.forward_robust(base[base.option == "call"], R0, T0),
        "both a call and a put",
    ),
)

oc = nv.otm_only(base, F_true)
check(
    "OTM filter keeps only calls above and puts below the forward",
    ((oc[oc.option == "call"].strike >= F_true).all())
    and ((oc[oc.option == "put"].strike <= F_true).all())
    and len(oc) < len(base),
)

print("\n=== Validation maths against known answers ===")
flat = smile_chain(lambda m: 0.18 + 0 * m)
res = nv.run_validation(flat, EXPIRY, ASOF, R0)
check(
    "a chain priced with ONE volatility (18%) is fitted back to 18%",
    abs(res["sigma"] - 0.18) < 1e-3,
)
check(
    "...and the model then matches the market closely everywhere (mean |error| < 0.3%)",
    res["detail"].pct_err.abs().mean() < 0.3,
)
check(
    "implied vols of that chain are all 18%",
    np.nanmax(np.abs(res["detail"].iv - 0.18)) < 2e-3,
)

res_s = nv.run_validation(base, EXPIRY, ASOF, R0)
d = res_s["detail"]
expected_iv = np.array([smile(np.log(k / F_true)) for k in d.strike])
mask = d.iv.notna() & (d.market_price > 5)
check(
    "implied vols recover the true smile (within 0.3 vol points, premiums > 5)",
    np.nanmax(np.abs(d.iv[mask] - expected_iv[mask])) < 3e-3,
)
s = res_s["summary"]
check(
    "with a real smile, error grows with distance from the forward",
    s["mean abs % err"].iloc[0] < s["mean abs % err"].iloc[-1],
)
check(
    "the summary has the expected columns (including the vol-point gap)",
    list(s.columns)
    == [
        "|ln(K/F)| bucket",
        "n",
        "mean abs % err",
        "median abs % err",
        "% within 5%",
        "mean abs IV gap (vol pts)",
    ],
)
check(
    "with a real smile, the implied vol drifts further from the flat fit as we move away from the forward",
    s["mean abs IV gap (vol pts)"].iloc[0] < s["mean abs IV gap (vol pts)"].iloc[-1],
)
check(
    "on a flat-vol chain the vol-point gap is ~0 in every bucket",
    (res["summary"]["mean abs IV gap (vol pts)"] < 0.2).all(),
)

res_a = nv.run_validation(base, EXPIRY, ASOF, R0, use_otm_only=False)
check(
    "including in-the-money options adds rows",
    len(res_a["detail"]) > len(res_s["detail"]),
)

res_spot = nv.run_validation(base, EXPIRY, ASOF, R0, spot=24100.0)
check(
    "the spot price does not change the answer (it cancels against the inferred carry)",
    abs(res_spot["sigma"] - res_s["sigma"]) < 1e-9
    and np.allclose(res_spot["detail"].model_price, d.model_price, atol=1e-6),
)
check(
    "an implied carry is reported when the spot is given",
    np.isfinite(res_spot["q"]) and abs(res_spot["q"] - res_s["q"]) > 1e-4,
)
res_lo = nv.run_validation(base, EXPIRY, ASOF, 0.05)
res_hi = nv.run_validation(base, EXPIRY, ASOF, 0.07)
check(
    "the risk-free rate barely matters (flat sigma moves < 0.3 vol points for 5% vs 7%)",
    abs(res_lo["sigma"] - res_hi["sigma"]) < 3e-3,
)
# the parity formula itself, worked by hand: F = K + e^{rT} (C - P)
hand = pd.DataFrame(
    {
        "strike": [1000.0, 1000.0],
        "option": ["call", "put"],
        "market_price": [150.0, 50.0],
        "price_source": "x",
    }
)
check(
    "parity by hand: K=1000, C-P=100, r=10%, T=1y gives F = 1000 + 100*e^0.1 = 1110.5171",
    abs(nv.forward_robust(hand, 0.10, 1.0) - (1000 + 100 * np.exp(0.10))) < 1e-9,
)

# run_validation must be as robust as forward_robust: one corrupt far-strike quote must not move the answer
res_bad = nv.run_validation(bad, EXPIRY, ASOF, R0)
check(
    "run_validation: a stale far-strike quote leaves the forward within 0.05%",
    abs(res_bad["F"] / F_true - 1) < 5e-4,
)
check(
    "run_validation: ...and the fitted sigma within 0.2 vol points of the clean run",
    abs(res_bad["sigma"] - res_s["sigma"]) < 2e-3,
)
check(
    "without a spot, the reported carry equals the rate (spot is set to the forward)",
    abs(res_s["q"] - R0) < 1e-12,
)
res_mp = nv.run_validation(base, EXPIRY, ASOF, R0, min_price=50.0)
check(
    "min_price filter: every option used has a premium of at least 50",
    (res_mp["detail"].market_price >= 50).all()
    and len(res_mp["detail"]) < len(res_s["detail"]),
)

few = base[(base.strike >= F_true - 60) & (base.strike <= F_true + 60)]
check(
    "too few options near the forward -> clear error",
    raises(lambda: nv.run_validation(few, EXPIRY, ASOF, R0), "need at least 4"),
)

# ------------------------------------------------------------------ end to end
print("\n=== End to end through the command line ===")
out_dirs = {}
runs = {
    "page": [
        "--csv",
        P("page.csv"),
        "--expiry",
        "2026-10-27",
        "--asof",
        "2026-10-19 15:30",
        "--rate",
        "0.065",
    ],
    "udiff": ["--csv", P("udiff.csv"), "--expiry", "2026-10-27", "--rate", "0.065"],
    "old": ["--csv", P("old.csv"), "--rate", "0.065"],
    "simple": [
        "--csv",
        P("simple.csv"),
        "--expiry",
        "2026-10-27",
        "--asof",
        "2026-10-19",
        "--rate",
        "0.065",
    ],
}
sigmas = {}
for name, argv in runs.items():
    out = os.path.join(tmp, "out_" + name)
    code, text = quiet_main(argv + ["--out-dir", out])
    out_dirs[name] = out
    check(f"{name}: command runs and exits 0", code == 0)
    res_csv = os.path.join(out, "nse_validation_results.csv")
    if os.path.exists(res_csv):
        sigmas[name] = text


def fitted_sigma(text):
    return float(
        next(l for l in text.splitlines() if l.startswith("Flat sigma"))
        .split(":")[1]
        .strip()
        .rstrip("%")
    )


vals = {k: fitted_sigma(t) for k, t in sigmas.items()}
check(
    "all four layouts of the same market give the same fitted sigma (within 0.05 vol points)",
    len(vals) == 4 and max(vals.values()) - min(vals.values()) < 0.05,
)

o = out_dirs["page"]
check(
    "results CSV, figure and README section are all written",
    all(
        os.path.exists(os.path.join(o, f)) and os.path.getsize(os.path.join(o, f)) > 0
        for f in ("nse_validation_results.csv", "nse_iv_smile.png", "readme_section.md")
    ),
)
section = open(os.path.join(o, "readme_section.md"), encoding="utf-8").read()
check(
    "README section names the expiry, the quote time and has a markdown table",
    "27 Oct 2026" in section
    and "19 Oct 2026 15:30" in section
    and "|---|" in section.replace(" ", ""),
)
check(
    "README section does not claim more than was done (no 'beats' or 'outperforms')",
    "outperform" not in section.lower() and "beats" not in section.lower(),
)
hdr = [l for l in section.splitlines() if l.startswith("| ") and "bucket" in l][0]
check(
    "markdown table: the header's pipe characters are escaped, so GitHub shows 6 columns",
    hdr.replace("\\|", "").count("|") == 7 and "\\|ln(K/F)\\| bucket" in hdr,
)
check(
    "the README sentence follows the data: smile chain -> says it degrades in the wings",
    "consistent with a volatility smile" in nv.conclusion(res_s["summary"]),
)
check(
    "the README sentence follows the data: flat chain -> says no smile is visible",
    "no smile is visible" in nv.conclusion(res["summary"])
    and "consistent with a volatility smile" not in nv.conclusion(res["summary"]),
)
noisy_flat = pd.DataFrame(
    {
        "|ln(K/F)| bucket": ["a", "b"],
        "n": [5, 5],
        "mean abs % err": [1.0, 1.0],
        "median abs % err": [1.0, 1.0],
        "% within 5%": [100.0, 100.0],
        "mean abs IV gap (vol pts)": [2.0, 1.8],
    }
)
check(
    "the README sentence follows the data: big gap everywhere but not growing -> says the error did NOT grow",
    "did NOT grow" in nv.conclusion(noisy_flat),
)
check(
    "the README sentence: a single bucket -> says there is too little to compare",
    "Too few" in nv.conclusion(noisy_flat.iloc[:1]),
)
png = open(os.path.join(o, "nse_iv_smile.png"), "rb").read(8)
check("the figure is a real PNG", png == b"\x89PNG\r\n\x1a\n")

code, text = quiet_main(
    ["--csv", P("page.csv"), "--asof", "2026-10-19 15:30", "--out-dir", P("x")]
)
check(
    "page file without --expiry: exit code 2 and a message",
    code == 2 and "--expiry" in text,
)
code, text = quiet_main(
    ["--csv", P("page.csv"), "--expiry", "2026-10-27", "--out-dir", P("x")]
)
check(
    "page file without --asof: exit code 2 and a message",
    code == 2 and "--asof" in text,
)
code, text = quiet_main(
    [
        "--csv",
        P("page.csv"),
        "--expiry",
        "2026-10-27",
        "--asof",
        "2026-10-28 10:00",
        "--out-dir",
        P("x"),
    ]
)
check("quotes after expiry: exit code 2", code == 2 and "after expiry" in text)
code, text = quiet_main(
    [
        "--csv",
        P("nope.csv"),
        "--expiry",
        "2026-10-27",
        "--asof",
        "2026-10-19",
        "--out-dir",
        P("x"),
    ]
)
check(
    "a missing file does not crash the program silently", code != 0 or "ERROR" in text
)

print(f"\n{passed}/{passed + failed} checks passed")
sys.exit(0 if failed == 0 else 1)
