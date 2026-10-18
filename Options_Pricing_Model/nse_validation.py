import argparse
import csv
import os
import sys
from datetime import datetime, time

import numpy as np
import pandas as pd

try:  # the file may be called market_validation.py or Market_Validation.py (GitHub shows the capitals)
    from market_validation import calibrate_flat_vol, summarize, validate_chain
except ModuleNotFoundError:
    from Market_Validation import calibrate_flat_vol, summarize, validate_chain

EXPIRY_TIME = time(15, 30)  # NSE equity-derivative contracts stop trading at 15:30 IST
DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%b-%Y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d-%b-%y",
    "%Y/%m/%d",
    "%d%b%Y",
)


# --------------------------------------------------------------------------- small helpers
def parse_number(x):
    """'24,950.00' -> 24950.0 ; '-', '', 'NA' -> nan."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("\u20b9", "")
    if s in ("", "-", "--", "NA", "N/A", "nan", "NaN", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_date(x):
    """Understands 2026-10-27, 27-Oct-2026, 27/10/2026, ... Returns a datetime (midnight) or None."""
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_asof(text):
    """'2026-10-19 15:15' or '2026-10-19' (a bare date means the 15:30 close)."""
    s = str(text).strip()
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%d-%m-%Y %H:%M",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    d = parse_date(s)
    if d is None:
        raise ValueError(
            f"Could not read the --asof time {text!r}. Use e.g. 2026-10-19 15:15"
        )
    return datetime.combine(d.date(), EXPIRY_TIME)


def read_rows(path):
    """All rows of a CSV file as lists of strings (tolerates UTF-8, UTF-8 with BOM and Windows-1252)."""
    last = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.reader(f))
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"Could not decode {path}: {last}")


# --------------------------------------------------------------------------- price rule
def choose_price(ltp, bid, ask, rule="mid", max_spread=0.5):
    """
    Returns (price, source). rule='mid': use (bid+ask)/2 when both are positive and sensible, otherwise
    fall back to the last traded price. A quote whose spread is wider than max_spread x mid is rejected
    (price = nan), because a mid of a 40-wide market tells you nothing.
    """
    if rule == "mid" and bid > 0 and ask > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
        if (ask - bid) > max_spread * mid:
            return np.nan, "wide"
        return mid, "mid"
    if ltp > 0:
        return ltp, "ltp"
    return np.nan, "none"


# --------------------------------------------------------------------------- layout A: option-chain page
def find_page_header(rows):
    for i, row in enumerate(rows[:15]):
        cells = [c.strip().upper() for c in row]
        if ("STRIKE" in cells or "STRIKE PRICE" in cells) and sum(
            c in ("LTP", "BID", "ASK", "OI", "IV") for c in cells
        ) >= 2:
            return i
    return None


def parse_option_chain_page(rows, h, rule="mid", max_spread=0.5):
    header = [c.strip().upper() for c in rows[h]]
    s = header.index("STRIKE") if "STRIKE" in header else header.index("STRIKE PRICE")
    sides = {"call": range(0, s), "put": range(s + 1, len(header))}
    names = {
        "ltp": ("LTP", "LAST PRICE", "LAST TRADED PRICE"),
        "bid": ("BID", "BID PRICE"),
        "ask": ("ASK", "ASK PRICE", "OFFER"),
    }

    col = {}
    for side, idx in sides.items():
        for key, opts in names.items():
            col[(side, key)] = next((j for j in idx if header[j] in opts), None)
        if col[(side, "ltp")] is None:
            raise ValueError(
                f"Option-chain file: no LTP column on the {side.upper()}S side. Header was: {rows[h]}"
            )

    def cell(row, j):
        return parse_number(row[j]) if (j is not None and j < len(row)) else np.nan

    out = []
    for row in rows[h + 1 :]:
        if len(row) <= s:
            continue
        strike = parse_number(row[s])
        if np.isnan(strike):
            continue
        for side, label in (("call", "call"), ("put", "put")):
            ltp, bid, ask = (
                np.nan_to_num(cell(row, col[(side, k)]), nan=0.0)
                for k in ("ltp", "bid", "ask")
            )
            price, src = choose_price(ltp, bid, ask, rule, max_spread)
            out.append(
                {
                    "strike": strike,
                    "option": label,
                    "market_price": price,
                    "price_source": src,
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- layout B: bhavcopy
_BHAV = {
    "symbol": ("TckrSymb", "SYMBOL"),
    "expiry": ("XpryDt", "EXPIRY_DT"),
    "strike": ("StrkPric", "STRIKE_PR"),
    "opt": ("OptnTp", "OPTION_TYP"),
    "close": ("ClsPric", "CLOSE"),
    "under": ("UndrlygPric",),
    "date": ("TradDt", "TIMESTAMP"),
    "inst": ("FinInstrmTp", "INSTRUMENT"),
}


def looks_like_bhavcopy(columns):
    cols = set(columns)
    return (
        any(c in cols for c in _BHAV["symbol"])
        and any(c in cols for c in _BHAV["expiry"])
        and any(c in cols for c in _BHAV["opt"])
    )


def parse_bhavcopy(path, symbol, expiry=None):
    raw = pd.read_csv(path, encoding_errors="replace")
    raw.columns = [str(c).strip() for c in raw.columns]
    pick = lambda key: next((c for c in _BHAV[key] if c in raw.columns), None)
    cs, ce, ck, co, cc = (
        pick(k) for k in ("symbol", "expiry", "strike", "opt", "close")
    )
    if None in (cs, ce, ck, co, cc):
        raise ValueError(
            f"Bhavcopy: missing symbol/expiry/strike/type/close columns. Columns present: {list(raw.columns)}"
        )
    ci, cu, cd = pick("inst"), pick("under"), pick("date")

    df = raw[raw[cs].astype(str).str.strip().str.upper() == symbol.upper()].copy()
    if (
        ci is not None
    ):  # keep index options only (UDiFF: IDO, old: OPTIDX); futures and stock options are dropped
        df = df[df[ci].astype(str).str.strip().str.upper().isin(("IDO", "OPTIDX"))]
    df["_opt"] = (
        df[co].astype(str).str.strip().str.upper().map({"CE": "call", "PE": "put"})
    )
    df = df.dropna(subset=["_opt"])
    if df.empty:
        raise ValueError(f"Bhavcopy has no index option rows for symbol {symbol!r}.")
    df["_exp"] = df[ce].map(parse_date)
    expiries = sorted({d for d in df["_exp"] if d is not None})
    if expiry is None:
        if len(expiries) != 1:
            listed = ", ".join(d.strftime("%Y-%m-%d") for d in expiries)
            raise ValueError(
                f"The file holds several expiries for {symbol}: {listed}. Pass --expiry YYYY-MM-DD."
            )
        expiry = expiries[0]
    df = df[df["_exp"] == expiry]
    if df.empty:
        raise ValueError(f"No rows for expiry {expiry:%Y-%m-%d}.")

    chain = pd.DataFrame(
        {
            "strike": df[ck].map(parse_number),
            "option": df["_opt"],
            "market_price": df[cc].map(parse_number),
            "price_source": "close",
        }
    )
    chain = chain[chain.market_price > 0].dropna().reset_index(drop=True)
    meta = {"expiry": expiry}
    if cu is not None and df[cu].map(parse_number).notna().any():
        meta["spot"] = float(df[cu].map(parse_number).median())
    if cd is not None:
        d = parse_date(df[cd].iloc[0])
        if d is not None:
            meta["trade_date"] = d
    return chain, meta


# --------------------------------------------------------------------------- layout C: simple CSV
def parse_simple(path):
    try:
        from market_validation import load_chain_csv
    except ModuleNotFoundError:
        from Market_Validation import load_chain_csv
    chain = load_chain_csv(path)
    chain["price_source"] = "file"
    return chain[chain.market_price > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- loading
def load_chain(path, symbol="NIFTY", expiry=None, rule="mid", max_spread=0.5):
    """Returns (chain, meta). chain has columns strike, option, market_price, price_source."""
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty.")
    h = find_page_header(rows)
    if h is not None:
        return parse_option_chain_page(rows, h, rule, max_spread), {
            "layout": "NSE option-chain page"
        }
    header = [c.strip() for c in rows[0]]
    if looks_like_bhavcopy(header):
        chain, meta = parse_bhavcopy(path, symbol, expiry)
        meta["layout"] = "NSE F&O bhavcopy"
        return chain, meta
    try:
        return parse_simple(path), {"layout": "simple strike/option/price CSV"}
    except ValueError as e:
        raise ValueError(
            "Could not recognise this file. Expected the NSE option-chain download, an F&O bhavcopy, or a CSV "
            f"with columns strike, option, market_price.\nFirst row of the file: {rows[0][:12]}\n({e})"
        )


# --------------------------------------------------------------------------- the maths around the chain
def time_to_expiry(expiry, asof):
    """Years (calendar-day basis, /365) from the quote time to 15:30 IST on the expiry date."""
    end = datetime.combine(expiry.date(), EXPIRY_TIME)
    T = (end - asof).total_seconds() / (365.0 * 24 * 3600)
    if T <= 0:
        raise ValueError(
            f"The quotes ({asof}) are at or after expiry ({end}). Check --asof and --expiry."
        )
    return T


def forward_robust(chain, r, T, k=5):
    """
    Forward price implied by the chain: F = K + e^{rT}(C - P) at the k strikes where |C - P| is smallest
    (the ones nearest the money), then the MEDIAN, so one stale quote cannot move the answer.
    """
    calls = chain[chain.option == "call"].groupby("strike").market_price.mean()
    puts = chain[chain.option == "put"].groupby("strike").market_price.mean()
    both = calls.index.intersection(puts.index)
    if len(both) == 0:
        raise ValueError(
            "No strike has both a call and a put price, so the forward cannot be inferred."
        )
    diff = calls[both] - puts[both]
    near = diff.abs().sort_values().index[:k]
    return float(np.median([K + np.exp(r * T) * diff[K] for K in near]))


def otm_only(chain, F):
    """Calls at or above the forward, puts at or below it."""
    is_call = chain.option == "call"
    return chain[
        (is_call & (chain.strike >= F)) | (~is_call & (chain.strike <= F))
    ].reset_index(drop=True)


BANDS = (0.03, 0.07, 0.15)


def bucket_summary(detail, sigma, bands=BANDS):
    """market_validation.summarize() plus one more column: how far each option's implied vol is from the flat sigma."""
    table = summarize(detail, bands)
    edges = (0.0,) + tuple(bands) + (np.inf,)
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = detail[(detail.moneyness.abs() >= lo) & (detail.moneyness.abs() < hi)]
        if len(b):
            gaps.append(100.0 * (b.iv - sigma).abs().mean())
    table["mean abs IV gap (vol pts)"] = gaps
    return table


def run_validation(
    chain, expiry, asof, r, spot=None, min_price=1.0, use_otm_only=True, k_parity=5
):
    T = time_to_expiry(expiry, asof)
    F = forward_robust(chain, r, T, k_parity)
    S = (
        spot if spot else F
    )  # S only enters through F e^{-rT}, so the result does not depend on it
    q = (
        r - np.log(F / S) / T
    )  # carry that makes the model forward equal the market forward
    sub = chain[chain.market_price >= min_price]
    if use_otm_only:
        sub = otm_only(sub, F)
    near_money = sub[np.abs(np.log(sub.strike / F)) <= 0.05]
    if len(near_money) < 4:
        raise ValueError(
            f"Only {len(near_money)} usable options within 5% of the forward; need at least 4. "
            "Lower --min-price, or use a more liquid expiry."
        )
    sigma = calibrate_flat_vol(sub, S, r, T, q, min_price)
    detail = validate_chain(sub, S, r, T, sigma, q, min_price)
    return {
        "T": T,
        "F": F,
        "S": S,
        "q": q,
        "sigma": sigma,
        "detail": detail,
        "summary": bucket_summary(detail, sigma),
    }


# --------------------------------------------------------------------------- outputs
def make_figure(res, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = res["detail"].dropna(subset=["iv"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for opt, marker, label in (("put", "o", "OTM puts"), ("call", "s", "OTM calls")):
        s = d[d.option == opt]
        a.scatter(s.moneyness, 100 * s.iv, marker=marker, s=22, alpha=0.8, label=label)
    a.axhline(
        100 * res["sigma"],
        color="k",
        ls="--",
        lw=1,
        label=f"flat sigma {100 * res['sigma']:.1f}%",
    )
    a.set_xlabel("ln(K / F)   (0 = at the money)")
    a.set_ylabel("implied volatility (%)")
    a.set_title("Volatility smile")
    a.legend(frameon=False)
    a.grid(alpha=0.25)

    b.axhline(0, color="k", lw=0.8)
    for opt, marker in (("put", "o"), ("call", "s")):
        s = res["detail"][res["detail"].option == opt]
        b.scatter(s.moneyness, s.pct_err, marker=marker, s=22, alpha=0.8)
    b.set_xlabel("ln(K / F)")
    b.set_ylabel("model price vs market (%)")
    b.set_title("Flat-vol Black-Scholes error")
    b.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_escape(text):
    return str(text).replace("|", "\\|")


def markdown_table(df):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(md_escape(c) for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(
                f"{v:.1f}" if isinstance(v, (float, np.floating)) else md_escape(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def conclusion(summary):
    """One sentence that says what THIS table shows, whichever way it came out."""
    s = summary
    if len(s) < 2:
        return "Too few options fall in more than one bucket to say how the error changes with distance from the forward."
    near, far = (
        s["mean abs IV gap (vol pts)"].iloc[0],
        s["mean abs IV gap (vol pts)"].iloc[-1],
    )
    # Thresholds are judgement calls, kept simple on purpose: half a vol point is "a flat vol fits",
    # and "grows" means at least a quarter of a vol point worse AND at least 1.5x the near-the-money gap.
    if max(near, far) < 0.5:
        return (
            f"In this sample one flat volatility fits every bucket to within half a vol point ({near:.1f} near the forward, {far:.1f} in the farthest bucket), "
            "so no smile is visible."
        )
    if far - near >= 0.25 and far > 1.5 * near:
        return (
            f"In this sample the implied volatility sits {near:.1f} vol points from the flat fit near the forward and "
            f"{far:.1f} points away in the farthest bucket, so a single volatility fits close to the forward and degrades in the wings, "
            "consistent with a volatility smile."
        )
    return (
        f"In this sample the implied volatility is {near:.1f} vol points from the flat fit near the forward and {far:.1f} in the farthest bucket, "
        "so the error did NOT grow with distance here; with one expiry on one day, quote noise or stale prices may matter more than the smile."
    )


def readme_section(meta, res, args, n_used, fig_rel):
    d = res["detail"]
    price_note = {
        "mid": "bid/ask midpoint (last traded price where no valid quote)"
    }.get(args.price, "last traded price")
    if meta.get("layout") == "NSE F&O bhavcopy":
        price_note = "closing prices from the F&O bhavcopy"
    lines = [
        f"### Real-market validation ({args.symbol} index options)",
        "",
        f"- **Data:** {meta.get('layout', 'NSE option chain')}, expiry {args.expiry_dt:%d %b %Y}, quotes as of {args.asof_dt:%d %b %Y %H:%M} IST.",
        f"- **Prices used:** {price_note}. Out-of-the-money options only{'' if not args.all_options else ' (this run used ALL options)'}; premiums below {args.min_price:g} point(s) excluded. {n_used} options in total.",
        f"- **Inputs:** time to expiry {res['T'] * 365:.1f} days, risk-free rate {100 * args.rate:.2f}%, forward {res['F']:,.2f} inferred from put-call parity.",
        f"- **Fit:** one flat volatility of **{100 * res['sigma']:.2f}%**, fitted to options within 5% of the forward.",
        "",
        "Error of the flat-volatility Black-Scholes price against the market price, by distance from the forward, where |ln(K/F)| is that distance:",
        "",
        markdown_table(res["summary"].round(1)),
        "",
        f"![Volatility smile and pricing error]({fig_rel})",
        "",
        conclusion(res["summary"]),
        "Percentage errors look large for cheap out-of-the-money options even when the volatility gap is small, which is why the gap in volatility points is shown too. "
        "These numbers come from one expiry on one day; quote timing and stale prices on quiet strikes affect them.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line
def build_parser():
    p = argparse.ArgumentParser(
        description="Validate Black-Scholes against an NSE option chain."
    )
    p.add_argument(
        "--csv",
        required=True,
        help="path to the downloaded option chain / bhavcopy CSV",
    )
    p.add_argument(
        "--symbol",
        default="NIFTY",
        help="underlying, e.g. NIFTY or BANKNIFTY (default NIFTY)",
    )
    p.add_argument(
        "--expiry",
        help="expiry date YYYY-MM-DD (needed unless the file holds only one expiry)",
    )
    p.add_argument(
        "--asof",
        help="when the quotes were taken, IST, e.g. '2026-10-19 15:15' (bhavcopy: defaults to its trade date 15:30)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=0.065,
        help="annual risk-free rate as a decimal, e.g. 0.0625 (default 0.065 placeholder)",
    )
    p.add_argument(
        "--spot",
        type=float,
        help="optional spot price, only used to report the implied carry",
    )
    p.add_argument(
        "--price",
        choices=("mid", "ltp"),
        default="mid",
        help="option-chain page only: bid/ask midpoint or last traded (default mid)",
    )
    p.add_argument(
        "--max-spread",
        type=float,
        default=0.5,
        help="drop quotes whose bid-ask spread exceeds this fraction of the mid (default 0.5)",
    )
    p.add_argument(
        "--min-price",
        type=float,
        default=1.0,
        help="ignore premiums below this many points (default 1.0)",
    )
    p.add_argument(
        "--all-options",
        action="store_true",
        help="include in-the-money options too (default: out-of-the-money only)",
    )
    p.add_argument(
        "--out-dir",
        default="nse_output",
        help="folder for results (default nse_output)",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        want_expiry = parse_date(args.expiry) if args.expiry else None
        if args.expiry and want_expiry is None:
            raise ValueError(
                f"Could not read --expiry {args.expiry!r}. Use YYYY-MM-DD."
            )
        chain, meta = load_chain(
            args.csv, args.symbol, want_expiry, args.price, args.max_spread
        )
        expiry = meta.get("expiry") or want_expiry
        if expiry is None:
            raise ValueError(
                "Pass --expiry YYYY-MM-DD (the option-chain page file does not contain it)."
            )
        if args.asof:
            asof = parse_asof(args.asof)
        elif "trade_date" in meta:
            asof = datetime.combine(meta["trade_date"].date(), EXPIRY_TIME)
        else:
            raise ValueError(
                "Pass --asof 'YYYY-MM-DD HH:MM' (when the quotes were taken, IST)."
            )
        args.expiry_dt, args.asof_dt = expiry, asof
        spot = args.spot or meta.get("spot")

        chain = chain.dropna(subset=["market_price"])
        res = run_validation(
            chain, expiry, asof, args.rate, spot, args.min_price, not args.all_options
        )
    except (ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2

    d = res["detail"]
    os.makedirs(args.out_dir, exist_ok=True)
    d.to_csv(os.path.join(args.out_dir, "nse_validation_results.csv"), index=False)
    fig_path = os.path.join(args.out_dir, "nse_iv_smile.png")
    make_figure(
        res,
        f"{args.symbol} options, expiry {expiry:%d %b %Y}, quotes {asof:%d %b %Y %H:%M} IST",
        fig_path,
    )
    section = readme_section(meta, res, args, len(d), "images/nse_iv_smile.png")
    with open(
        os.path.join(args.out_dir, "readme_section.md"), "w", encoding="utf-8"
    ) as f:
        f.write(section)

    print(f"File layout        : {meta.get('layout')}")
    print(f"Options in file    : {len(chain)}   used after filters: {len(d)}")
    print(f"Time to expiry     : {res['T'] * 365:.2f} days")
    print(
        f"Forward (parity)   : {res['F']:,.2f}"
        + (
            f"   spot {spot:,.2f}   implied carry q = {100 * res['q']:.2f}%"
            if spot
            else ""
        )
    )
    print(f"Flat sigma (ATM)   : {100 * res['sigma']:.2f}%")
    print(
        f"Price sources      : { {k: int(v) for k, v in chain.price_source.value_counts().items()} }"
    )
    print("\nError by distance from the forward:\n")
    print(res["summary"].round(2).to_string(index=False))
    print(
        f"\nSaved in {args.out_dir}/ : nse_validation_results.csv, nse_iv_smile.png, readme_section.md"
    )
    if args.rate == 0.065 and "--rate" not in (argv if argv is not None else sys.argv):
        print(
            "\nNOTE: --rate was not given, so the 6.5% placeholder was used. Pass the current 91-day T-bill yield."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
