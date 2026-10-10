"""""
    python test_connection.py --broker oanda
    python test_connection.py --broker ibkr
    python test_connection.py --broker oanda --order-test    # places and immediately closes a 1-unit trade
"""
import argparse
import sys

from Broker_base import BrokerError
from Config import ConfigError, Settings, load_dotenv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", choices=["oanda", "ibkr"], required=True)
    ap.add_argument("--order-test", action="store_true",
                    help="buy 1 unit with a stop/target, then close it (paper accounts only)")
    args = ap.parse_args()
    load_dotenv()
    try:
        s = Settings.from_env()
        s.broker = args.broker
        s.validate()
    except ConfigError as e:
        sys.exit(f"configuration error: {e}\nFill in your .env (see env.example).")
    from bot import build_broker
    try:
        b = build_broker(s)
        print(f"[1/4] connected to {b.name}")
        print(f"[2/4] account equity : {b.equity():,.2f}")
        bid, ask = b.quote(s.instrument)
        print(f"[3/4] {s.instrument} quote  : bid {bid} / ask {ask}  (spread {ask - bid:.5f})")
        c = b.candles(s.instrument, s.granularity, 5)
        print(f"[4/4] last candles ({len(c)}):")
        print(c.tail(3).to_string(index=False))
        print(f"current position     : {b.position(s.instrument)} units")
        if args.order_test:
            if b.position(s.instrument) != 0:
                sys.exit("refusing --order-test: you already hold a position in this instrument")
            ref = ask
            f = b.market_order(s.instrument, 1, stop_loss=round(ref * 0.99, 5),
                               take_profit=round(ref * 1.01, 5), tag="connection-test")
            print(f"order test: bought 1 unit @ {f.price}  (order {f.order_id})")
            print(f"position now: {b.position(s.instrument)}")
            c2 = b.close_position(s.instrument)
            print(f"closed @ {c2.price}, realised P&L {c2.pl:+.5f}; position now {b.position(s.instrument)}")
        print("\nOK - connection works.")
    except BrokerError as e:
        sys.exit(f"\nFAILED: {e}")


if __name__ == "__main__":
    main()