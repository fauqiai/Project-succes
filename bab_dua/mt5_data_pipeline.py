import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time


SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1

SAVE_PATH = "xauusd_live.csv"

BARS_TO_FETCH = 1500


# =====================================
# CONNECT
# =====================================

def connect_mt5():

    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    print("✅ MT5 Connected")


# =====================================
# FETCH DATA
# =====================================

def fetch_rates():

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        BARS_TO_FETCH
    )

    if rates is None:
        raise RuntimeError("Failed to fetch MT5 rates")

    df = pd.DataFrame(rates)

    df['time'] = pd.to_datetime(df['time'], unit='s')

    df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    }, inplace=True)

    return df[["time","open","high","low","close"]]


# =====================================
# SAVE
# =====================================

def save_csv(df):

    df.to_csv(SAVE_PATH, index=False)


# =====================================
# SINGLE UPDATE
# =====================================

def update_once():

    df = fetch_rates()

    save_csv(df)

    print(f"{datetime.now()} ✅ Data updated")


# =====================================
# LIVE LOOP
# =====================================

def start_pipeline():

    connect_mt5()

    print("\n🚀 STARTING DATA PIPELINE\n")

    while True:

        try:

            update_once()

            # wait until next candle
            time.sleep(60)

        except KeyboardInterrupt:

            print("Pipeline stopped.")
            mt5.shutdown()
            break

        except Exception as e:

            print("ERROR:", e)
            time.sleep(5)


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    connect_mt5()

    df = fetch_rates()

    print(df.tail())

    save_csv(df)

    print("\n✅ TEST PASSED")
