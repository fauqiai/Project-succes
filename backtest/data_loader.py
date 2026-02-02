"""
data_loader.py
--------------
Module untuk memuat dan menyiapkan data OHLC dari CSV.
Semua komentar menggunakan ASCII agar aman di Notepad Windows.
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD CSV
# ============================================================

def load_csv(path):
    """
    Memuat file CSV OHLC.
    """
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print("Error load CSV:", e)
        return None


# ============================================================
# 2. VALIDATE AND CLEAN DATA
# ============================================================

def validate_ohlc(data):
    """
    Memastikan kolom OHLC lengkap.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        return False

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in data.columns:
            return False

    return True


def prepare_datetime(data):
    """
    Mengonversi kolom waktu menjadi datetime
    dan menjadikannya index.
    """

    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        data = data.set_index("time")

    elif "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        data = data.set_index("datetime")

    else:
        # Jika tidak ada kolom waktu, biarkan index apa adanya
        return data

    data = data.sort_index()
    data = data[~data.index.isna()]

    return data


# ============================================================
# 3. RESAMPLING (OPTIONAL)
# ============================================================

def resample_timeframe(data, timeframe="1min"):
    """
    Resample data ke timeframe lain.
    """

    try:
        new_data = data.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })

        return new_data.dropna()

    except Exception as e:
        print("Error resample:", e)
        return data


# ============================================================
# 4. FULL LOADER PIPELINE
# ============================================================

def load_and_prepare(path, timeframe=None):
    """
    Pipeline lengkap:
    1. Load CSV
    2. Validasi OHLC
    3. Konversi ke datetime
    4. Resample jika diminta
    """

    data = load_csv(path)
    if data is None:
        return None

    if not validate_ohlc(data):
        print("CSV tidak valid: kolom OHLC tidak lengkap.")
        return None

    data = prepare_datetime(data)

    if timeframe is not None:
        data = resample_timeframe(data, timeframe)

    return data


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    # Self test sederhana tanpa file eksternal
    np.random.seed(42)
    size = 50

    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=size, freq="1min"),
        "open": np.random.rand(size) * 100,
        "high": np.random.rand(size) * 100,
        "low": np.random.rand(size) * 100,
        "close": np.random.rand(size) * 100,
    })

    test_path = "test_ohlc.csv"
    df.to_csv(test_path, index=False)

    loaded = load_and_prepare(test_path, timeframe="5min")

    print("Loaded data sample:")
    print(loaded.head())

    print("data_loader.py self-test OK")

