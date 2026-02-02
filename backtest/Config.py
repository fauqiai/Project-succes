"""
config.py
---------
File konfigurasi untuk Quant Behavior.
Semua parameter penting ditempatkan di sini agar mudah diubah
tanpa menyentuh kode utama.
Komentar ASCII-only agar aman disimpan di Notepad Windows.
"""

# ============================================================
# TIMEFRAME SETTINGS
# ============================================================

# Default timeframe untuk backtest (contoh: "1min" atau None)
DEFAULT_TIMEFRAME = "1min"

# ============================================================
# VOLATILITY MEASURES
# ============================================================

ATR_PERIOD = 14
STD_PERIOD = 20

# ============================================================
# IMPULSE SETTINGS
# ============================================================

IMPULSE_MULTIPLIER = 1.5      # candle > ATR * multiplier dianggap impuls
MIN_IMPULSE_BODY = 0.6        # 60 percent body dianggap impuls

# ============================================================
# RETRACEMENT SETTINGS
# ============================================================

RETRACE_WINDOW = 10
RETRACE_MIN_PERCENT = 0.2     # minimal 20 percent retrace
RETRACE_MAX_PERCENT = 0.7     # maksimal 70 percent retrace sehat

# ============================================================
# CONSOLIDATION SETTINGS
# ============================================================

CONSOLIDATION_WINDOW = 5
CONSOLIDATION_THRESHOLD = 0.3

# ============================================================
# REGIME SETTINGS
# ============================================================

TREND_MA_PERIOD = 20
TREND_STRENGTH_THRESHOLD = 0.7

RANGE_WINDOW = 20
RANGE_THRESHOLD = 0.3

SQUEEZE_LOOKBACK = 20

# ============================================================
# MICROSTRUCTURE SETTINGS
# ============================================================

SWEEP_WICK_RATIO = 0.6
IMBALANCE_BODY_THRESHOLD = 0.6
DISPLACEMENT_MULTIPLIER = 1.5
SFP_LOOKBACK = 5

# ============================================================
# CE SETTINGS
# ============================================================

FORWARD_POINTS = 10

# ============================================================
# REPORT SETTINGS
# ============================================================

REPORT_FILE = "quant_behavior_report.txt"
CE_CSV_FILE = "ce_table.csv"
STRATEGY_CSV_FILE = "best_strategies.csv"

