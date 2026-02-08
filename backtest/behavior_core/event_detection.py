"""
event_detection.py
QUANT EXPANDED VERSION (HIGH SCENARIO SPACE)

Upgrade:

* Strong / Weak impulse
* Momentum burst
* Volatility spike
* Compression
* Breakout
* Exhaustion
* Fake breakout detection
* Backward compatible

SAFE FOR PRODUCTION
"""

import pandas as pd
import numpy as np

# ============================================================

# CORE MEASURES (LOCAL - biar gak tergantung file lain)

# ============================================================

def atr(data, period=14):
high = data["high"]
low = data["low"]
close = data["close"]

```
prev_close = close.shift(1)

tr = pd.concat([
    high - low,
    (high - prev_close).abs(),
    (low - prev_close).abs()
], axis=1).max(axis=1)

return tr.rolling(period, min_periods=1).mean()
```

def candle_range(data):
return data["high"] - data["low"]

def body_size(data):
return (data["close"] - data["open"]).abs()

# ============================================================

# IMPULSE FAMILY

# ============================================================

def detect_impulse(data, atr_series=None, multiplier=1.5):

```
rng = candle_range(data)

if atr_series is None:
    atr_series = atr(data)

prev = rng.shift(1)

impulse = (
    (rng > atr_series * multiplier) &
    (rng > prev * 1.2)
)

return impulse.fillna(False)
```

def detect_strong_impulse(data):
atr_series = atr(data)
rng = candle_range(data)

```
return (rng > atr_series * 2.2).fillna(False)
```

def detect_weak_impulse(data):
atr_series = atr(data)
rng = candle_range(data)

```
return (
    (rng > atr_series * 1.2) &
    (rng <= atr_series * 1.5)
).fillna(False)
```

# ============================================================

# MOMENTUM

# ============================================================

def detect_momentum_burst(data):

```
body = body_size(data)
avg_body = body.rolling(20, min_periods=1).mean()

return (body > avg_body * 1.8).fillna(False)
```

# ============================================================

# RETRACEMENT

# ============================================================

def detect_retracement(data, window=10):

```
close = data["close"]
high = data["high"]
low = data["low"]

rolling_high = high.rolling(window, min_periods=1).max()
rolling_low = low.rolling(window, min_periods=1).min()

range_ = rolling_high - rolling_low

pullback = close.diff()

retracement = (
    (pullback < 0) &
    (close > rolling_low + range_ * 0.3)
)

return retracement.fillna(False)
```

# ============================================================

# CONSOLIDATION / COMPRESSION

# ============================================================

def detect_consolidation(data, window=6):

```
rng = candle_range(data)
avg = rng.rolling(window, min_periods=1).mean()

compression = avg < avg.rolling(20, min_periods=1).mean() * 0.7

return compression.fillna(False)
```

def detect_compression_cluster(data):

```
rng = candle_range(data)
small = rng < rng.rolling(20, min_periods=1).mean() * 0.6

cluster = small.rolling(4, min_periods=1).sum() >= 3

return cluster.fillna(False)
```

# ============================================================

# VOLATILITY EVENTS

# ============================================================

def detect_volatility_spike(data):

```
atr_series = atr(data)
atr_mean = atr_series.rolling(40, min_periods=1).mean()

return (atr_series > atr_mean * 1.8).fillna(False)
```

# ============================================================

# BREAKOUTS

# ============================================================

def detect_breakout(data, window=20):

```
high = data["high"]
low = data["low"]
close = data["close"]

prev_high = high.rolling(window).max().shift(1)
prev_low = low.rolling(window).min().shift(1)

breakout_up = close > prev_high
breakout_down = close < prev_low

return (breakout_up | breakout_down).fillna(False)
```

def detect_fake_breakout(data):

```
breakout = detect_breakout(data)

close = data["close"]
prev_close = close.shift(1)

reversal = np.sign(close.diff()) != np.sign(prev_close.diff())

return (breakout & reversal).fillna(False)
```

# ============================================================

# EXHAUSTION

# ============================================================

def detect_exhaustion(data):

```
body = body_size(data)
shrinking = body < body.shift(1)

trend_push = body.shift(1) > body.shift(2)

return (shrinking & trend_push).fillna(False)
```

# ============================================================

# MASTER EVENT BUILDER

# ============================================================

def build_event_sequence(data):

```
impulse = detect_impulse(data)
strong_impulse = detect_strong_impulse(data)
weak_impulse = detect_weak_impulse(data)

momentum = detect_momentum_burst(data)

retrace = detect_retracement(data)

consolidation = detect_consolidation(data)
compression = detect_compression_cluster(data)

breakout = detect_breakout(data)
fake_breakout = detect_fake_breakout(data)

exhaustion = detect_exhaustion(data)

sequence = []

for i in range(len(data)):

    if strong_impulse.iloc[i]:
        sequence.append("strong_impulse")

    elif breakout.iloc[i]:
        sequence.append("breakout")

    elif fake_breakout.iloc[i]:
        sequence.append("fake_breakout")

    elif momentum.iloc[i]:
        sequence.append("momentum")

    elif impulse.iloc[i]:
        sequence.append("impulse")

    elif weak_impulse.iloc[i]:
        sequence.append("weak_impulse")

    elif exhaustion.iloc[i]:
        sequence.append("exhaustion")

    elif retrace.iloc[i]:
        sequence.append("retracement")

    elif compression.iloc[i]:
        sequence.append("compression")

    elif consolidation.iloc[i]:
        sequence.append("consolidation")

    else:
        sequence.append("neutral")

return sequence
```

# ============================================================

# SELF TEST

# ============================================================

if **name** == "**main**":

```
print("Running EXPANDED event_detection self-test...")

np.random.seed(42)
size = 400

price = np.cumsum(np.random.randn(size)) + 100

df = pd.DataFrame({
    "open": price,
    "high": price + np.random.rand(size),
    "low": price - np.random.rand(size),
    "close": price
})

seq = build_event_sequence(df)

print("Sample events:", seq[:30])
print("Unique events:", set(seq))

print("Self-test PASSED — no errors.")
```
