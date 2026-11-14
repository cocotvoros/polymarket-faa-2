#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Polymarket FAA attribution forecast (make.fun polymarket-faa-attribution-02).

For each market in a case, forecast the Yes price at every minute of the
next `horizon_minutes` minutes as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) of that minute's price.

These markets move like a staircase: flat for long stretches, a tick now
and then, and occasionally a jump that reprices the market for hours.
The forecaster is built around that shape, in four layers:

* Base fan - persistence of the last price, with a normal fan whose
  per-minute sigma is the average of the realised RMS of minute diffs
  over the last hour and the last four hours (untrimmed: trimming the
  largest diffs would delete exactly the jumps that set the tails).
  The spread grows sub-diffusively, as sigma * h^0.3, with level
  factors [1.0, 0.85, 0, 0.85, 1.0]: on bounded prices the fan must
  saturate, not keep widening like sqrt(h).
* Own-history empirical offsets - the quantiles of the market's own
  realised h-minute changes over the last 24 hours, taken at anchor
  horizons and interpolated in h^0.3, median removed and shrunk by
  0.6, blended at weight 0.3 into the fan.  This carries the market's
  own asymmetry (its jumps are rarely symmetric) without letting its
  noisy median wander.
* Board-pooled bucket offsets (only on boards of 8+ markets) - the
  same h-minute change quantiles, but pooled across every market on
  the board whose price sat in the same level bucket, over the last
  12 hours.  Pooling gives thousands of samples per bucket, so the
  level-conditional shape (stale longshots decay, leaders behave
  differently from longshots) is estimated from data, not assumed.
  The blend weight is discounted by the bucket's sample count.
* Momentum-reversion centre - an up-move over the last hour is, on
  these boards, more often a transient repricing than news: the whole
  fan is bled downward by the size of the last hour's rise, with a
  (1 - exp(-h/240)) profile.  Down-moves are not bled (they stick).

Finally the fan is recalibrated around its own median with per-level
scale factors fitted by direct pinball minimisation on the kit's
practice chain (wider early/late horizon bands, a compressed upper
shoulder - on these boards the 0.75 quantile is realised far less
often than a symmetric fan implies).  Small boards (like the final's
four attribution markets, which are not a single-winner board) skip
the pooling layer and use near-neutral recalibration instead.

The program reads every *.in file in $MAKEFUN_INPUT_DIR (default
/input), writes one *.out per case to $MAKEFUN_OUTPUT_DIR (default
/output), uses only the standard library, needs no network, and is
fully deterministic.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
K_LEVEL = (1.0, 0.85, 0.0, 0.85, 1.0)
ALPHA = 0.3

ANCHORS = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240)
# Own-history empirical layer.
EMP_WINDOW = 1440
W_EMP = 0.3
EMP_SHRINK = 0.6
# Sigma: untrimmed RMS of minute diffs, windows averaged equally.
SIGMA_WINDOWS = (60, 240)
SIGMA_FLOOR = 0.0001
SIGMA_CAP = 0.02
# Board-pooled bucket layer.
BUCKET_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 1.01)
BUCKET_WINDOW = 720
W_BUCKET = 0.7
BUCKET_SHRINK_N = 2000.0
BOARD_MIN_MARKETS = 8
# Momentum reversion centre.
REV_K_UP = 1.0
REV_TAU = 240.0
# Recalibration around the median: per-tau scale, per-horizon-band scale.
RECAL_BOARD_M = (0.88, 0.95, 1.0, 0.64, 0.92)
RECAL_BOARD_G = (1.08, 0.93, 1.08)  # h 1-60, 61-180, 181-240
RECAL_SMALL_M = (0.95, 1.0, 1.0, 0.90, 0.95)
RECAL_SMALL_G = (1.0, 1.0, 1.0)


def quantile_sorted(vals, tau):
    """Linear-interpolation quantile of an ascending list."""
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bucket_of(x):
    for b in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[b] <= x < BUCKET_EDGES[b + 1]:
            return b
    return len(BUCKET_EDGES) - 2


def interp_anchors(anchor_h, anchor_q, horizon):
    """Interpolate anchor quantile curves in h**ALPHA space.

    anchor_h: increasing list of horizons; anchor_q: per-anchor list of
    five offsets.  Returns a list of `horizon` rows of five offsets.
    """
    ga = [h ** ALPHA for h in anchor_h]
    rows = []
    ai = 0
    for h in range(1, horizon + 1):
        g = h ** ALPHA
        if g <= ga[0]:
            rows.append(list(anchor_q[0]))
            continue
        if g >= ga[-1]:
            rows.append(list(anchor_q[-1]))
            continue
        while ai + 1 < len(ga) - 1 and ga[ai + 1] < g:
            ai += 1
        frac = (g - ga[ai]) / (ga[ai + 1] - ga[ai])
        rows.append([a * (1.0 - frac) + b * frac
                     for a, b in zip(anchor_q[ai], anchor_q[ai + 1])])
    return rows


def sigma_estimate(prices):
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    if not diffs:
        return SIGMA_FLOOR
    acc = 0.0
    for w in SIGMA_WINDOWS:
        seg = diffs[-w:] if w < len(diffs) else diffs
        acc += math.sqrt(sum(d * d for d in seg) / len(seg))
    sig = acc / len(SIGMA_WINDOWS)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sig))


def own_empirical_rows(prices, horizon):
    """Own-history empirical offsets (median removed), or None."""
    n = len(prices)
    start = max(0, n - 1 - EMP_WINDOW)
    anchor_h, anchor_q = [], []
    for h in ANCHORS:
        if h > horizon:
            break
        end = n - h
        if end - start < 8:
            continue
        changes = sorted(prices[t + h] - prices[t] for t in range(start, end))
        anchor_h.append(h)
        anchor_q.append([quantile_sorted(changes, t) for t in TAUS])
    if not anchor_h:
        return None
    rows = interp_anchors(anchor_h, anchor_q, horizon)
    out = []
    for row in rows:
        med = row[2]
        out.append([(v - med) * EMP_SHRINK for v in row])
    return out


def bucket_tables(all_prices, horizon):
    """Pooled per-bucket anchor quantile tables for the whole board."""
    nb = len(BUCKET_EDGES) - 1
    per_bucket = {b: {} for b in range(nb)}  # b -> anchor -> [changes]
    for prices in all_prices:
        n = len(prices)
        start = max(0, n - 1 - BUCKET_WINDOW)
        for h in ANCHORS:
            if h > horizon:
                break
            end = n - h
            if end - start < 8:
                continue
            for t in range(start, end):
                b = bucket_of(prices[t])
                per_bucket[b].setdefault(h, []).append(prices[t + h] - prices[t])
    tables = {}
    for b in range(nb):
        anchor_h, anchor_q, counts = [], [], []
        for h in ANCHORS:
            ch = per_bucket[b].get(h)
            if ch:
                ch = sorted(ch)
                anchor_h.append(h)
                anchor_q.append([quantile_sorted(ch, t) for t in TAUS])
                counts.append(len(ch))
        if anchor_h:
            counts.sort()
            nmed = counts[len(counts) // 2]
            tables[b] = (anchor_h, anchor_q, nmed)
    return tables


def forecast_case(all_prices, horizon):
    """Forecast every market of one case; returns one row-block each."""
    board = len(all_prices) >= BOARD_MIN_MARKETS
    tables = bucket_tables(all_prices, horizon) if board else {}
    m_recal = RECAL_BOARD_M if board else RECAL_SMALL_M
    g_recal = RECAL_BOARD_G if board else RECAL_SMALL_G
    results = []
    for prices in all_prices:
        x = prices[-1]
        sig = sigma_estimate(prices)
        # Base fan offsets.
        rows = []
        for h in range(1, horizon + 1):
            s = sig * (h ** ALPHA)
            rows.append([k * z * s for k, z in zip(K_LEVEL, Z)])
        # Own-history empirical blend.
        emp = own_empirical_rows(prices, horizon)
        if emp is not None:
            for i in range(horizon):
                for j in range(5):
                    rows[i][j] = (1.0 - W_EMP) * rows[i][j] + W_EMP * emp[i][j]
        # Board-pooled bucket blend (offsets are around the last price).
        if board:
            b = bucket_of(x)
            if b in tables:
                anchor_h, anchor_q, nmed = tables[b]
                brows = interp_anchors(anchor_h, anchor_q, horizon)
                weff = W_BUCKET * (nmed / (nmed + BUCKET_SHRINK_N))
                for i in range(horizon):
                    for j in range(5):
                        rows[i][j] = (1.0 - weff) * rows[i][j] + weff * brows[i][j]
        # Momentum-reversion centre.
        mom60 = prices[-1] - prices[-61] if len(prices) > 60 else 0.0
        rev = REV_K_UP * max(mom60, 0.0)
        out = []
        for i in range(horizon):
            h = i + 1
            c = x - rev * (1.0 - math.exp(-h / REV_TAU))
            row = sorted(c + off for off in rows[i])
            out.append([min(1.0, max(0.0, v)) for v in row])
        # Recalibration around the median.
        final = []
        for i in range(horizon):
            h = i + 1
            g = g_recal[0] if h <= 60 else (g_recal[1] if h <= 180 else g_recal[2])
            med = out[i][2]
            row = sorted(med + (v - med) * m_recal[j] * g
                         for j, v in enumerate(out[i]))
            final.append([round(min(1.0, max(0.0, v)) * 1e6) / 1e6 + 0.0
                          for v in row])
        results.append(final)
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in files:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case["quantiles"]) != list(TAUS):
            raise ValueError(f"{name} asks for quantiles {case['quantiles']}")
        horizon = int(case["horizon_minutes"])
        all_rows = forecast_case([m["prices"] for m in case["markets"]], horizon)
        forecasts = [
            {"id": m["id"], "q": rows}
            for m, rows in zip(case["markets"], all_rows)
        ]
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
