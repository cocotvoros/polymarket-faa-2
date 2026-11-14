# FAA Attribution Forecast

Entry for make.fun **polymarket-faa-attribution-02**: forecast the Yes
price of every market of Polymarket's FAA outage attribution family at
every minute of the 4 hours after each checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per minute, scored by mean pinball loss.

## Program

`main.py` (Python 3, standard library only) is run once per case. It
reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`), in the exact `{"forecasts":[{"id","q"}]}` shape the rules
require: 240 rows per market, five non-decreasing quantiles in [0, 1]
per row. It runs in about half a second per case and is deterministic.

## Method

These markets move like a staircase: flat for long stretches, a tick
now and then, and occasionally a jump that reprices the market for
hours. The forecaster is built in four layers around that shape:

- **Base fan** — persistence of the last price with a normal fan whose
  per-minute sigma is the average realised RMS of minute diffs over
  the last hour and the last four hours (untrimmed: trimming the
  largest diffs would delete exactly the jumps that set the tails).
  The spread grows sub-diffusively, `sigma * h^0.3`, with level
  factors [1.0, 0.85, 0, 0.85, 1.0], so on bounded prices the fan
  saturates instead of widening like `sqrt(h)`.
- **Own-history empirical offsets** — quantiles of the market's own
  realised h-minute changes over the last 24 hours at anchor horizons,
  interpolated in `h^0.3`, median removed and shrunk by 0.6, blended
  at weight 0.3. This carries each market's own asymmetry without
  letting its noisy median wander.
- **Board-pooled bucket offsets** (boards of 8+ markets only) — the
  same change quantiles pooled across every market whose price sat in
  the same level bucket, over the last 12 hours. Pooling yields
  thousands of samples per bucket, so the level-conditional shape
  (stale longshots decay; leaders behave differently) is estimated
  from data. The blend weight is discounted by the bucket's sample
  count. The final's four attribution markets are not a
  single-winner board, so this layer switches itself off there.
- **Momentum-reversion centre** — an up-move over the last hour is, on
  these boards, more often a transient repricing than news: the fan
  is bled downward by the size of the last hour's rise with a
  `(1 - exp(-h/240))` profile. Down-moves are not bled (they stick).

Finally the fan is recalibrated around its own median with per-level
scale factors fitted by direct pinball minimisation on the kit's
practice chain (the upper shoulder is compressed: on these boards the
0.75 quantile is realised far less often than a symmetric fan
implies). On the practice chain (cases p00–p04 scored against the
following case's recorded prices) this scores a mean pinball loss of
0.2988 against the house persistence baseline's 0.3925.

## License

MIT — see `LICENSE`.
