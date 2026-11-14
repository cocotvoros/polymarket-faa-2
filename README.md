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
per row.

## Method

Persistence anchored on the last price, with a predictive distribution
estimated from each market's own history in the case file:

- **Empirical view** — quantiles of realised h-minute price changes at
  anchor horizons (1…240 minutes), from two samples of the recent past
  (last 48 h and last 12 h, blended equally), shrunk by 0.7 and
  interpolated in sqrt(h) between anchors. This preserves the markets'
  real shape: long flat stretches, asymmetric moves, occasional jumps.
- **Smooth view** — a normal random walk whose per-minute sigma is the
  realised RMS of minute diffs over the last hour and last four hours,
  stabilising long horizons where empirical samples are scarce.

Final offset per quantile: `0.6 * 0.7 * empirical + 0.4 * z * sigma *
sqrt(h)`, clamped to [0, 1] around the last price.

The constants were chosen by backtesting inside the kit's practice
histories (pseudo-checkpoints scored with the contest's pinball loss),
where this configuration beat the house persistence baseline
(sigma = 0.0025 fixed) by roughly a tenth in both calm and volatile
regimes. The program is deterministic and needs no network.

## License

MIT — see `LICENSE`.
