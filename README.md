# roblox-up-and-coming-forecaster

Forecasts Roblox's **Up-and-Coming** chart — the top 10, hour by hour, for
the 72 hours after entries close — as a probability for each game at each
place, plus a best-guess top 10 in order. Built for the make.fun
competition `roblox-up-and-coming-forecast-01`.

## What it does

`main.py` reads the chart's history (`history.ndjson`, one read per line:
rank, universe id, name and playing count per game) and `task.json`, and
writes `forecast.json` (`probabilities` + `top10`).

The chart is a **growth** ranking, not a level ranking: places churn
fast (in the practice history the top 10 changes a median of 6 places
between reads a few hours apart, and #1 changed 13 times in 45 reads),
and playing counts do not predict rank changes (measured correlation
≈ 0). So the model works on ranks alone:

1. **Calibrate from the history.** For every consecutive pair of reads,
   the change in log-rank per hour and per square-root hour is pooled by
   starting rank band, giving each band a drift (mean log-rank change per
   hour) and a diffusion (its standard deviation).
2. **Simulate.** Each game on the chart as the history ends follows a
   log-rank random walk with its band's drift and diffusion, plus its own
   recent log-rank velocity (shrunk — measured velocity autocorrelation is
   only ~0.21). Games whose details carry a `created` date and which age
   past the chart's 28-day window are dropped from their exit hour on.
3. **Count.** Over 1000 runs × 72 simulated hours, the frequency with
   which each game lands at each place is its probability at a random
   hour of the scored window. Each place leaves 25% unnamed (room for
   games that enter the chart during the window, which no history can
   name). The top 10 is the order by expected simulated rank.

## Running it

```sh
# locally (the grader's container leaves both unset and uses /input, /output)
mkdir -p out
MAKEFUN_INPUT_DIR=kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic (fixed
seed — two runs on the same input produce byte-identical output), ~17 s
for the practice history, far inside the 300 s limit.

## Files

- `main.py` — the forecaster.
- `LICENSE` — MIT.
- `README.md` — this file.
- `SESSION-LOG.md` — the build log (steps, decisions, measured scores),
  the transcript of the agent sessions that built this entry.

## Measured performance

On the practice forecast (history: the archive's 34 reads to
2026-11-07T20:53:19Z; scored on the 10 archive reads of
2026-11-08 → 2026-11-10), scored with a local scorer that reproduces
the house baseline's published practice score exactly (-43.512039), this
forecaster scores **-33.988381** (typical probability per place 0.0334
vs the baseline's 0.0129; 7.3 top-10 hits per read vs 6.4; 60.5
agreement points vs 53.5). See `SESSION-LOG.md` for the validation and
tuning record.
