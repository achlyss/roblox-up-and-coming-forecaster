#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun competition
roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (one chart read per line: rank, universe_id,
name, playing per game) and /input/task.json, writes
/output/forecast.json: for each of the top 10 places, a probability for
each game named being at that place, plus the best-guess top 10 in order.

Method
------
The chart is a growth ranking, not a level ranking: a game's place is
driven by how fast it is rising, and places churn fast (in the practice
history the median read-to-read change of the top 10 is 6 places, and
the #1 changed 13 times in 45 reads). Playing counts do not predict
rank changes (measured correlation ~0), so the model works on ranks
alone: a Monte Carlo simulation of log-rank paths.

For every game on the chart as the history ends, its log-rank follows a
random walk with a drift and a diffusion that both depend on its current
rank band, calibrated from the history itself (all consecutive-read
transitions, pooled by starting rank band):
  * drift  = the band's mean log-rank change per hour, plus a shrunk
    recent velocity (the game's own log-rank slope over its last few
    reads, shrunk by MOMENTUM_SHRINK, because measured velocity
    autocorrelation is only ~0.21);
  * diffusion = the band's standard deviation of log-rank change per
    square-root hour.
Each simulated hour the games are ordered by simulated log-rank and the
top 10 is recorded; over RUNS x 72 hours the frequencies give, for each
game and place, the probability it holds that place at a random hour of
the scored window. Probabilities are capped so each place leaves
UNNAMED_SHARE unnamed (room for games that enter the chart during the
window, which no history can name), tiny probabilities are dropped, and
the top 10 is the order by expected simulated rank.

Games whose details carry a `created` date (house-recorder reads) and
which age past the chart's 28-day window before or during the scored
window are dropped / dropped from their exit hour on: only games
published in the last 28 days are on the chart.

Standard library only, no network, deterministic (fixed seed): two runs
on the same input produce byte-identical output.
"""

import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
RUNS = 1000
HOURS = 72
PLACES = 10
# The scored window is 72 whole hours; the simulation covers each of them.
MOMENTUM_SHRINK = 0.25
# Multiplier on the calibrated diffusion (1.0 = as measured).
DIFF_SCALE = 1.0
# Recent velocity is the log-rank slope over the last few reads.
VELOCITY_READS = 4
# Leave this much of every place unnamed (new entrants the history
# cannot name) and drop probabilities below this.
UNNAMED_SHARE = 0.25
MIN_P = 0.003
# Games absent from the last read but seen recently may re-enter; their
# last seen rank is penalised by this many places.
REENTRY_PENALTY = 15
REENTRY_READS = 5
# The chart lists games published in the last 28 days.
CHART_MAX_AGE_DAYS = 28

# Rank bands for the drift/diffusion calibration (by starting rank).
BANDS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9)]


def _parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-07T20:53:19.000Z
    return int(
        __import__("datetime").datetime.fromisoformat(
            s.replace("Z", "+00:00")
        ).timestamp()
    )


def _band(rank):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= rank <= hi:
            return i
    return len(BANDS) - 1


def load_history(input_dir):
    task = json.load(open(os.path.join(input_dir, "task.json")))
    reads = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Skip reads that did not return a chart (a failed read
            # carries no up_and_coming rows).
            if r.get("up_and_coming"):
                reads.append(r)
    if not reads:
        raise ValueError("no valid chart reads in history.ndjson")
    return task, reads


def calibrate(reads):
    """Drift (per hour) and diffusion (per sqrt-hour) of log-rank change,
    pooled by starting rank band, over all consecutive-read transitions.
    Falls back to global values when a band has too few transitions."""
    per_band = {i: [] for i in range(len(BANDS))}
    all_d = []
    for a, b in zip(reads, reads[1:]):
        dt = (_parse_time(b["at"]) - _parse_time(a["at"])) / 3600.0
        if dt <= 0:
            continue
        ra = {g["universe_id"]: g["rank"] for g in a["up_and_coming"]}
        rb = {g["universe_id"]: g["rank"] for g in b["up_and_coming"]}
        for u in set(ra) & set(rb):
            d = math.log(rb[u]) - math.log(ra[u])
            per_band[_band(ra[u])].append((d, dt))
            all_d.append((d, dt))

    def stats(vals):
        if not vals:
            return 0.0, 0.05
        n = len(vals)
        drift = sum(d / dt for d, dt in vals) / n
        diffs = [d / math.sqrt(dt) for d, dt in vals]
        mean = sum(diffs) / n
        var = sum((x - mean) ** 2 for x in diffs) / n
        return drift, math.sqrt(max(var, 1e-6))

    global_drift, global_diff = stats(all_d)
    params = []
    for i in range(len(BANDS)):
        vals = per_band[i]
        if len(vals) < 30:
            params.append((global_drift, global_diff))
        else:
            params.append(stats(vals))
    return params


def recent_velocity(series):
    """Shrunk log-rank slope per hour over the game's last few reads.
    series: list of (timestamp_seconds, rank), oldest first."""
    if len(series) < 2:
        return 0.0
    pts = series[-VELOCITY_READS:]
    t0 = pts[0][0]
    xs = [(t - t0) / 3600.0 for t, _ in pts]
    ys = [math.log(r) for _, r in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return MOMENTUM_SHRINK * slope


def build_forecast(input_dir):
    task, reads = load_history(input_dir)
    places = int(task.get("places", PLACES))
    window_from = _parse_time(task["window"]["from"])
    window_until = _parse_time(task["window"]["until"])
    hours = int(round((window_until - window_from) / 3600.0))
    hours = max(1, min(hours, 24 * 7))

    params = calibrate(reads)
    last = reads[-1]
    last_at = _parse_time(last["at"])

    # Per-game rank series and created dates (when details carry them).
    series = {}
    created = {}
    for r in reads:
        t = _parse_time(r["at"])
        for g in r["up_and_coming"]:
            series.setdefault(g["universe_id"], []).append((t, g["rank"]))
        details = r.get("details") or {}
        if isinstance(details, dict):
            for u, d in details.items():
                if d and d.get("created"):
                    created[int(u)] = d["created"]

    max_age = CHART_MAX_AGE_DAYS * 86400
    games = {}  # universe_id -> (log_rank0, velocity, exit_hour or None)
    for g in last["up_and_coming"]:
        u = g["universe_id"]
        exit_hour = None
        if u in created:
            born = _parse_time(created[u])
            age_at_close = window_from - born
            if age_at_close >= max_age:
                continue  # already too old to be on the chart
            exit_hour = (born + max_age - window_from) // 3600
            if exit_hour < 0:
                exit_hour = 0
        games[u] = (math.log(g["rank"]), recent_velocity(series[u]),
                    exit_hour)
    # Recent drop-outs may re-enter.
    last_ids = set(games)
    for idx in range(max(0, len(reads) - REENTRY_READS), len(reads) - 1):
        for g in reads[idx]["up_and_coming"]:
            u = g["universe_id"]
            if u in last_ids or u in games:
                continue
            if u in created:
                born = _parse_time(created[u])
                if window_from - born >= max_age:
                    continue
            games[u] = (math.log(g["rank"] + REENTRY_PENALTY),
                        recent_velocity(series[u]), None)

    ids = sorted(games)
    log_rank0 = [games[u][0] for u in ids]
    velocity = [games[u][1] for u in ids]
    exit_hour = [games[u][2] for u in ids]

    rng = random.Random(SEED)
    counts = [[0] * places for _ in ids]
    rank_sum = [0.0] * len(ids)
    rank_n = [0] * len(ids)
    for _ in range(RUNS):
        keys = list(log_rank0)
        for h in range(hours):
            for i in range(len(ids)):
                if exit_hour[i] is not None and h >= exit_hour[i]:
                    keys[i] = float("inf")
                    continue
                band = _band(max(1, int(round(math.exp(min(keys[i], 10.0))))))
                drift, diff = params[band]
                keys[i] += drift + velocity[i] \
                    + diff * DIFF_SCALE * rng.gauss(0.0, 1.0)
            order = sorted(range(len(ids)), key=lambda i: keys[i])
            for place in range(min(places, len(order))):
                counts[order[place]][place] += 1
            for pos, i in enumerate(order):
                if pos < 200:
                    rank_sum[i] += pos + 1
                    rank_n[i] += 1

    total = RUNS * hours
    cap = 1.0 - UNNAMED_SHARE
    probabilities = []
    for place in range(places):
        rows = []
        s = 0.0
        for i, u in enumerate(ids):
            p = counts[i][place] / total
            if p >= MIN_P:
                rows.append([u, p])
                s += p
        if s > cap and s > 0:
            scale = cap / s
            rows = [[u, p * scale] for u, p in rows]
        for u, p in rows:
            probabilities.append({"universe_id": u, "place": place + 1,
                                  "p": round(p, 6)})

    expected_rank = []
    for i, u in enumerate(ids):
        if rank_n[i]:
            expected_rank.append((rank_sum[i] / rank_n[i], u))
    expected_rank.sort()
    top10 = [u for _, u in expected_rank[:places]]
    return {"probabilities": probabilities, "top10": top10}


def main():
    forecast = build_forecast(INPUT_DIR)
    task, _ = load_history(INPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    print("wrote", out_path,
          len(forecast["probabilities"]), "rows, top10:",
          forecast["top10"])


if __name__ == "__main__":
    main()
