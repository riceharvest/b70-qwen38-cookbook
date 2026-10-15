#!/usr/bin/env python3
"""Honest batch-throughput numbers for run_tput.sh results.

gen = c / median TPOT assumes all c requests decode at the same time. When memory limits how many run (MTP's extra state
slots, fp32 GDN state), requests queue, TPOT stays low, and c / TPOT overstates throughput many times over. This reads the
engine's 10 s log lines ("Avg generation throughput", "Running: N reqs") inside each bench's time window and reports:
  running   mean / max number of running requests while the bench ran (vs c)
  eng_gen   the engine's own generation throughput (mean of its 10 s samples while requests were running)
  gen_run   mean_running / median TPOT (what c / TPOT means when c is replaced by what actually ran)
usage: tput_metric.py RESULT.json [...]   (vllm bench serve --save-result files; the server log `docker logs <server>` saved as
       serve_<label>.log next to them, where label = result file name up to -txt-/-rnd-)
"""
import datetime as dt, json, os, re, sys

LINE = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) .*Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")

def samples(log, year):
    out = []
    for l in open(log, errors="replace"):
        m = LINE.search(l)
        if m:
            mo, d, h, mi, s = map(int, m.groups()[:5])
            out.append((dt.datetime(year, mo, d, h, mi, s), float(m.group(6)), int(m.group(7)), int(m.group(8))))
    return out

print(f"{'result':34s} {'c':>3s} {'c/TPOT':>8s} {'run mean/max':>13s} {'wait max':>8s} {'eng_gen':>8s} {'gen_run':>8s} {'out':>7s}  note")
for path in sys.argv[1:]:
    d = json.load(open(path)); base = os.path.basename(path)[:-5]
    label = re.split(r"-(txt|rnd)-c\d+$", base)[0]
    log = os.path.join(os.path.dirname(path), f"serve_{label}.log")
    c = int(d.get("max_concurrency") or 0); tp = d.get("median_tpot_ms") or 0
    end = dt.datetime.strptime(d["date"], "%Y%m%d-%H%M%S"); start = end - dt.timedelta(seconds=float(d["duration"]))
    note = ""
    if not os.path.exists(log):
        print(f"{base:34s} {c:3d} {1000 * c / tp if tp else 0:8.1f} {'':>13s} {'':>8s} {'':>8s} {'':>8s} {d['output_throughput']:7.1f}  no serve log {os.path.basename(log)}")
        continue
    # the bench saves its result when it ends; allow one log interval of slack on each side
    win = [x for x in samples(log, end.year) if start - dt.timedelta(seconds=5) <= x[0] <= end + dt.timedelta(seconds=10)]
    busy = [x for x in win if x[2] > 0]
    if not busy:
        note = "no engine samples in the bench window (clock offset?)"; rm = rx = wx = 0; eg = 0.0
    else:
        rm = sum(x[2] for x in busy) / len(busy); rx = max(x[2] for x in busy); wx = max(x[3] for x in win)
        eg = sum(x[1] for x in busy) / len(busy)
        if rx < c: note = f"ONLY {rx} OF {c} RAN AT ONCE: c/TPOT is not a throughput"
    gr = 1000 * rm / tp if tp else 0
    print(f"{base:34s} {c:3d} {1000 * c / tp if tp else 0:8.1f} {rm:6.1f}/{rx:<6d} {wx:8d} {eg:8.1f} {gr:8.1f} {d['output_throughput']:7.1f}  {note}")
