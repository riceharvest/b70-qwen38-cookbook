"""bench/tput_metric.py on synthetic `vllm bench serve` results and server logs: it must call out runs where fewer than c
requests ran at once, since c / TPOT then overstates throughput."""
import json
import pathlib
import subprocess
import sys

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"


def log_line(sec, gen, running, waiting=0):
    return (f"INFO 10-24 09:{sec // 60:02d}:{sec % 60:02d} [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
            f"Avg generation throughput: {gen} tokens/s, Running: {running} reqs, Waiting: {waiting} reqs, GPU KV cache usage: 5%\n")


def make(tmp_path, name, c, running, tpot=40.0, with_log=True, offset=0):
    # the bench ended at 09:03:00 and lasted 120 s; engine samples every 10 s, running[i] requests in sample i
    result = {"max_concurrency": c, "median_tpot_ms": tpot, "output_throughput": 1000.0, "duration": 120.0, "date": "20261024-090300"}
    (tmp_path / f"{name}.json").write_text(json.dumps(result))
    if with_log:
        lines = [log_line(offset + 70 + 10 * i, 100.0 * r, r, 2 if r < c else 0) for i, r in enumerate(running)]
        (tmp_path / "serve_run.log").write_text("".join(lines))
    return tmp_path / f"{name}.json"


def metric(path):
    r = subprocess.run([sys.executable, str(BENCH / "tput_metric.py"), str(path)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()[1]


def test_all_requests_ran_at_once(tmp_path):
    line = metric(make(tmp_path, "run-rnd-c64", 64, [64] * 5))
    assert "ONLY" not in line
    assert " 64.0/64 " in line  # mean/max running
    assert "1600.0" in line  # c / TPOT = 64 / 40 ms
    assert "6400.0" in line  # engine generation: mean of the busy samples


def test_queued_requests_are_called_out(tmp_path):
    line = metric(make(tmp_path, "run-rnd-c64", 64, [30, 32, 32, 28, 0]))
    assert "ONLY 32 OF 64 RAN AT ONCE" in line
    assert "1600.0" in line  # c / TPOT still printed, but flagged
    assert " 30.5/32 " in line  # idle sample (running = 0) is excluded from the mean
    assert "762.5" in line  # gen_run = mean running / TPOT = 30.5 / 40 ms


def test_missing_log_is_reported(tmp_path):
    assert "no serve log serve_run.log" in metric(make(tmp_path, "run-rnd-c8", 8, [], with_log=False))


def test_clock_offset_is_reported(tmp_path):
    # the log timestamps are rewritten to 05:xx, hours before the bench window (a clock offset between host and container)
    path = make(tmp_path, "run-rnd-c8", 8, [8] * 5)
    log = (tmp_path / "serve_run.log").read_text().replace("09:", "05:")
    (tmp_path / "serve_run.log").write_text(log)
    assert "no engine samples in the bench window" in metric(path)
