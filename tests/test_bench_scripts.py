"""bench/ scripts against a fake vLLM server on localhost (CPU only): checks the request shapes the scripts send and the
numbers they derive (tokens per MTP step, acceptance per position, prefill rate, arithmetic score)."""
import http.server
import json
import pathlib
import re
import shutil
import subprocess
import sys
import threading

import pytest

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
BENCH_RUN = BENCH
COMPLETION_TOKENS, DRAFTS_PER_REQ, ACCEPTED = 40, 10, [9, 7, 5, 3]


class Fake(http.server.BaseHTTPRequestHandler):
    log = []          # (path, body) of every POST
    counters = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": [0, 0, 0, 0]}
    wrong_arith = False

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        c = Fake.counters
        if self.path == "/v1/models":
            return self._send(200, json.dumps({"data": [{"id": "fake"}]}))
        if self.path == "/metrics":
            lines = [f'vllm:spec_decode_num_drafts_total{{engine="0"}} {c["drafts"]}',
                     f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {c["draft_tokens"]}',
                     f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {c["accepted"]}',
                     'vllm:num_requests_running{engine="0"} 0.0']
            lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}} {v}'
                      for i, v in enumerate(c["pos"])]
            return self._send(200, "\n".join(lines) + "\n", "text/plain")
        self._send(404, "{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.log.append((self.path, body))
        prompt = json.dumps(body.get("messages") or body.get("prompt"))
        c = Fake.counters
        c["drafts"] += DRAFTS_PER_REQ; c["draft_tokens"] += 4 * DRAFTS_PER_REQ; c["accepted"] += sum(ACCEPTED)
        c["pos"] = [p + a for p, a in zip(c["pos"], ACCEPTED)]
        if self.path == "/v1/chat/completions" and "evaluate to in Python" in prompt:
            expr = re.search(r"`([^`]+)`", body["messages"][0]["content"]).group(1)
            ans = str(eval(expr))  # the script's own generator made it; evaluating it is the reference answer
            if Fake.wrong_arith:
                ans = str(int(ans) + 1)
            return self._send(200, json.dumps({"choices": [{"message": {"content": ans}}]}))
        usage = {"prompt_tokens": 1234, "completion_tokens": COMPLETION_TOKENS}
        chat = self.path == "/v1/chat/completions"
        chunk = lambda t: {"choices": [{"delta": {"content": t}} if chat else {"text": t}]}
        out = "".join(f"data: {json.dumps(chunk('x'))}\n\n" for _ in range(3))
        out += f"data: {json.dumps({'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n"
        self._send(200, out, "text/event-stream")


@pytest.fixture
def server():
    Fake.log, Fake.wrong_arith = [], False
    Fake.counters = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": [0, 0, 0, 0]}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(scope="session")
def bench_dir(tmp_path_factory):
    """A copy of bench/ with a small synthetic corpus: the real bench/corpus.txt is gitignored (10 MB), so a fresh
    checkout (CI) doesn't have it, and the tests must not depend on it."""
    d = tmp_path_factory.mktemp("bench")
    for f in BENCH.glob("*.py"):
        shutil.copy2(f, d / f.name)
    (d / "corpus.txt").write_text("".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(20000)))
    return d


@pytest.fixture(autouse=True)
def _use_bench_dir(bench_dir):
    global BENCH_RUN
    BENCH_RUN = bench_dir


def run(script, *args):
    r = subprocess.run([sys.executable, str(BENCH_RUN / script), *map(str, args)], capture_output=True, text=True,
                       timeout=120, cwd=BENCH_RUN, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_harness_decode_records(server, tmp_path):
    out = run("harness.py", "--port", server, "--label", "t", "--decode", "--reps", 1, "--prefill", "", "--out", tmp_path / "r.jsonl")
    recs = [json.loads(line) for line in out.splitlines()]
    assert [r["case"] for r in recs] == ["code", "json", "prose", "think"]
    for r in recs:
        assert r["kind"] == "decode" and r["label"] == "t" and r["ctoks"] == COMPLETION_TOKENS
        assert r["tok_per_step"] == COMPLETION_TOKENS / DRAFTS_PER_REQ  # tokens per verify step, from the metric deltas
        assert r["accept_pos"] == [a / DRAFTS_PER_REQ for a in ACCEPTED]  # acceptance rate at each draft position
        assert r["decode_tps"] > 0 and r["ttft"] >= 0
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 4  # also appended to --out
    chats = [b for p, b in Fake.log if p == "/v1/chat/completions"]
    assert [b["chat_template_kwargs"]["enable_thinking"] for b in chats] == [False, False, False, True]
    assert all(b["temperature"] == 0 and b["max_tokens"] == 512 for b in chats)


def test_harness_sampling_flag_changes_request(server, tmp_path):
    run("harness.py", "--port", server, "--label", "t", "--decode", "--sample", "--reps", 2, "--prefill", "", "--out", tmp_path / "r.jsonl")
    chats = [b for p, b in Fake.log if p == "/v1/chat/completions"]
    assert all((b["temperature"], b["top_p"], b["top_k"]) == (1.0, 0.95, 20) for b in chats)
    assert sorted({b["seed"] for b in chats}) == [1, 2]  # one seed per rep


def test_harness_prefill_defeats_the_prefix_cache(server, tmp_path):
    out = run("harness.py", "--port", server, "--label", "t", "--prefill", "1000", "--reps", 2, "--out", tmp_path / "r.jsonl")
    recs = [json.loads(line) for line in out.splitlines()]
    assert len(recs) == 2 and all(r["kind"] == "prefill" and r["ptoks"] == 1234 for r in recs)
    assert all(r["prefill_tps"] > 0 for r in recs)
    prompts = [b["prompt"] for p, b in Fake.log if p == "/v1/completions"]
    assert len(set(prompts)) == 2  # unique nonce + different corpus slice per rep: no cache hit
    assert all(re.match(r"\[[0-9a-f]{32}\]\n", p) for p in prompts)


def test_arith_battery_scores_a_correct_model(server):
    assert "arith battery 100/100" in run("arith_battery.py", server)
    assert "arith battery 20/20" in run("arith_battery.py", server, 20)


def test_arith_battery_detects_a_wrong_model(server):
    Fake.wrong_arith = True
    out = run("arith_battery.py", server, 10, "-v")
    assert "arith battery 0/10" in out and out.count("BAD ") == 10


def test_arith_questions_are_seeded_and_stable(server):
    run("arith_battery.py", server, 30)
    first = [b["messages"][0]["content"] for _, b in Fake.log]
    Fake.log.clear()
    run("arith_battery.py", server, 30)
    assert first == [b["messages"][0]["content"] for _, b in Fake.log]  # same questions every run, so scores compare
