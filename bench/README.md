Scripts talk to the server on `localhost:8000` (or pass the port). Several of them need a big text file at `bench/corpus.txt`;
any ~10 MB code or prose dump works (I used a code corpus).

- `harness.py --label X --decode --prefill 8000,32000 [--ctxdecode 32000,65000] [--sample]`: decode tok/s, ms/step, tokens per step, prefill tok/s
- `canary.py`: six exact-answer arithmetic/code questions (sym W4A16 exports of this model answer 30 to the first one)
- `needle.py PORT TOKENS`: retrieval at long context. `concur.py`: 12 mixed concurrent requests vs. solo. `prefix.py PORT`: prefix-cache TTFT. `vision.py PORT`: image smoke test
- `fmha_bench.py`, `fmha_stress.py`, `fmha_decode_check.py`, `bitid.py`: attention kernel speed, exhaustive fp8 accuracy, bit-identity hashes (run inside the vLLM image with the `.so` mounted)
- `arith_battery.py [PORT] [N]`: 100 seeded in-head Python/arithmetic questions; a bigger version of the canary
- `decode_kl.py gen|score|cmp`: decode-path quality check for anything that only acts during decode (e.g. the GDN state dtype): generate greedily on the arm under test, re-score the same tokens with `prompt_logprobs` on a reference server, KL and top-1 by position. Run it for the reference arm too, as the floor. (Prompt-window checks go through chunked prefill and miss state-storage effects.)
- `tput_metric.py RESULT.json...`: for `vllm bench serve` results, reads the server log in each run's time window and reports how many requests actually ran at once, the engine's own generation rate, and running/TPOT next to concurrency/TPOT. Concurrency / TPOT is only a throughput when all requests ran at once; with speculative decoding at high concurrency, memory often limits that to a few.

