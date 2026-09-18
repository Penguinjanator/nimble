# Nimble × Jev comparison app

Open <http://127.0.0.1:8891> on the computer running the app. The app is local;
Nimble inference runs on a private RunPod H100 and on this MacBook’s M5 Pro
through MLX. Jev uses the official TypeSafe API. The selected Nimble adapter was trained on 2,826 examples.

Select an example to see its context, question, candidate definitions, and all three
backends’ results. **Compare this example** runs all three backends concurrently on the
same input. Each card updates immediately when its backend responds; a slower
MacBook or remote request does not delay the other cards. Pending cards keep
their own loading state. Completed comparisons are saved even if the browser
disconnects. **Run all 30** evaluates the sample sequentially; its stop button
finishes the current comparison before stopping. **Export results** downloads the
saved comparisons, including edited inputs.

The sample is deterministic: seed 17 selects 10 Choice, 10 Noul (Boolean), and
10 Score examples from `data/eval.jsonl`, without using predictions or references
to select them. The 324-row holdout is unchanged. Reference labels are synthetic
and model-checked. They are joined after inference and never sent to providers.
Editing the context disables reference scoring for that edited input.

The result cards show the selected answer and normalized candidate probabilities.
Scores show the probability-weighted level and the most likely level; reference
matches and model agreement use the most likely level. API probability rounding
is normalized while preserving the original candidate order for ties. Jev's
separate distribution-confidence statistic is not treated as correctness.

**Response time** measures each request through JSON response parsing, from the
local app process. Nimble's request includes transport through an SSH tunnel;
Jev’s includes HTTPS transport and service time. MacBook response time includes
the local loopback request. **H100 scoring** and **Mac scoring** use separately
synchronized CUDA and Metal timers around the model calls. Jev’s API does not
provide a model-only scoring time. Different network routes and
serving stacks prevent treating these as hardware-normalized speed measurements.
Each Nimble backend receives three unscored warmups before it reports ready; Jev's first
request is retained. These are interactive observations, not a formal load test.

Both Nimble backends use the same adapter trained on 2,826 examples, with
2,048 input tokens maximum. Mac uses a merged BF16 checkpoint and an FP32
candidate head; H100 uses the unmerged BF16 adapter. Small numerical differences
are possible. Each produces its candidate distribution in one forward pass,
without autoregressive text generation. Agreement requires all three successful
answers to match. Older two-backend runs remain readable, but do not count toward
the three-way summary until rerun. Errors remain visible independently.

## Run locally

Use the existing Curator environment, which supplies `requests` and
`python-dotenv`:

```sh
.venv-curator/bin/python -m nimble.playground.server \
  --port 8891 --gpu-url http://127.0.0.1:18766 --mac-url http://127.0.0.1:18767
```

Start the MacBook worker in a separate terminal, using the prepared merged
checkpoint and MLX environment:

```sh
.venv-mlx/bin/python -m nimble.playground.mac_worker \
  --model-dir .cache/models/nimble-diverse9b-v2-bf16 \
  --warmups .cache/comparison-gpu/warmups.json --port 18767
```

The Mac worker keeps roughly 18 GiB of model memory resident for repeated
inference. It binds only to loopback and runs all Metal calls on one dedicated
thread. Stop that terminal process when finished to release the memory. Detached
session process IDs and logs are in `.cache/comparison-app/`.

The CUDA worker command is:

```sh
python worker.py --model-dir model --warmups warmups.json --port 8765
```

The worker expects the selected adapter, pinned base, and portable reference
`inference.py` / `parallel_schema.py` alongside the adapter. It binds only to
the GPU machine's loopback address. An SSH tunnel maps local port 18766 to
the GPU's loopback port 8765. Both model weights and provider inputs remain
off the public web; only the existing SSH port is exposed on the pod.

The current session's provisioning, tunnel helper, and automatic deletion guard
are recorded in `.cache/comparison-gpu/`. Its 30-minute deadline is shown in the
app. Results remain readable after the GPU is deleted. MacBook and Jev continue to
work; the H100 card reports its connection error until a new GPU session starts. The temporary GPU rate is capped at $3.50/hour.

The TypeSafe key is loaded from local `.env` and used only against
`https://api.typesafe.ai/v1/systemone`; it is never uploaded to RunPod or returned
to the browser. The local UI validates Host and Origin and requires a per-process
request token for inference. It binds only to `127.0.0.1`.

All comparisons are saved to `.cache/comparison-app/comparisons.jsonl`. The app
restores saved results on reload. It does not automatically retry failed requests;
any backend’s error remains visible without hiding the other backends’ results.

Tests:

```sh
.venv-curator/bin/python -m unittest tests.test_playground
```
