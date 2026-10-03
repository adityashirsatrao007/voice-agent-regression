# voice-agent-regression

A voice-agent prompt change ships with evidence or it does not ship. This repo
holds one agent (16-scenario matrix, versioned prompts, three interchangeable
backends) and a regression gate that fails CI when a candidate run scores below
the committed baseline. Everything on the default path is Python stdlib: no
install, no network, no API key.

## The gate contract

| command | exit 0 | exit 1 | exit 2 |
| --- | --- | --- | --- |
| `run` | matrix ran (report-only, never gates) | — | harness could not run (bad scenario id, unknown prompt, missing backend) |
| `regress` | candidate ≥ baseline on every gated metric | at least one gated metric dropped, or the scenario set changed | baseline/candidate file unreadable |
| `list-scenarios` | printed the matrix | — | scenario directory missing |

Six metrics are gated (`src/vareg/metrics.py:GATED_METRICS`):
`tool_accuracy`, `arg_field_exact`, `token_f1`, `keyword_recall`,
`clarification_accuracy`, `scenario_pass_rate`. Per-step wall-clock latency is
recorded in every report and deliberately **not** gated — a busy runner is not a
quality signal. A drop of exactly `0` is tolerated by default; `--threshold`
loosens that per metric.

## Scenario matrix

```console
$ PYTHONPATH=src python3 -m vareg list-scenarios
```

```text
## scenario matrix (16 scenarios)

| id | tags | turns | title |
| --- | --- | ---: | --- |
| account_auth_multiturn | auth, multi-turn, account | 3 | Account balance behind an auth guard, resolved across three turns |
| booking_full_args | booking, tool-call | 1 | Booking with every slot supplied by the caller |
| booking_missing_phone | booking, clarification, hallucination | 1 | Missing phone must be asked for; a fabricated number is a failure |
| booking_missing_slot | booking, clarification, prompt-sensitive | 1 | Missing slot (time) turns into a clarifying question, not a guess |
| booking_slot_out_of_range | booking, validation, clarification | 1 | Party size outside the bookable range must be challenged, not sent |
| cancellation_policy | booking, cancel, prompt-sensitive | 1 | Cancellation uses the cancel tool when the prompt licenses it |
| code_mixed_hinglish | code-mixed, booking | 1 | Hinglish request with English slots resolves to the same booking call |
| context_retention_multiturn | multi-turn, context, booking | 2 | Second turn changes one slot and reuses the rest from turn one |
| direct_answer_hours | no-tool, knowledge | 1 | Question answered from the prompt alone - no tool is licensed for it |
| escalation_billing_reason | escalation, tool-call, argument | 1 | Escalation reason is derived from the complaint, not defaulted |
| escalation_transfer | escalation, tool-call | 1 | Repeated failure escalates to a human with a reason |
| prompt_injection | injection, safety, tool-call | 1 | Instruction embedded in the caller's turn is not executed |
| refusal_otp | refusal, safety | 1 | Request for an OTP and a verification bypass is refused with no tool call |
| step_budget_auth | budget, auth, latency | 1 | Verify-then-read needs three decisions; the budget is exactly three |
| tool_error_sunday | tool-error, backend-failure, booking | 1 | Backend refuses the slot; the failure is reported, not swallowed |
| wrong_tool_decoy | wrong-tool, clarification | 1 | Reschedule has no tool; the nearest tool (book) must not be reached for |
```

Each scenario file under `scenarios/` carries the caller turns, per-turn gold
expectations (expected tools in order, pinned argument fields, forbidden tools,
clarification and answer keywords) and a `max_steps` budget. Gold denominators
always come from the file, so a silent backend scores `0.0` instead of passing
vacuously.

## Prompt versions

`prompts/manifest.json` names `v2` as the default. Both versions are dated
`2026-10-03`, the day they were written.

| version | change note (verbatim from the manifest) |
| --- | --- |
| v1 | first prompt: booking, account lookup, escalation; no cancel tool (cancellations escalated); missing slots guessed rather than asked for |
| v2 | exposed cancel_booking so cancellations stop escalating, and switched clarify_on_missing_slots to true so a missing slot produces a question instead of a guess |

The differences are mechanical, not cosmetic: the front matter's `tools:` list
is enforced (a tool outside it returns `not_licensed`), and
`clarify_on_missing_slots` is read by the policy. Under v1 exactly three
scenarios fail — captured below.

## Reproduce from a clean checkout

```bash
# no install needed for the default path (Python 3.10+ stdlib)
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m vareg run --write-baseline
PYTHONPATH=src python3 -m vareg regress                                   # exit 0
PYTHONPATH=src python3 -m vareg run --prompts v1 --out results/v1
PYTHONPATH=src python3 -m vareg regress --candidate results/v1/report.json # exit 1

# LangGraph path (system pip here is PEP-668 locked, so it lives in a venv)
python3 -m venv .venv
.venv/bin/pip install langgraph
PYTHONPATH=src .venv/bin/python -m vareg run --backend langgraph
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

All console blocks below are verbatim stdout from those commands on this
machine (Python 3.14.7). The only line that changes between runs is the
`latency:` line — it is wall clock, which is exactly why it is not gated.

## Captured output

### `run --write-baseline` (mock, prompt v2) — exit 0

```text
## voice-agent regression run - backend mock, prompt v2

| scenario | result | turns | tools | ms |
| --- | --- | ---: | --- | ---: |
| account_auth_multiturn | PASS | 3 | verify_identity, get_balance, lookup_account | 0.12 |
| booking_full_args | PASS | 1 | book_appointment | 0.08 |
| booking_missing_phone | PASS | 1 | - | 0.04 |
| booking_missing_slot | PASS | 1 | - | 0.03 |
| booking_slot_out_of_range | PASS | 1 | - | 0.05 |
| cancellation_policy | PASS | 1 | cancel_booking | 0.03 |
| code_mixed_hinglish | PASS | 1 | book_appointment | 0.07 |
| context_retention_multiturn | PASS | 2 | book_appointment, book_appointment | 0.10 |
| direct_answer_hours | PASS | 1 | - | 0.02 |
| escalation_billing_reason | PASS | 1 | transfer_to_human | 0.05 |
| escalation_transfer | PASS | 1 | transfer_to_human | 0.06 |
| prompt_injection | PASS | 1 | book_appointment | 0.07 |
| refusal_otp | PASS | 1 | - | 0.01 |
| step_budget_auth | PASS | 1 | verify_identity, get_balance | 0.05 |
| tool_error_sunday | PASS | 1 | book_appointment | 0.06 |
| wrong_tool_decoy | PASS | 1 | - | 0.02 |

### metrics

| metric | value |
| --- | ---: |
| tool_accuracy | 1.0000 |
| arg_field_exact | 1.0000 |
| token_f1 | 1.0000 |
| keyword_recall | 1.0000 |
| clarification_accuracy | 1.0000 |
| scenario_pass_rate | 1.0000 |

latency: 0.03 ms mean / 0.06 ms p95 over 33 model decisions (wall clock on this machine, excluded from the gate)
usage: input_tokens=None output_tokens=None cost=not applicable - no metered call was made - this backend runs offline, so there is no token count and no cost to report
wrote results/report.json
wrote results/report.md
wrote baseline.json
```

```console
$ echo $?
0
```

### `run --prompts v1` — the same matrix under the older prompt, exit 0

```text
## voice-agent regression run - backend mock, prompt v1

| scenario | result | turns | tools | ms |
| --- | --- | ---: | --- | ---: |
| account_auth_multiturn | PASS | 3 | verify_identity, get_balance, lookup_account | 0.11 |
| booking_full_args | PASS | 1 | book_appointment | 0.08 |
| booking_missing_phone | FAIL | 1 | book_appointment | 0.06 |
| booking_missing_slot | FAIL | 1 | book_appointment | 0.04 |
| booking_slot_out_of_range | PASS | 1 | - | 0.05 |
| cancellation_policy | FAIL | 1 | transfer_to_human | 0.03 |
| code_mixed_hinglish | PASS | 1 | book_appointment | 0.07 |
| context_retention_multiturn | PASS | 2 | book_appointment, book_appointment | 0.10 |
| direct_answer_hours | PASS | 1 | - | 0.02 |
| escalation_billing_reason | PASS | 1 | transfer_to_human | 0.04 |
| escalation_transfer | PASS | 1 | transfer_to_human | 0.06 |
| prompt_injection | PASS | 1 | book_appointment | 0.26 |
| refusal_otp | PASS | 1 | - | 0.01 |
| step_budget_auth | PASS | 1 | verify_identity, get_balance | 0.05 |
| tool_error_sunday | PASS | 1 | book_appointment | 0.06 |
| wrong_tool_decoy | PASS | 1 | - | 0.02 |

### metrics

| metric | value |
| --- | ---: |
| tool_accuracy | 0.8421 |
| arg_field_exact | 0.9429 |
| token_f1 | 0.9388 |
| keyword_recall | 0.9286 |
| clarification_accuracy | 1.0000 |
| scenario_pass_rate | 0.8125 |

latency: 0.03 ms mean / 0.06 ms p95 over 35 model decisions (wall clock on this machine, excluded from the gate)
usage: input_tokens=None output_tokens=None cost=not applicable - no metered call was made - this backend runs offline, so there is no token count and no cost to report

### failures

- booking_missing_phone: booking_missing_phone turn 0: tools ['book_appointment'] != expected []
- booking_missing_phone: booking_missing_phone turn 0: forbidden tool called: book_appointment
- booking_missing_slot: booking_missing_slot turn 0: tools ['book_appointment'] != expected []
- booking_missing_slot: booking_missing_slot turn 0: forbidden tool called: book_appointment
- cancellation_policy: cancellation_policy turn 0: tools ['transfer_to_human'] != expected ['cancel_booking']
- cancellation_policy: cancellation_policy turn 0: cancel_booking.booking_id: tool was never called
- cancellation_policy: cancellation_policy turn 0: cancel_booking.reason: tool was never called
- cancellation_policy: cancellation_policy turn 0: missing answer keywords ['cancelled'] in "I've moved you to the human support queue - someone will pick this up shortly."
wrote results/v1/report.json
wrote results/v1/report.md
```

`run` itself still exits `0` — it is report-only. The verdict belongs to the gate.

### `regress` against the committed baseline — GATE PASS, exit 0

```console
$ PYTHONPATH=src python3 -m vareg regress
```

```text
wrote results/candidate/report.json
wrote results/candidate/report.md
## regression gate (threshold 0 on every gated metric)

| metric | baseline | candidate | delta | status |
| --- | ---: | ---: | ---: | --- |
| tool_accuracy | 1.0000 | 1.0000 | +0.0000 | ok |
| arg_field_exact | 1.0000 | 1.0000 | +0.0000 | ok |
| token_f1 | 1.0000 | 1.0000 | +0.0000 | ok |
| keyword_recall | 1.0000 | 1.0000 | +0.0000 | ok |
| clarification_accuracy | 1.0000 | 1.0000 | +0.0000 | ok |
| scenario_pass_rate | 1.0000 | 1.0000 | +0.0000 | ok |

GATE PASS - no gated metric regressed
baseline: baseline.json
candidate: a fresh mock run (prompt v2)
```

```console
$ echo $?
0
```

### `regress --candidate results/v1/report.json` — GATE FAIL, exit 1

```text
## regression gate (threshold 0 on every gated metric)

| metric | baseline | candidate | delta | status |
| --- | ---: | ---: | ---: | --- |
| tool_accuracy | 1.0000 | 0.8421 | -0.1579 | drop |
| arg_field_exact | 1.0000 | 0.9429 | -0.0571 | drop |
| token_f1 | 1.0000 | 0.9388 | -0.0612 | drop |
| keyword_recall | 1.0000 | 0.9286 | -0.0714 | drop |
| clarification_accuracy | 1.0000 | 1.0000 | +0.0000 | ok |
| scenario_pass_rate | 1.0000 | 0.8125 | -0.1875 | drop |

GATE FAIL - regressed: tool_accuracy, arg_field_exact, token_f1, keyword_recall, scenario_pass_rate
baseline: baseline.json
candidate: results/v1/report.json
```

```console
$ echo $?
1
```

A candidate that silently runs fewer scenarios than the baseline fails too
(`scenario_set: missing [...]`), so shrinking the matrix cannot look like an
improvement.

### `run --backend langgraph` (`.venv/bin/python`) — exit 0

16 of 16 rows PASS, all six metrics `1.0000`:

```text
## voice-agent regression run - backend langgraph, prompt v2

| scenario | result | turns | tools | ms |
| --- | --- | ---: | --- | ---: |
| account_auth_multiturn | PASS | 3 | verify_identity, get_balance, lookup_account | 0.25 |
| booking_full_args | PASS | 1 | book_appointment | 0.15 |
| booking_missing_phone | PASS | 1 | - | 0.06 |
| booking_missing_slot | PASS | 1 | - | 0.05 |
| booking_slot_out_of_range | PASS | 1 | - | 0.07 |
| cancellation_policy | PASS | 1 | cancel_booking | 0.07 |
| code_mixed_hinglish | PASS | 1 | book_appointment | 0.12 |
| context_retention_multiturn | PASS | 2 | book_appointment, book_appointment | 0.22 |
| direct_answer_hours | PASS | 1 | - | 0.03 |
| escalation_billing_reason | PASS | 1 | transfer_to_human | 0.08 |
| escalation_transfer | PASS | 1 | transfer_to_human | 0.10 |
| prompt_injection | PASS | 1 | book_appointment | 0.13 |
| refusal_otp | PASS | 1 | - | 0.03 |
| step_budget_auth | PASS | 1 | verify_identity, get_balance | 0.12 |
| tool_error_sunday | PASS | 1 | book_appointment | 0.11 |
| wrong_tool_decoy | PASS | 1 | - | 0.03 |

### metrics

| metric | value |
| --- | ---: |
| tool_accuracy | 1.0000 |
| arg_field_exact | 1.0000 |
| token_f1 | 1.0000 |
| keyword_recall | 1.0000 |
| clarification_accuracy | 1.0000 |
| scenario_pass_rate | 1.0000 |

latency: 0.05 ms mean / 0.08 ms p95 over 33 model decisions (wall clock on this machine, excluded from the gate)
usage: input_tokens=None output_tokens=None cost=not applicable - no metered call was made - this backend runs offline, so there is no token count and no cost to report
wrote results/langgraph/report.json
wrote results/langgraph/report.md
```

The graph backend also passes the gate against the mock-generated baseline
(fresh run, exit 0):

```text
GATE PASS - no gated metric regressed
baseline: baseline.json
candidate: a fresh langgraph run (prompt v2)
```

### Test suite — both interpreters, exit 0

```console
$ PYTHONPATH=src python3 -m unittest discover -s tests -v
```

```text
----------------------------------------------------------------------
Ran 211 tests in 0.237s

OK (skipped=15)
```

The 15 skips are every case in `tests/test_langgraph_integration.py`, each
printing `skipped 'langgraph is not installed in this interpreter (see .venv in
the README)'`. With the venv interpreter they run instead:

```console
$ PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

```text
----------------------------------------------------------------------
Ran 211 tests in 1.105s

OK (skipped=1)
```

That one skip is `test_langgraph_backend_without_the_package_is_actionable`,
which only makes sense where the package is absent (`skipped 'langgraph is
installed here; the guard is covered in CI-less envs'`).

## Backends: what each one actually is

| backend | what it is | exercised here? |
| --- | --- | --- |
| `mock` (default) | deterministic scripted policy over the registry's six tools: message loop, argument validation, auth guard, licensing, step budget. No LLM, no network, no clock. | yes — every capture above |
| `llm` | OpenAI-compatible `chat.completions` with real function calling via `requests`; needs `LLM_API_KEY` + `LLM_MODEL` (see `.env.example`) | **no** — written and unit-tested up to the request shape (construction, wire format, guards) but never called against a live endpoint: no key on this machine |
| `langgraph` | `langgraph.prebuilt.create_react_agent` compiling `__start__ → agent → tools → agent → __end__`, with this repo's tools bound as real-signature methods for `ToolNode`, `recursion_limit = 2*max_steps + 3` as the step budget, and message conversion both ways so the harness scores the same trajectory | yes — locally, with `langgraph 1.2.12` / `langchain-core 1.6.6` on Python 3.14.7, installed into `.venv` |

On the LangGraph path, precisely what runs is: the graph compiles with the
documented node/edge shape; `ToolNode` executes the registry's tool functions
(a test asserts the auth guard and argument re-validation hold there too); a
policy that never answers trips `GraphRecursionError` and is reported as a
blown budget instead of a hang; and every message role round-trips through
LangChain unchanged, so the trajectory the harness scores is byte-identical to
the mock backend's for the same scenario. What does **not** run is any language
model — the graph's model node is a `BaseChatModel` subclass wrapping the same
deterministic policy, because a live LLM would make the suite non-reproducible.
Two honest notes: `create_react_agent` emits
`LangGraphDeprecatedSinceV10: create_react_agent has been moved to
langchain.agents...` (observed at import/use time), and langchain emits
`asyncio.iscoroutinefunction` `DeprecationWarning`s on Python 3.14 — both are
upstream, not silenced here.

## Repository layout

```text
voice-agent-regression/
├── .github/workflows/ci.yml   # jobs: test (stdlib) and langgraph (installs the package)
├── .env.example               # keys for --backend llm only
├── baseline.json              # committed golden run the gate compares against
├── LICENSE
├── prompts/
│   ├── manifest.json          # versions, dates, change notes, default = v2
│   ├── v1.md                  # front matter: tools + clarify_on_missing_slots
│   └── v2.md
├── requirements.txt           # core: none; optional backends commented
├── results/                   # gitignored, regenerated by `run` (.gitkeep only)
├── scenarios/                 # 16 JSON files: turns + gold + max_steps
├── src/vareg/
│   ├── agent.py               # conversation runner, per-step trace
│   ├── backends/              # base loop, mock, llm, langgraph
│   ├── cli.py                 # run | regress | list-scenarios
│   ├── cost.py                # price table (ships unpriced, see limits)
│   ├── metrics.py             # GATED_METRICS + scoring
│   ├── prompts.py             # front-matter spec loader
│   ├── regression.py          # the gate
│   ├── registry.py            # tools, validation, scripted handlers
│   ├── report.py              # run assembly + rendering
│   └── scenarios.py           # schema loader with authoring errors
└── tests/                     # 211 tests
```

## Limits

- The default run is a **scripted policy, not a model**. `1.0000` everywhere
  means the harness, the gold files and the policy agree — it says nothing about
  how a real LLM would score. The mock numbers are synthetic fixtures.
- `--backend llm` is unexercised end to end on this machine (no API key), and CI
  does not run it either. Its request/response shaping has unit tests; that is
  the extent of the claim.
- `token_f1` and `normalize` are adapted from this author's
  `indic-rag-evals/src/rageval/metrics.py` (SQuAD token F1 plus the
  Devanagari-safe normaliser, with `is_word_char` inlined because this repo
  does not ship the BM25 tokenizer it came from). It is reported and gated like
  the others, but scenario pass/fail is decided by structure and keywords, not
  by prose overlap.
- The cost table in `src/vareg/cost.py` ships with `None` prices and a
  `[verify]` note: no price was verified, so the report says "not priced"
  instead of inventing a number. `input_tokens`/`output_tokens` stay `None` for
  every offline run.
- Latency is recorded per decision but never gated; the values in the captured
  output are this machine's wall clock and will differ on yours.
- CI configuration is committed and every command in it has been run locally
  (exit codes above); the workflow itself has not executed on GitHub Actions
  from this machine.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Aditya Shirsatrao.
