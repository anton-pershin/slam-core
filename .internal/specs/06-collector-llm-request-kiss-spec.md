## 06-collector-llm-request-kiss-spec

### 1. Requirement analysis

#### 1.1 Motivation

The ecosystem requirement is that one place decides what an LLM request is (constitution FR5 "shared core", NFR4 "modularity"): rally owns the request shape, slam-core wraps it in `Model` classes, and a consumer must not rebuild a request of its own.

What is missing: with performance monitoring enabled, slam-eval's streaming collector performs the prediction itself and builds that request from scratch — its own endpoint, its own headers, its own field list, and its own copy of the message assembly. Two consequences follow. Generation behaviour configured on the `Llm` never reaches the measured request, so the output-token cap and thinking control are dead configuration while monitoring is on and the two paths disagree about the same config. And the request the eval measures is only assumed to be the request `predict` would make, because nothing compares them.

#### 1.2 Functional requirements

**FR1.** The collector obtains the endpoint, the headers and the body from the `Llm` it is given — `build_headers()` and `build_payload(messages)` — and holds no url, authorization or model field of its own.

**FR2.** The collector adds exactly the streaming keys (`stream: True`, `stream_options: {"include_usage": True}`) to that body. The non-streaming request adds nothing. One body construction serves both paths.

**FR3.** Generation behaviour configured on the `Llm` therefore reaches the monitored request without slam-eval knowing about it: the output-token cap under both key names rally sends, and thinking control through `chat_template_kwargs` when `enable_thinking` is configured.

**FR4.** Equivalence: for the same messages and the same `Llm`, the monitored request equals the prediction request — same endpoint, same headers, same body modulo the streaming keys.

**FR5.** The messages are assembled in exactly one place, in slam-core, and used by both `predict` and the eval loop; slam-eval keeps no second copy of the rule.

**FR6.** The eval loop passes no request-shaping parameter to the collector and no longer aborts when no cap is configured — the cap is whatever the `Llm` carries, and an unset cap simply means no cap key goes out.

**FR7.** Measurement semantics are unchanged: TTFT from the first content chunk, TPOT from inter-chunk deltas, server usage preferred over chunk counting for token counts, and the streaming-unavailability contract keeps its three signals — the `supported`/`fallback_reason` block in run metadata, `n: 0` aggregates instead of omitted ones, and a warning emitted only when the fallback is not config-intended. Authorization failure still raises; other failures still produce a fallback record with `e2e_time_s: None`.

**FR8.** Tests assert the collector's request against the `Llm`'s own builder output — equivalence, headers, the thinking key, both cap keys — and the existing measurement, fallback, authorization and streaming-disabled behaviour stays green on the new constructor.

#### 1.3 Non-functional requirements

**NFR1.** rally remains the only place that knows the OpenAI request shape; slam-eval knows only the two streaming keys.

**NFR2.** No new dependencies; the suites run in the existing `~/venvs/slam`.

**NFR3.** The repository's linters report no message that is absent from the pre-change commit.

**NFR4.** Metric keys, record shape and aggregation are unchanged; stored artifacts remain readable by the existing reader.

**NFR5.** Intended wire changes, measured against the pre-change revision rather than asserted: `max_completion_tokens` is added on both the streaming and the non-streaming path, `chat_template_kwargs` is added when thinking is configured, and nothing is removed.

**NFR6.** The consequences of the thinking flag now applying are accepted and documented: the streamed content may carry the reasoning trace, which becomes `y_pred` and therefore the score; TTFT and `generated_tokens` include the reasoning while the flag is on, so thinking-on and thinking-off runs are not comparable; and with the flag on and a cap below the reasoning budget the collector reports streaming unavailability instead of an error. No removal of the trace happens here — slam-core strips it only on the in-process path, and rally's opt-in removal helper is not implemented.

**NFR7.** Scope: the in-process `LocalCausalLm` path, the memory sampler, the statistics registry and the storage adapter are untouched, and rally itself is not modified.

#### 1.4 Expected behavioural variants

| # | Situation | Expected behaviour |
|---|-----------|--------------------|
| 1 | `enable_thinking` configured on the `Llm` | the body carries `chat_template_kwargs` with that value (today: absent, flag ignored) |
| 2 | `enable_thinking` unset | no `chat_template_kwargs` key is sent |
| 3 | cap configured on the `Llm` | the body carries `max_completion_tokens` and `max_tokens` with the cap |
| 4 | no cap configured | neither cap key is sent and the run proceeds (today: the run aborts) |
| 5 | authorization configured | the `Authorization` header is sent verbatim (unchanged) |
| 6 | no authorization | no `Authorization` header (unchanged) |
| 7 | streaming request | body equals the `Llm` body plus `stream` and `stream_options` |
| 8 | non-streaming request | body equals the `Llm` body, no streaming keys |
| 9 | same messages, same `Llm` | the monitored request equals the prediction request — endpoint, headers and body modulo the streaming keys |
| 10 | system prompt present / absent | the shared builder produces the same two message lists `predict` uses |
| 11 | HTTP 401 or 403 | `RuntimeError`, the run aborts, never a silent metrics fallback (unchanged) |
| 12 | other HTTP status | fallback record, `supported: false`, `fallback_reason: streaming_request_rejected`, warning unless config-intended (unchanged) |
| 13 | transport error | fallback record with `e2e_time_s: None`, never a fabricated zero (unchanged) |
| 14 | server reports no usage | token counts from chunk counting, `tokens_source: chunk_count`, usage warning (unchanged) |
| 15 | thinking on with a cap below the reasoning budget | no content chunks arrive; streaming unavailability is reported, not an error (accepted consequence) |
| 16 | thinking on and the content carries a reasoning trace | the trace reaches `y_pred` and the score follows it (accepted consequence) |

### 2. Tests

[List all the tests explicitly covering all the requirements and the expected variants mentioned in the requirement analysis. Tag them as T1, T2 etc.]

### 3. Implementation plan

#### 3.1 Implementation repos

[List all the repos expected to be involved in any implementation]

#### 3.2 High-level design

[Mermaid diagram (typically, flowchart, but it is ultimately up to a planner) describing the high-level design. Take the high-level design from the constitution/validation spec and draw how your proposal fits into it.]

#### 3.3 Todo list

[Write a todo list with all the steps necessary to create an implementation which will allow the tests to be passed. Below is the template where the first two steps are mandatory]

1. [ ] Write the tests
2. [ ] Run all the tests and ensure that they fail
3. [ ] ...

#### 3.4 Modification summary

[Fill the table below specifying which files are going to be modified and which are going to be created]

| File | Action |
|------|--------|
| ... | Modified: add X, modify Y, etc. |
| ... | New |
