## 05-rally-request-migration-kiss-spec

### 1. Requirement analysis

#### 1.1 Motivation

The ecosystem requirement is that slam components take their LLM behaviour from the shared libraries they depend on, instead of rebuilding it themselves (constitution FR5 "shared core", NFR4 "modularity"). Rally has moved request construction inside the `Llm` object: `Llm.request(message_history)` is now the only entry point, and the module-level request functions are gone.

What is missing: `slam_core.model.LlmViaOpenAiApi` still builds its request by calling a removed module-level function and re-listing server url, authorization, model name and output-token limit itself. As a result the model does not import at all, every slam repo that imports `slam_core.model` is blocked, and generation behaviour that belongs to the `Llm` — notably thinking control — never reaches the request.

#### 1.2 Functional requirements

**FR1.** `LlmViaOpenAiApi.predict` obtains the assistant message through the `Llm` instance it holds — `self.llm.request(messages)` — passing the message list as the only argument. Exactly one request is issued per `predict` call.

**FR2.** `slam-core` imports no rally symbol other than `Llm`, and no module-level rally request function is referenced anywhere in the slam repos — production code or tests.

**FR3.** Generation parameters (server url, authorization, model name, output-token limit, thinking control) are supplied by the `Llm` object and are not re-listed at the call site: changing any of them requires no change in slam-core.

**FR4.** A `Llm` configured with `enable_thinking` therefore reaches the request unchanged — thinking control works for the OpenAI-compatible model with no slam-core code, and no slam-core code strips or inspects the reasoning trace.

**FR5.** When the `Llm` returns `None` (the request failed), `predict` raises an explicit error naming the model, instead of failing on a subscript of `None`.

**FR6.** Tests exercise the boundary, not the old symbol: slam-core's model tests drive `predict` against a `Llm` double and assert the message list handed to it; slam-eval's end-to-end tests stub the request at the `Llm` boundary.

**FR7.** The slam repos import and their suites run again: `import slam_core.model` succeeds, slam-core's suite collects, and slam-eval's end-to-end suite collects and passes.

#### 1.3 Non-functional requirements

**NFR1.** Only `rally.llm.Llm` is imported from rally by slam-core; the removed module-level functions appear nowhere in the slam repos.

**NFR2.** The message list slam-core assembles is unchanged: the system message when a system prompt is present, then the user message.

**NFR3.** No new dependencies; the suites run in the existing `~/venvs/slam`.

**NFR4.** The repository's own linters (black, isort, pylint, mypy) report no message that is absent from the pre-change commit.

**NFR5.** Scope: the performance-monitoring collection path in slam-eval is not modified by this spec; it bypasses `predict` today and continues to do so.

**NFR6.** On the `predict` path the only intended change to what goes on the wire is the thinking-control key now being sent when configured; the remaining keys and the message list are identical to today. This is measured against the pre-change revision, not asserted.

#### 1.4 Expected behavioural variants

| # | Situation | Expected behaviour |
|---|-----------|--------------------|
| 1 | `predict` with a system prompt | the `Llm` receives `[system, user]` |
| 2 | `predict` without a system prompt | the `Llm` receives `[user]` |
| 3 | the `Llm` returns an assistant message | `predict` returns its `content` |
| 4 | the `Llm` returns `None` (request failed) | `predict` raises an error naming the model (today: `TypeError` on subscripting `None`) |
| 5 | `enable_thinking` configured on the `Llm` | the value reaches the request through the `Llm`; slam-core passes no generation parameter and no thinking key of its own |
| 6 | `enable_thinking` unset | no thinking key is sent; the request is otherwise as today |
| 7 | url / authorization / model / output-token limit configured | read by the `Llm`, not re-listed at the call site |
| 8 | the assistant `content` carries a reasoning trace | returned verbatim by `predict` (unchanged); no removal happens in slam-core |
| 9 | any slam module imports `slam_core.model` | import succeeds; no reference to the removed functions anywhere in slam-core or slam-eval |
| 10 | slam-eval end-to-end run with the request stubbed at the `Llm` boundary | answers and scores are the ones the suite asserts today |
| 11 | `predict` called repeatedly with different inputs | one `Llm.request` call per `predict`, in order, no caching |
| 12 | performance monitoring enabled during an eval run | unchanged by this spec: the collector path still builds its own request |

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
