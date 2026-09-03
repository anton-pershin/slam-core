## slam-train: minimal trl-based SFT with slam-eval evaluation

### 1. Requirement analysis

#### 1.1 Motivation

Tasks 4 and 5 in section 6.1 of the constitution spec require extending `slam-core` with
training support and creating component repos for training. The `slam-train` repo exists
but contains only a scaffold with an empty `main()`.

The goal of this spec is deliberately narrow: a **simple, fully `trl`-based LoRA SFT
implementation** whose training data is derived from an existing `slam-core`
`EvalCaseCollection`, and whose resulting adapter can be evaluated by `slam-eval` without
manual conversion. No custom abstractions are introduced in `slam-train` beyond what is
needed to run the pipeline — the purpose is to *discover* which abstractions the ecosystem
actually needs by first building the simplest thing that works. Abstraction design is
explicitly deferred to a later spec.

A secondary motivation is that acceptance criterion 3 of the constitution spec
("a model trained by `slam-train` can be loaded and evaluated by `slam-eval` using the same
collection and scorer definitions") is currently unsatisfiable: `slam-core` has no `Model`
implementation that loads a local checkpoint. Both existing implementations
(`LlmViaOpenAiApi`, `EmbeddingBasedTextClassifier`) are unsuitable — the former requires a
running inference server, the latter is a classifier. This spec closes that gap.

#### 1.2 Functional requirements

**FR1 — Training data derived from an `EvalCaseCollection`.** `slam-train` must build its
SFT dataset by consuming a `slam_core.collections.base.EvalCaseCollection` instantiated from
a shared Hydra collection config, not by parsing dataset files itself. This guarantees the
training prompt is produced by the same `user_prompt_template` code path that `slam-eval`
uses at evaluation time, so the two cannot drift.

**FR2 — Prompt-completion dataset construction.** Each eval case must be converted into a
`trl` prompt-completion record: `prompt` from `x["user_prompt"]` and `completion` from
`y_true`. Loss must be computed on the completion only.

**FR3 — Ground-truth serialization.** When `y_true` is not a string (for `MergeQuality` it is
a `dict`), it must be serialized to the completion string with
`json.dumps(y_true, ensure_ascii=False, indent=2)`. This matches what
`slam_core.scorers.merge_quality_scorer.safe_parse_prediction` will parse back at eval time,
so the model is trained to emit exactly the format the scorer accepts. String `y_true` values
are used verbatim.

**FR4 — System prompt handling.** If `x["system_prompt"]` is not `None`, it must be preserved
in the training record. If it is `None` (the case for all current collections), no system
message is emitted.

**FR5 — Shuffled, ratio-based train/eval split.** The collection must be materialized in full,
shuffled with a configurable random seed, and split into disjoint training and evaluation
subsets by a configurable **ratio** (e.g. `train_ratio: 0.8`). The split is a property of the
training procedure, not of the collection: the ratio and the seed are configured in
`slam-train` only, and `EvalCaseCollection` is left unchanged.

Shuffling is mandatory, not cosmetic: nothing guarantees that a collection's on-disk order is
random, so splitting the raw order risks a split biased by whatever order the generator emitted
(e.g. by persona, chunk format, or difficulty). The seed must be Hydra-configurable so that a
given (collection, seed, ratio) triple always yields exactly the same split.

Training must never see examples used for evaluation. The available
`merge_quality` datasets make this mandatory rather than optional:
`merge_quality_dataset_easy_tiny.jsonl` consists of 3 lines that are byte-identical to lines
of `merge_quality_dataset_easy.jsonl`, so the default `slam-eval` configuration would
evaluate on training data. (`merge_quality_hard.jsonl`, referenced by
`config/collection/merge_quality/merge_quality_hard.yaml`, does not exist on disk and is out
of scope here.)

**FR6 — LoRA fine-tuning via `trl` + `peft`.** Training must use `trl.SFTTrainer` with a
`peft.LoraConfig` passed as `peft_config`. All LoRA hyperparameters (`r`, `lora_alpha`,
`lora_dropout`, `target_modules`) must be Hydra-configurable; `target_modules` in particular
must not be hardcoded, since it is architecture-dependent.

**FR7 — Adapter artifact output.** Training must write the trained LoRA adapter, the
tokenizer, and the resolved training configuration to a configurable output directory, in the
standard `peft` adapter layout so that `peft.PeftModel.from_pretrained` can load it.

**FR8 — Local causal-LM `Model` in `slam-core`.** A new
`slam_core.model.LocalCausalLm(Model)` must load a base causal LM and, optionally, a LoRA
adapter directory, and implement `predict(x: TextGenerationInput) -> str` by applying the
tokenizer's chat template and generating a completion. It must accept a `TextGenerationInput`
exactly as the existing collections emit it, so no collection changes are required.

**FR9 — Deterministic decoding by default.** `LocalCausalLm` generation parameters
(`max_new_tokens`, `do_sample`, `temperature`) must be configurable, defaulting to greedy
decoding (`do_sample: false`) so evaluation results are reproducible. Only the newly generated
completion must be returned — the echoed prompt must be stripped.

**FR10 — Shared Hydra config for the new model.** A `config/model/local_hf_causal_lm.yaml`
must be added to `slam-core`'s shared config tree so `slam-eval` resolves it via `searchpath`
and can evaluate a trained adapter with a single `model=` override plus an adapter-path
override.

**FR11 — Evaluation stays a separate invocation.** `slam-train` must not depend on the
`slam-eval` package and must not call its `main()`. The train→eval handoff is the adapter
directory artifact: after training, `slam-eval` is run separately against the same collection
and scorer. This keeps the constitution's dependency graph intact, where `slam-core` is the
only package dependency and `slam-train → slam-eval` is an artifact edge. Publishing a
generated collection config to the shared registry (FR12) does not violate this: it is a write
to a shared config namespace, not a package dependency, and `slam-eval` is never imported.

**FR12 — Eval subset published to the shared config registry.** Because the split is owned by
`slam-train` and `EvalCaseCollection` is unchanged (FR5), `slam-eval` has no way to reconstruct
which examples were held out. Training must therefore publish the holdout as a pair of
artifacts:

1. **The eval subset as JSONL**, written in the same schema as the source dataset file, so that
   a `MergeQuality`-style collection can read it via `jsonl_path`.
2. **A generated collection config** for that JSONL, written into the shared config tree under
   `collection/`. The config is derived from the resolved source collection config node by
   copying it verbatim and overriding only `jsonl_path` (pointing at the artifact from 1) and
   `name`. Every other field — `_target_`, `user_prompt_template` — is inherited unchanged, so
   the holdout collection is evaluated through exactly the same prompt-construction code path
   as training used, preserving FR1's no-drift property.

The shared config tree is addressed through the same `${slam_shared_config}` key used for the
searchpath (FR13), so the generated config is resolvable by `slam-eval` with a single
`collection=` override and no additional searchpath entry. This tree is understood as a *slam
registry* — a shared namespace expected to move out of `slam-core` into its own location — so
publishing into it is a deliberate registry write, not a cross-repo modification.

**FR12.1 — Split-identifying hash in the collection name.** The generated collection's `name`
must be the source collection name suffixed with a short hash derived from a canonical string
of **all** split parameters — the train ratio *and* the random seed. Both are required: the
same ratio with a different seed yields an entirely different holdout, so hashing the ratio
alone would let two incompatible splits collide on one name. This guarantees that distinct
splits can never be confused with one another, and that re-running with identical split
parameters is idempotent (it regenerates the same name and overwrites the same files).

The name is what `EvalStorageAdapter.save()` embeds into `result_id`
(`eval:{group_id}:{timestamp}_M_{model.name}_C_{collection.name}`) and stores as
`eval_case_collection`, so a hash-bearing name is what keeps evaluation results on different
splits distinguishable downstream in `slam-monitoring`.

**FR12.2 — Scope: file-backed collections only.** This mechanism applies to collections whose
examples are read from a local file path (e.g. `MergeQuality` via `jsonl_path`). Collections
that load from a remote source with no overridable path (e.g. `BigBenchHard`, which calls
`datasets.load_dataset`) cannot express a holdout this way; splitting those is out of scope for
this spec. The dataset builder must fail with an explicit error, rather than silently producing
an unusable config, if the configured source collection has no `jsonl_path` field.

**FR13 — Hydra-configured entry point.** Training must run via
`slam_train/scripts/train_sft.py` configured by `config/config_train_sft.yaml`, following the
established pattern (`defaults` list with `user_settings` and `hydra`). The shared config tree
of `slam-core` must be resolved **exactly as `slam-eval` does it** — the same
`slam_shared_config: ${oc.env:SLAM_SHARED_CONFIG,<path-to-slam-core/config>}` key and the same
`hydra.searchpath: [file://${slam_shared_config}]` block as in
`slam-eval/config/config_main.yaml`, so both repos consume shared `collection/`, `model/`, and
`scorer/` configs through an identical mechanism. The scaffold's placeholder `main.py` /
`config_main.yaml` are removed, since `main` does not describe what the script does.

#### 1.3 Non-functional requirements

**NFR1 — No new abstractions in `slam-train`.** Beyond a single dataset-building helper and
the training script, no base classes, registries, or plugin layers are introduced. Deciding
what the ecosystem's training abstractions should be is deferred until this simple
implementation has been exercised.

**NFR2 — `slam-core` is the only package dependency.** `slam-train` depends on `slam-core`,
`trl`, `peft`, `transformers`, `torch`, `datasets`, and `hydra-core`. It must not depend on
`slam-eval`, `slam-datagen`, or `slam-monitoring`.

**NFR3 — Recent library releases.** Pin reasonably fresh minimum versions compatible with the
installed `transformers` 5.x / `torch` 2.13 line: `trl >= 1.12`, `peft >= 0.20`. The
prompt-completion dataset format with `completion_only_loss` defaulting to `True` is a `trl`
1.x behaviour and must not be re-implemented manually.

**NFR4 — Local verification is smoke-test-only.** The development machine has no GPU
(2 cores, ~1 GB RAM, CUDA unavailable), so it can only verify wiring and shapes, using a
tiny randomly-initialised model on CPU. Real training runs on 0.5B-class models
(Qwen2.5-0.5B / Qwen3-0.6B class) happen on other hardware and are out of scope for automated
verification here.

**NFR5 — Configuration over literals.** No model names, paths, hyperparameters, split ratios,
or random seeds appear as literals in code; all come from Hydra configs, with user-specific
paths in `user_settings.yaml`.

**NFR6 — Linters and typing.** All new code passes `black`, `isort`, `pylint`, and `mypy`
with type hints, consistent with the other ecosystem repos.

#### 1.4 Expected behavioural variants

| Situation | Expected behaviour |
|---|---|
| Collection yields `y_true` as a `dict` (e.g. `MergeQuality`) | Serialized via `json.dumps(..., ensure_ascii=False, indent=2)` into the completion |
| Collection yields `y_true` as a `str` (e.g. `BigBenchHard`) | Used verbatim as the completion, no serialization |
| `x["system_prompt"]` is `None` | No system message in the training record |
| `x["system_prompt"]` is set | System prompt preserved in the training record |
| `train_ratio` is set (e.g. `0.8`) | Collection is materialized, shuffled with the configured seed, then split into an 80% training subset and a disjoint 20% evaluation subset |
| Same collection, seed, and ratio used twice | Identical split both times — training subset and eval subset are reproducible |
| Same collection and ratio, different seed | Different train/eval subsets, still disjoint and still in the configured proportion |
| `train_ratio: 1.0` | All examples used for training; eval subset is empty and no eval JSONL is written |
| `train_ratio` outside `(0, 1]` | Error at config validation, before any model is loaded |
| Split produces an empty training subset (tiny collection with a small ratio) | Error, since `SFTTrainer` cannot train on nothing |
| Training completes with a non-empty eval subset | Eval subset written as JSONL in the source dataset schema, plus a generated collection config published to the shared registry under `collection/` |
| Generated collection config is inspected | Identical to the source collection config except `jsonl_path` (points at the holdout JSONL) and `name` (hash-suffixed) |
| Two runs differ only by `train_ratio` | Different hashes → different collection names and different config files; neither overwrites the other |
| Two runs differ only by `seed` | Different hashes → different collection names and different config files; neither overwrites the other |
| Re-run with identical collection, ratio, and seed | Same hash → same name, same file paths; artifacts are overwritten idempotently |
| Source collection has no `jsonl_path` (e.g. `BigBenchHard`) | Explicit error — holdout publication is unsupported for non-file-backed collections |
| `LocalCausalLm` configured with an adapter path | Base model loaded, then LoRA adapter applied on top |
| `LocalCausalLm` configured with `adapter_path: null` | Base model evaluated as-is (baseline run, no adapter) |
| `LocalCausalLm` with `do_sample: false` (default) | Greedy decoding; repeated runs give identical output |
| Generated output contains the echoed prompt | Only newly generated tokens are returned |
| Training completes | Adapter, tokenizer, and resolved config written to the output directory |
| Output directory already exists | Contents overwritten (standard `Trainer` behaviour), no separate guard |

### 2. Tests

Testing is split into two tiers, reflecting NFR4: everything deterministic is covered by fast
unit tests with test doubles, and exactly one test touches `SFTTrainer`. No test downloads a
model or a dataset from the network, and no test requires a GPU.

**Test doubles.** Following the convention already used in `slam-core/tests/` and
`slam-eval/tests/e2e/test_main.py`, tests use fake `EvalCaseCollection` subclasses rather than
mocking the base class. Two are needed:
- `FakeDictCollection` — yields `y_true` as a `dict` (mimics `MergeQuality`).
- `FakeStrCollection` — yields `y_true` as a `str` (mimics `BigBenchHard`).

#### 2.1 Dataset building (`slam-train`, unit)

| # | Test | Covers |
|---|---|---|
| T1 | Prompt of each built record equals `x["user_prompt"]` verbatim from the collection | FR1, FR2 |
| T2 | Built records use the `prompt`/`completion` key pair expected by `trl` | FR2 |
| T3 | `dict` `y_true` becomes `json.dumps(y_true, ensure_ascii=False, indent=2)` in `completion` | FR3, row 1 |
| T4 | Round-trip: `json.loads` of a built `completion` equals the original `y_true` dict — i.e. `safe_parse_prediction` accepts what the model is trained to emit | FR3 |
| T5 | `str` `y_true` is used verbatim, with no JSON quoting applied | FR3, row 2 |
| T6 | `system_prompt=None` produces a record with no system message | FR4, row 3 |
| T7 | `system_prompt` set is preserved in the record | FR4, row 4 |

#### 2.2 Splitting (`slam-train`, unit)

| # | Test | Covers |
|---|---|---|
| T8 | `train_ratio=0.8` over 100 examples yields 80 train / 20 eval | FR5, row 5 |
| T9 | Train and eval subsets are disjoint and their union is the whole collection | FR5 |
| T10 | The split is not raw file order — a collection with a sorted on-disk order does not yield a train subset equal to its first N items | FR5 (shuffle is real) |
| T11 | Same (collection, seed, ratio) twice → identical split | FR5, row 6 |
| T12 | Same collection and ratio, different seed → different split, still disjoint and still in the configured proportion | FR5, row 7 |
| T13 | `train_ratio=1.0` → all examples train, eval subset empty, no eval artifacts written | row 8 |
| T14 | `train_ratio` of `0.0`, `-0.1`, `1.5` → error raised before any model is loaded | row 9 |
| T15 | Ratio small enough to empty the train subset on a tiny collection → error | row 10 |

#### 2.3 Holdout publication (`slam-train`, unit)

| # | Test | Covers |
|---|---|---|
| T16 | Eval JSONL is written, has one line per held-out example, and each line matches the source dataset schema | FR12, row 11 |
| T17 | Written eval JSONL is loadable by the real `MergeQuality` collection and yields the expected number of eval cases | FR12 |
| T18 | Generated collection config is written under `collection/` in the shared config tree | FR12, row 11 |
| T19 | Generated config differs from the source config **only** in `jsonl_path` and `name`; `_target_` and `user_prompt_template` are byte-identical | FR12, row 12 |
| T20 | Generated `name` carries a hash suffix and differs when only `train_ratio` differs | FR12.1, row 13 |
| T21 | Generated `name` differs when only `seed` differs | FR12.1, row 14 |
| T22 | Identical collection, ratio, and seed → same hash, same name, same paths; re-running overwrites rather than accumulating | FR12.1, row 15 |
| T23 | Source collection config without `jsonl_path` → explicit error, no partial artifacts left behind | FR12.2, row 16 |

#### 2.4 `LocalCausalLm` (`slam-core`, unit)

These use a tiny randomly-initialised causal LM built from a small `transformers` config
(a few layers, tiny hidden size, small vocab) constructed in-process — never downloaded.

| # | Test | Covers |
|---|---|---|
| T24 | `predict()` accepts a `TextGenerationInput` exactly as collections emit it and returns a `str` | FR8 |
| T25 | Returned string does not contain the prompt — only newly generated tokens | FR9, row 20 |
| T26 | With `do_sample=False`, two `predict()` calls on the same input return identical output | FR9, row 19 |
| T27 | `max_new_tokens` is respected — generated token count does not exceed it | FR9 |
| T28 | `adapter_path=None` loads the base model with no `peft` wrapping | row 18 |
| T29 | `adapter_path` set to a saved adapter directory produces a `PeftModel`-wrapped model | FR8, row 17 |
| T30 | `system_prompt` is included in the chat template when present and omitted when `None` | FR8 |

#### 2.5 Config composition (both repos, unit)

| # | Test | Covers |
|---|---|---|
| T31 | `config_train_sft.yaml` composes successfully and resolves shared `collection/` and `model/` groups via searchpath | FR13 |
| T32 | `slam-train`'s `slam_shared_config` / `hydra.searchpath` block is structurally identical to `slam-eval`'s | FR13 |
| T33 | `config/model/local_hf_causal_lm.yaml` instantiates `LocalCausalLm` via `hydra.utils.instantiate` with all generation parameters bound | FR10 |

#### 2.6 Training smoke test (`slam-train`, slow)

A single end-to-end test, marked `@pytest.mark.slow` so the default `pytest` run stays fast.

| # | Test | Covers |
|---|---|---|
| T34 | Full `train_sft` run on a tiny randomly-initialised model, a ~10-example fake collection, 1 epoch, CPU: completes and writes an adapter directory loadable by `peft.PeftModel.from_pretrained`, plus tokenizer, resolved config, eval JSONL, and generated collection config; re-running into the same output directory overwrites it | FR6, FR7, FR12, rows 21, 22 |

T34 asserts *wiring*, not learning: it makes no claim about loss decreasing or scores improving.
Importing `torch` alone takes ~2.5 minutes on the development machine (1 GB RAM, no GPU, heavy
swapping), so T34 is expected to be slow and is excluded from the default run.

#### 2.7 Manual verification

The following cannot be meaningfully automated here and are verified by hand on GPU hardware:

1. A real LoRA SFT run on a 0.5B-class model (Qwen2.5-0.5B / Qwen3-0.6B) over
   `merge_quality_easy` completes with a decreasing training loss.
2. `slam-eval` run against the generated holdout collection config with
   `model=local_hf_causal_lm` and the trained `adapter_path` produces `MergeQualityScorer`
   results, and the adapter scores above the `adapter_path: null` baseline on the same holdout.
3. `black`, `isort`, `pylint`, and `mypy` pass on both repos (NFR6).

### 3. Implementation plan

#### 3.1 Implementation repos

- **`slam-train`** (anton-pershin/slam-train) — the SFT pipeline: dataset building, splitting,
  holdout publication, the training script, and its configs. Currently a scaffold with a
  single `Add scaffold` commit on `master`.
- **`slam-core`** (anton-pershin/slam-core) — receives `LocalCausalLm` (FR8), its shared config
  `config/model/local_hf_causal_lm.yaml` (FR10), and the new `torch`/`transformers`/`peft`
  dependencies this requires. `slam-core` is also the management repo where this spec lives,
  but the changes listed here are library changes, not spec changes.

`slam-eval` is **not** an implementation repo: it consumes the generated collection config and
the shared model config through the existing searchpath mechanism, and needs no code change.
Note that `slam-train`'s default branch is `master` while the other repos use `main`; the
implementation branch is created from `master` accordingly.

#### 3.2. Solution design

**`slam-core` changes.**

1. **`LocalCausalLm(Model)`** added to `slam_core/model.py`, alongside the existing
   `LlmViaOpenAiApi` and `EmbeddingBasedTextClassifier`. Constructor takes
   `name`, `base_model_path`, `adapter_path: Optional[str]`, `max_new_tokens`, `do_sample`,
   `temperature`, and `device`. It loads the base model with
   `AutoModelForCausalLM.from_pretrained`, wraps it with `PeftModel.from_pretrained` only when
   `adapter_path` is not `None`, and loads the tokenizer from the base model path.
   `predict(x: TextGenerationInput) -> str` builds a message list (system message only when
   `x["system_prompt"]` is not `None`), applies `tokenizer.apply_chat_template` with
   `add_generation_prompt=True`, generates, and decodes **only the tokens beyond the prompt
   length** so the echoed prompt is never returned.
2. **`config/model/local_hf_causal_lm.yaml`** — a shared config targeting `LocalCausalLm` with
   every generation parameter bound and `do_sample: false` by default, in the same style as the
   existing `local_llm.yaml`.
3. **Dependencies** — `torch`, `transformers`, and `peft` added to `slam-core`'s
   `pyproject.toml` and `requirements.txt`. `slam-core` currently declares none of them, yet
   `LocalCausalLm` cannot work without them.

**`slam-train` changes.** Three small modules under `slam_train/`, deliberately plain functions
and one dataclass rather than a class hierarchy (NFR1):

1. **`slam_train/data/sft_dataset.py`** — the collection→SFT conversion.
   - `materialize_collection(collection) -> list[EvalCase]`: calls `load()` and drains the
     iterator into a list (the collection is a one-shot iterator, so this is required).
   - `serialize_completion(y_true) -> str`: returns `y_true` unchanged if it is a `str`,
     otherwise `json.dumps(y_true, ensure_ascii=False, indent=2)` (FR3).
   - `build_records(cases) -> list[dict]`: produces `{"prompt": ..., "completion": ...}` pairs,
     carrying the system prompt only when present (FR2, FR4).
   - `split_cases(cases, train_ratio, seed) -> tuple[list, list]`: validates the ratio is in
     `(0, 1]`, copies the list, shuffles it with `random.Random(seed)`, and slices at
     `floor(n * train_ratio)`; raises if the training subset would be empty (FR5).
2. **`slam_train/data/holdout.py`** — holdout publication (FR12).
   - `split_hash(train_ratio, seed) -> str`: a short hex digest (first 8 chars of a SHA-256)
     over a canonical `f"ratio={train_ratio}|seed={seed}"` string. Both parameters participate,
     so different seeds at the same ratio cannot collide (FR12.1).
   - `write_holdout_jsonl(...)`: writes the held-out examples in the *source dataset schema*.
     Because `MergeQuality` discards the original line shape when building an `EvalCase`, the
     split is carried out over the **raw JSONL lines** and the eval cases are built from them,
     rather than trying to reconstruct source records from eval cases.
   - `publish_collection_config(...)`: deep-copies the resolved source collection config node,
     overrides `jsonl_path` and `name` (source name plus `_holdout_<hash>`), and
     `OmegaConf.save`s it under `${slam_shared_config}/collection/`. Raises if the source config
     has no `jsonl_path` key (FR12.2).
3. **`slam_train/scripts/train_sft.py`** — the entry point. Instantiates the collection from the
   shared config, splits, builds records, wraps the training records in a
   `datasets.Dataset`, constructs `LoraConfig` and `SFTConfig` from Hydra, runs
   `SFTTrainer(model=..., train_dataset=..., args=..., peft_config=...)`, then saves the adapter
   and tokenizer and publishes the holdout artifacts.

**Why the split runs over raw lines.** FR12 requires the holdout JSONL to match the source
dataset schema, but an `EvalCase` is a lossy projection of a source line (`MergeQuality` keeps
only `attributes`, `provided_identifiers`, and chunk `content`, dropping `format` and
`owner_id`). Splitting the raw lines first and deriving eval cases from the chosen subsets is
the only way to emit a faithful holdout file. This is also why FR12.2 restricts the mechanism to
file-backed collections.

**Loss masking is not implemented.** `trl >= 1.12` computes loss on the completion only for
prompt-completion datasets by default (`completion_only_loss=True`), so no manual label masking
is written. Likewise LoRA is applied by passing `peft_config` to `SFTTrainer` rather than
wrapping the model with `get_peft_model` by hand.

**Hydra config layout in `slam-train`:**

```
config/
  config_train_sft.yaml        # defaults: user_settings, hydra, collection, + searchpath
  lora/default.yaml            # r, lora_alpha, lora_dropout, target_modules
  sft/default.yaml             # epochs, batch size, lr, max_length, output_dir
  user_settings/user_settings.yaml
  hydra/...                    # unchanged from scaffold
```

`config_train_sft.yaml` carries the `slam_shared_config` key and `hydra.searchpath` block copied
verbatim from `slam-eval/config/config_main.yaml` (FR13), plus `split.train_ratio`, `split.seed`,
and `base_model_path`.

#### 3.3 Todo list

1. [ ] Write the tests (T1–T34) as specified in section 2, including the `FakeDictCollection` /
       `FakeStrCollection` doubles and the `slow` marker registration for T34
2. [ ] Run all the tests and ensure that they fail
3. [ ] (slam-core) Add `torch`, `transformers`, `peft` to `pyproject.toml` and
       `requirements.txt`
4. [ ] (slam-core) Implement `LocalCausalLm` in `slam_core/model.py`
5. [ ] (slam-core) Add `config/model/local_hf_causal_lm.yaml`
6. [ ] (slam-core) Run T24–T30 and T33 and ensure they pass
7. [ ] (slam-train) Add dependencies: `slam-core`, `trl >= 1.12`, `peft >= 0.20`,
       `transformers`, `torch`, `datasets`
8. [ ] (slam-train) Implement `slam_train/data/sft_dataset.py`
9. [ ] (slam-train) Implement `slam_train/data/holdout.py`
10. [ ] (slam-train) Run T1–T23 and ensure they pass
11. [ ] (slam-train) Add `config/lora/default.yaml`, `config/sft/default.yaml`, and
        `config/config_train_sft.yaml` with the searchpath block copied from `slam-eval`
12. [ ] (slam-train) Implement `slam_train/scripts/train_sft.py`
13. [ ] (slam-train) Remove the scaffold placeholders `slam_train/scripts/main.py`,
        `config/config_main.yaml`, and the empty `tests/test_main.py`
14. [ ] (slam-train) Run T31–T32 and ensure they pass
15. [ ] (slam-train) Run the T34 smoke test and ensure it passes
16. [ ] Update `README.md` in both repos: `slam-train` gets a `train_sft.py` section
        (configuration, outputs, and the follow-up `slam-eval` invocation against the generated
        holdout config); `slam-core` gets `LocalCausalLm` documented in `docs/models.md`
17. [ ] Run `black`, `isort`, `pylint`, `mypy` on both repos and fix all findings
18. [ ] Manual verification items 1–3 from section 2.7 on GPU hardware

#### 3.4 Modification summary

| File | Repo | Action |
|------|------|--------|
| `slam_core/model.py` | slam-core | Modified: add `LocalCausalLm(Model)` |
| `config/model/local_hf_causal_lm.yaml` | slam-core | New: shared config targeting `LocalCausalLm` |
| `pyproject.toml` | slam-core | Modified: add `torch`, `transformers`, `peft` dependencies |
| `requirements.txt` | slam-core | Modified: add `torch`, `transformers`, `peft` |
| `docs/models.md` | slam-core | Modified: document `LocalCausalLm` |
| `tests/test_model.py` | slam-core | Modified: add T24–T30 |
| `tests/test_config.py` | slam-core | New: T33 (model config instantiation) |
| `slam_train/data/__init__.py` | slam-train | New |
| `slam_train/data/sft_dataset.py` | slam-train | New: materialize, serialize, build records, split |
| `slam_train/data/holdout.py` | slam-train | New: split hash, holdout JSONL, config publication |
| `slam_train/scripts/train_sft.py` | slam-train | New: Hydra entry point running `SFTTrainer` |
| `slam_train/scripts/main.py` | slam-train | Deleted: scaffold placeholder replaced by `train_sft.py` |
| `config/config_train_sft.yaml` | slam-train | New: defaults, split params, searchpath block |
| `config/config_main.yaml` | slam-train | Deleted: scaffold placeholder |
| `config/lora/default.yaml` | slam-train | New: LoRA hyperparameters |
| `config/sft/default.yaml` | slam-train | New: `SFTConfig` training arguments |
| `config/user_settings/user_settings.yaml` | slam-train | Modified: add `dataset_root`, `slam_shared_config` |
| `pyproject.toml` | slam-train | Modified: add `slam-core`, `trl >= 1.12`, `peft >= 0.20`, `transformers`, `torch`, `datasets` |
| `requirements.txt` | slam-train | Modified: same dependency additions |
| `requirements_dev.txt` | slam-train | Modified: add `pytest` |
| `README.md` | slam-train | Modified: document `train_sft.py` and the follow-up `slam-eval` run |
| `tests/test_main.py` | slam-train | Deleted: empty scaffold test |
| `tests/test_sft_dataset.py` | slam-train | New: T1–T15 |
| `tests/test_holdout.py` | slam-train | New: T16–T23 |
| `tests/test_config.py` | slam-train | New: T31–T32 |
| `tests/e2e/test_train_sft.py` | slam-train | New: T34 (marked `slow`) |
| `pyproject.toml` (pytest markers) | slam-train | Modified: register the `slow` marker |


