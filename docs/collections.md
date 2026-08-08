# Collections

A collection represents a set of evaluation examples used to test and benchmark models. Each collection is an iterable of `EvalCase` items — each item contains an input (`x`) and a ground truth (`y_true`).

## Core abstractions

### `EvalCase`

A `TypedDict` with two keys:

- `x` — the input to the model (type varies by collection)
- `y_true` — the expected output (type varies by collection)

### `EvalCaseCollection`

An abstract base class that all collections inherit.

```python
from slam_core.collections.base import EvalCaseCollection

class MyCollection(EvalCaseCollection):
    def __init__(self, name: str):
        super().__init__(name)

    def _load(self) -> CollectionInfo:
        # Return an iterator and its length
        ...

    def __next__(self) -> EvalCase:
        # Return the next EvalCase
        ...
```

**Lifecycle:**

1. Construct the collection (lazy — no data loaded yet).
2. Call `.load()` to materialise the underlying iterator.
3. Iterate with `for case in collection:` or `next(collection)`.
4. Calling `len()` or `next()` before `.load()` raises `CollectionNotLoadedError`.

### `CollectionInfo`

A `TypedDict` returned by `_load()`:

- `collection` — an `Iterator` over items.
- `collection_len` — the total number of items.

---

## Text generation collections

### `BigBenchHard`

Loads a Big-Bench Hard subset from the Hugging Face Hub via `datasets`.

```python
from slam_core.collections.text_generation import BigBenchHard

collection = BigBenchHard(
    name="dyck",
    dataset_name="lighteval/big_bench_hard",
    split="test",
    subset="dyck_languages",
    user_prompt_template="Parse the following: {original_input}",
)

collection.load()
for case in collection:
    prompt = case["x"]["user_prompt"]
    target = case["y_true"]
```

### `MergeQuality`

Loads evaluation examples from a local JSONL file.

```python
from slam_core.collections.text_generation import MergeQuality

collection = MergeQuality(
    name="merge_quality_test",
    jsonl_path="path/to/examples.jsonl",
    user_prompt_template="Identifiers: {unique_identifiers}\nData: {data_chunks}",
)
```

The JSONL format expects each line to have:

```json
{
  "ground_truth": {
    "attributes": { ... },
    "provided_identifiers": { ... }
  },
  "provided_identifiers": { ... },
  "chunks": [
    {"content": "chunk text 1"},
    {"content": "chunk text 2"}
  ]
}
```

### `TextGenerationInput`

A `TypedDict` used as the `x` field in text generation cases:

- `system_prompt` — optional system message (`str | None`).
- `user_prompt` — the user message (`str`).

---

## IFBench collection

### `IFBench`

Loads IFBench instruction-following examples from a local JSONL file. Downloads from the given URL if the file does not exist.

```python
from slam_core.collections.ifbench import IFBench

collection = IFBench(
    name="ifbench",
    jsonl_path="/path/to/ifbench.jsonl",
    download_url="https://example.com/ifbench.jsonl",
)
```

Each case yields a `TextGenerationWithUniqueGroundTruth` where `y_true` contains:

- `instruction_id_list` — list of instruction IDs to check.
- `kwargs` — per-instruction kwargs for the checker.

---

## Text classification collection

### `RedditPostsSmallDataset`

Loads messages from a `MessageProvider` (from `kygs`) for text classification.

```python
from slam_core.collections.text_classification import RedditPostsSmallDataset
from kygs.message_provider import MessageProvider

provider = MessageProvider(...)
collection = RedditPostsSmallDataset(name="reddit", message_provider=provider)
```

Each case yields `TextClassificationWithUniqueGroundTruth`:

- `x` — the message text (`str`).
- `y_true` — the label (`str`), taken from `true_label` or `label`.
