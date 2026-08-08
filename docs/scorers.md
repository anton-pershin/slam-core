# Scorers

Scorers compare a model's prediction against the ground truth and return a `Score` object with a primary metric value (0.0–1.0) and optional sub-scores.

## Core abstraction

### `Score`

A frozen dataclass:

- `primary` — the main score (`float`, typically 0.0–1.0).
- `sub_scores` — an optional `dict[str, float]` of named sub-metrics.

### `Scorer`

An abstract base class. Every scorer implements `__call__(y_true, y_pred) -> Score`.

```python
from slam_core.scorers.base import Scorer, Score

class MyScorer(Scorer):
    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, y_true: Any, y_pred: Any) -> Score:
        ...
```

---

## Built-in scorers

### `ExactMatch`

Returns 1.0 if `y_true` and `y_pred` are equal after optional preprocessing, 0.0 otherwise.

```python
from slam_core.scorers.base import ExactMatch

scorer = ExactMatch("em")
scorer("hello", "hello")   # Score(primary=1.0)
scorer("hello", "world")   # Score(primary=0.0)
```

**Preprocessing:** pass a callable to transform values before comparison.

```python
from slam_core.scorers.base import build_json_string_to_dict

scorer = ExactMatch("em", preprocessing_func=build_json_string_to_dict())
scorer('{"a": 1}', {"a": 1})  # Score(primary=1.0)
```

The `json_string_to_dict` helper parses JSON strings (stripping markdown code fences) or returns the original value if parsing fails.

### `IgnoreAllWhitespaces`

Returns 1.0 if the non-whitespace characters in `y_pred` match those in `y_true` in the same order, 0.0 otherwise.

```python
from slam_core.scorers.base import IgnoreAllWhitespaces

scorer = IgnoreAllWhitespaces("ws")
scorer("hello world", "helloworld")     # Score(primary=1.0)
scorer("hello world", "hello\nworld")   # Score(primary=1.0)
scorer("hello", "world")                # Score(primary=0.0)
```

Useful when the model output may have formatting differences (extra spaces, newlines, tabs) but the same non-whitespace content.

---

## MergeQuality scorer

### `MergeQualityScorer`

Scores structured predictions (dict outputs) by comparing ground truth against a model's predicted JSON. Returns a composite score based on multiple metrics.

```python
from slam_core.scorers.merge_quality_scorer import MergeQualityScorer

scorer = MergeQualityScorer("merge_quality")
gt = {"title": "Test", "body": "Content"}
pred = '{"title": "Test", "body": "Content"}'
score = scorer(gt, pred)
# Score(primary=0.95, sub_scores={"f1": ..., "precision": ..., "recall": ...,
#          "completeness": ..., "hallucination": ..., "structure_similarity": ...})
```

The composite score is calculated as:

```
total_score = 0.4 * f1
            + 0.2 * structure_similarity
            + 0.25 * completeness
            + 0.15 * hallucination_penalty
```

| Sub-score | Description |
|-----------|-------------|
| `f1` | Harmonic mean of precision and recall on leaf values |
| `precision` | Fraction of predicted keys that match ground truth |
| `recall` | Fraction of ground truth keys that are predicted |
| `completeness` | Overlap of predicted keys with ground truth keys |
| `hallucination` | Fraction of predicted keys not in ground truth |
| `structure_similarity` | Jaccard similarity of nested structure paths |

---

## IFBench scorer

### `IFBenchScorer`

Scores instruction-following responses against a list of constraints. Each constraint is checked by a factory-produced checker.

```python
from slam_core.scorers.ifbench.checker_factory import IFBenchCheckerFactory
from slam_core.scorers.ifbench.scorer import IFBenchScorer

factory = IFBenchCheckerFactory()
factory.register("has_code", lambda: checker_instance)

scorer = IFBenchScorer(name="ifbench", checker_factory=factory)

y_true = {
    "instruction_id_list": ["has_code", "no_apology"],
    "kwargs": [{}, {}],
}
y_pred = "Here is the solution: ..."
score = scorer(y_true, y_pred)
# Score(primary=0.5, sub_scores={"has_code": 1.0, "no_apology": 0.0})
```

The scorer averages results across all instructions, returning per-instruction sub-scores. It also caches `build_description` signatures to efficiently filter kwargs.
