## slam-eval refactor: depend on slam-core

### 1. Requirement analysis

1. `slam-eval` must depend on `slam-core` as a pip-installable package dependency
2. All modules already extracted into `slam-core` must be removed from `slam-eval` and imported from `slam_core` instead:
   - `slam_eval.collections.*` → `slam_core.collections.*`
   - `slam_eval.model` → `slam_core.model`
   - `slam_eval.scorer` (base scorers) → `slam_core.scorers.*`
   - `slam_eval.ifbench.*` → `slam_core.scorers.ifbench.*`
3. `slam-eval` must retain only the logic not yet moved to `slam-core`: the evaluation pipeline (`main()`), config loading, result aggregation, and any engine-specific code
4. All existing tests in `slam-eval` that now cover `slam-core` functionality should be removed (tests already live in `slam-core`)
5. Hydra config structure in `slam-eval` must extend `slam-core` defaults via `searchpath` (`pkg://slam_core.config`)

### 2. Tests

1. **Collection tests** — remove from `slam-eval` (already in `slam-core/tests/test_collection.py`)
2. **Scorer tests** — remove from `slam-eval` (already in `slam-core/tests/test_scorer.py`)
3. **Model tests** — remove from `slam-eval` (already in `slam-core/tests/test_model.py`)
4. **`tests/e2e/test_main.py`** — move existing `test_main.py` to `tests/e2e/` and update test doubles to inherit from `slam_core` base classes (`slam_core.collections.base.EvalCaseCollection`, `slam_core.scorers.base.Scorer`, etc.)



### 3. Implementation plan

#### 3.1 Implementation repos

- `slam-eval` (anton-pershin/slam-eval) — the only implementation repo for this KISS spec

#### 3.2. Solution design

[Introduce the main parts of your implementation]

**Key changes in `slam-eval`:**

1. **Add `slam-core` dependency** — add `slam-core` to `pyproject.toml` dependencies
2. **Redirect imports** — replace `slam_eval.collections.*` → `slam_core.collections.*`, `slam_eval.model` → `slam_core.model`, `slam_eval.scorer` → `slam_core.scorers.*`, `slam_eval.ifbench.*` → `slam_core.scorers.ifbench.*`
3. **Remove extracted files** — delete `slam_eval/collections/`, `slam_eval/model.py`, `slam_eval/scorer.py`, `slam_eval/ifbench/` (all now in `slam-core`)
4. **Move and adapt tests** — move `tests/test_main.py` → `tests/e2e/test_main.py`, update test doubles to import from `slam_core`
5. **Update Hydra config** — add `searchpath` entry to extend `slam-core` defaults via `pkg://slam_core.config`

#### 3.3 Todo list

1. [ ] Write the e2e tests: move `tests/test_main.py` to `tests/e2e/test_main.py`, update test doubles to import from `slam_core`
2. [ ] Run the e2e tests and ensure they fail (imports from `slam_eval.collections`, `slam_eval.model`, `slam_eval.scorer`, `slam_eval.ifbench` are broken because the files don't exist yet in the refactored branch)
3. [ ] Add `slam-core` as a dependency in `pyproject.toml` — use `slam-core @ git+https://github.com/anton-pershin/slam-core.git` with an inline comment explaining the local dev override (`pip install -e ../slam-core`). Add the same hint to `README.md`.
4. [ ] Update all imports in `slam_eval/` source files: redirect to `slam_core.*` for collections, model, scorers, ifbench
5. [ ] Remove extracted files from `slam_eval/`: `collections/`, `model.py`, `scorer.py`, `ifbench/`
6. [ ] Update Hydra config to extend `slam-core` defaults via `searchpath`
7. [ ] Run e2e tests and ensure they pass

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `slam_eval/pyproject.toml` | Modified: add `slam-core @ git+https://github.com/anton-pershin/slam-core.git` dependency |
| `slam_eval/README.md` | Modified: add hint about local dev override (`pip install -e ../slam-core`) |
| `slam_eval/scripts/main.py` | Modified: update imports to use `slam_core.*` (if any reference collections/model/scorer) |
| `slam_eval/storage_adapter.py` | Modified: replace `from slam_eval.collections.base import ...` with `from slam_core.collections.base import ...`, replace `from slam_eval.model import Model` with `from slam_core.model import Model`, replace `from slam_eval.scorer import Score` with `from slam_core.scorers.base import Score` |
| `slam_eval/ifbench/__init__.py` | Modified: replace imports to `slam_core.scorers.ifbench.*` |
| `slam_eval/ifbench/scorer.py` | Modified: replace `from slam_eval.ifbench.checker_factory` → `from slam_core.scorers.ifbench.checker_factory`, `from slam_eval.scorer` → `from slam_core.scorers.base` |
| `slam_eval/ifbench/checkers.py` | Modified: replace imports to `slam_core.scorers.ifbench.*` |
| `slam_eval/ifbench/instructions_registry.py` | Modified: replace import to `slam_core.scorers.ifbench.third_party.instructions` |
| `slam_eval/collections/` | Deleted: all files (extracted to `slam-core`) |
| `slam_eval/model.py` | Deleted: extracted to `slam-core` |
| `slam_eval/scorer.py` | Deleted: extracted to `slam-core` |
| `slam_eval/ifbench/` | Deleted: all files (extracted to `slam-core`) |
| `tests/test_collection.py` | Deleted: tests now in `slam-core/tests/test_collection.py` |
| `tests/test_scorer.py` | Deleted: tests now in `slam-core/tests/test_scorer.py` |
| `tests/test_model.py` | Deleted: tests now in `slam-core/tests/test_model.py` |
| `tests/test_main.py` | Moved to `tests/e2e/test_main.py`, modified: update test double imports to `slam_core.*` |
| `tests/test_embedding_classifier.py` | Modified: update imports to `slam_core.*` |
| `tests/test_storage_adapter.py` | Modified: update imports to `slam_core.*` |
| `config/` | Modified: add `searchpath` entry for `pkg://slam_core.config` in Hydra defaults |
