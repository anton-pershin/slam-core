## slam-eval refactor: depend on slam-core

### 1. Requirement analysis

1. `slam-eval` must depend on `slam-core` as a pip-installable package dependency
2. All modules already extracted into `slam-core` must be removed from `slam-eval` and imported from `slam_core` instead:
   - `slam_eval.collections.*` → `slam_core.collections.*`
   - `slam_eval.model` → `slam_core.model`
   - `slam_eval.scorer` (base scorers) → `slam_core.scorers.*`
   - `slam_eval.ifbench.*` → `slam_core.scorers.ifbench.*`
   - `slam_eval.merge_quality` → `slam_core.scorers.merge_quality_scorer.*`
3. `slam-eval` must retain only the evaluation pipeline (`main()`), config loading, result aggregation, and any engine-specific code. Storage must NOT stay in `slam-eval`: `EvalStorageAdapter` is not unique to evaluation (e.g., `slam-monitoring` also uses it), so it is extracted into `slam-core` as `slam_core.storage`
4. All existing tests in `slam-eval` that now cover `slam-core` functionality should be removed (tests already live in `slam-core`)
5. Hydra config structure in `slam-eval` must extend shared configs (from slam-core) via Hydra's `searchpath` plugin with a `file://` path. Duplicated configs (`config/collection/`, `config/model/`, `config/scorer/`) must be removed from slam-eval and resolved via searchpath. Only slam-eval-specific configs remain: `config_main.yaml`, `config/user_settings/`, `config/storage_adapter/`, `config/hydra/`.

### 2. Tests

1. **Collection tests** — remove from `slam-eval` (already in `slam-core/tests/test_collection.py`)
2. **Scorer tests** — remove from `slam-eval` (already in `slam-core/tests/test_scorer.py`)
3. **Model tests** — remove from `slam-eval` (already in `slam-core/tests/test_model.py`)
4. **Storage tests** — move `tests/test_storage_adapter.py` from `slam-eval` to `slam-core/tests/test_storage.py` (storage now lives in `slam-core`)
5. **`tests/e2e/test_main.py`** — move existing `test_main.py` to `tests/e2e/` and update test doubles to inherit from `slam_core` base classes (`slam_core.collections.base.EvalCaseCollection`, `slam_core.scorers.base.Scorer`, etc.)


### 3. Implementation plan

#### 3.1 Implementation repos

- `slam-eval` (anton-pershin/slam-eval) — remove extracted modules (including storage), redirect imports, update configs
- `slam-core` (anton-pershin/slam-core) — receive `EvalStorageAdapter` as `slam_core.storage` (storage is shared with `slam-monitoring`, so it belongs in the core)

**Note:** slam-core is included in the implementation repos for this spec only to receive the storage extraction. All other slam-core changes (config target fixes, spec updates) are handled separately.

#### 3.2. Solution design

[Introduce the main parts of your implementation]

**Key changes in `slam-core`:**

1. **Receive storage** — create `slam_core/storage/` with the `EvalStorageAdapter` base class and `LocalJsonlAdapter`, moved from `slam_eval/storage_adapter.py`. Also receive its unit tests as `tests/test_storage.py`.

**Key changes in `slam-eval`:**

1. **Add `slam-core` dependency** — add `slam-core` to `pyproject.toml` dependencies
5. **Redirect imports** — replace `slam_eval.collections.*` → `slam_core.collections.*`, `slam_eval.model` → `slam_core.model`, `slam_eval.scorer` → `slam_core.scorers.*`, `slam_eval.ifbench.*` → `slam_core.scorers.ifbench.*`, `slam_eval.merge_quality` → `slam_core.scorers.merge_quality_scorer.*`
6. **Remove extracted files** — delete `slam_eval/collections/`, `slam_eval/model.py`, `slam_eval/scorer.py`, `slam_eval/ifbench/`, `slam_eval/merge_quality/` (all now in `slam-core`)
7. **Move and adapt tests** — move `tests/test_main.py` → `tests/e2e/test_main.py`, update test doubles to import from `slam_core`
8. **Update Hydra config** — add `searchpath` entry using `file://` pointing to slam-core's `config/` directory. Remove duplicated config directories (`config/collection/`, `config/model/`, `config/scorer/`) from slam-eval so they resolve via searchpath. Keep only slam-eval-specific configs: `config_main.yaml`, `config/user_settings/`, `config/storage_adapter/`, `config/hydra/`.
9. **Extract storage** — remove `slam_eval/storage_adapter.py` from slam-eval; re-export or re-import `EvalStorageAdapter`/`LocalJsonlAdapter` from `slam_core.storage` where the pipeline uses them. Update `slam_eval/scripts/main.py` imports accordingly.

#### 3.3 Todo list

1. [ ] Write the e2e tests: move `tests/test_main.py` to `tests/e2e/test_main.py`, update test doubles to import from `slam_core`
2. [ ] Run the e2e tests and ensure they fail (imports from `slam_eval.collections`, `slam_eval.model`, `slam_eval.scorer`, `slam_eval.ifbench` are broken because the files don't exist yet in the refactored branch)
3. [ ] Add `slam-core` as a dependency in `pyproject.toml` — use `slam-core @ git+https://github.com/anton-pershin/slam-core.git` with an inline comment explaining the local dev override (`pip install -e ../slam-core`). Add the same hint to `README.md`.
4. [ ] Update all imports in `slam_eval/` source files: redirect to `slam_core.*` for collections, model, scorers, ifbench, merge_quality
5. [ ] Remove extracted files from `slam_eval/`: `collections/`, `model.py`, `scorer.py`, `ifbench/`, `merge_quality/`
6. [ ] Update Hydra config: add `searchpath` entry using `file://${user_settings.shared_config_path}` pointing to slam-core's `config/` directory. Remove duplicated config directories (`config/collection/`, `config/model/`, `config/scorer/`) from slam-eval. Add `shared_config_path` field to `user_settings` template.
7. [ ] Run e2e tests and ensure they pass
8. [ ] (slam-core) Create `slam_core/storage/` module with `EvalStorageAdapter` and `LocalJsonlAdapter` moved from `slam_eval/storage_adapter.py`; add `tests/test_storage.py` (moved from `slam-eval/tests/test_storage_adapter.py`)
9. [ ] (slam-eval) Remove `slam_eval/storage_adapter.py` and `tests/test_storage_adapter.py`; update `slam_eval/scripts/main.py` and e2e test imports to use `slam_core.storage`

#### 3.4 Modification summary

| File | Repo | Action |
|------|------|--------|
| `slam_core/storage/__init__.py` | slam-core | New: re-export `EvalStorageAdapter`, `LocalJsonlAdapter` |
| `slam_core/storage/base.py` | slam-core | New: `EvalStorageAdapter` moved from `slam_eval/storage_adapter.py` |
| `slam_core/storage/local_jsonl.py` | slam-core | New: `LocalJsonlAdapter` moved from `slam_eval/storage_adapter.py` |
| `slam_core/tests/test_storage.py` | slam-core | New: moved from `slam-eval/tests/test_storage_adapter.py` |
| `slam_eval/pyproject.toml` | slam-eval | Modified: add `slam-core @ git+https://github.com/anton-pershin/slam-core.git` dependency |
| `slam_eval/README.md` | slam-eval | Modified: add hint about local dev override (`pip install -e ../slam-core`) |
| `slam_eval/scripts/main.py` | slam-eval | Modified: update imports to use `slam_core.*` (storage, collections, model, scorer) |
| `slam_eval/storage_adapter.py` | slam-eval | Deleted: extracted to `slam-core` as `slam_core.storage` |
| `slam_eval/ifbench/` | slam-eval | Deleted: all files (extracted to `slam-core`) |
| `slam_eval/merge_quality/` | slam-eval | Deleted: all files (extracted to `slam-core`) |
| `slam_eval/collections/` | slam-eval | Deleted: all files (extracted to `slam-core`) |
| `slam_eval/model.py` | slam-eval | Deleted: extracted to `slam-core` |
| `slam_eval/scorer.py` | slam-eval | Deleted: extracted to `slam-core` |
| `tests/test_collection.py` | slam-eval | Deleted: tests now in `slam-core/tests/test_collection.py` |
| `tests/test_scorer.py` | slam-eval | Deleted: tests now in `slam-core/tests/test_scorer.py` |
| `tests/test_model.py` | slam-eval | Deleted: tests now in `slam-core/tests/test_model.py` |
| `tests/test_storage_adapter.py` | slam-eval | Deleted: tests moved to `slam-core/tests/test_storage.py` |
| `tests/test_main.py` | slam-eval | Moved to `tests/e2e/test_main.py`, modified: update test double imports to `slam_core.*` |
| `tests/test_embedding_classifier.py` | slam-eval | Modified: update imports to `slam_core.*` |
| `config/config_main.yaml` | slam-eval | Modified: add `hydra.searchpath` with `file://${user_settings.shared_config_path}` |
| `config/user_settings/user_settings.yaml` | slam-eval | Modified: add `shared_config_path` field |
| `config/collection/` | slam-eval | Deleted: resolved via searchpath from slam-core |
| `config/model/` | slam-eval | Deleted: resolved via searchpath from slam-core |
| `config/scorer/` | slam-eval | Deleted: resolved via searchpath from slam-core |
