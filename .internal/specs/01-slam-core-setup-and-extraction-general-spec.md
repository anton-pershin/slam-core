## [Spec name]

### 1. Executive summary

#### 1.1 Spec description 

Set up `slam-core` as an installable Hydra-based Python package and extract the core abstractions (`EvalCaseCollection`, `Model`, `Scorer`, and training mixins) from `slam-eval` into it, so that all ecosystem repos can depend on a single shared source of truth.

#### 1.2 Spec motivation

The constitution spec (Step 6.1 and 6.2) requires `slam-core` to be an installable package with shared abstractions. Currently these classes live only in `slam-eval`, causing duplication and making it impossible for other repos to import them without depending on the evaluation library.

#### 1.3 Implementation repos

- `slam-core` (management repo and target package; classes will be copied from `slam-eval` as reference)

### 2. Requirement analysis
#### 2.1 Functional requirements

1. `slam-core` must be installable as a Python package via `pip install -e .`
2. `slam-core` must follow the Hydra repo template: `config/`, `docs/`, `pyproject.toml`, `requirements.txt`, `requirements_dev.txt`, `tests/`, `run_linters.sh`, and the `slam_core/` source directory
3. `slam-core` must include the `collections`, `model`, and `scorer` modules taken from `slam-eval`
4. The extracted classes must be importable as `from slam_core.collections import EvalCaseCollection`, etc.
5. `slam-core` must include `config/user_settings/user_settings.yaml` so that downstream repos can extend the config tree via Hydra's `searchpath` plugin (`pkg://slam_core.config`)

#### 2.2 Non-functional requirements

1. No executable scripts in `slam-core` (it is a library, not a tool)
2. The package structure must follow the Hydra repo convention from the template
3. All code must pass the linters defined in `run_linters.sh`
4. Unit tests must verify that extracted classes are importable and functional

### 3. Acceptance criteria

[Describe how we will check whether the implementation satisfy the requirements. In the case of software projects, it is done via tests]

### 4. Insight

**Option 1: Copy classes verbatim.** Take `collections.py`, `model.py`, `scorer.py` from `slam-eval` and paste them into `slam-core` as-is.
- **Pros:** Fastest path, minimal risk of introducing bugs, preserves existing functionality.
- **Cons:** May include `slam-eval`-specific imports or dependencies that don't belong in a shared core.

**Option 2: Refactor during extraction.** Clean up the classes while moving them, removing `slam-eval`-specific code, adding proper abstractions.
- **Pros:** Cleaner code in `slam-core`, better separation of concerns.
- **Cons:** Higher risk of breaking existing functionality, requires careful diffing and testing.

**Decision:** Option 1 for now. Copy the classes verbatim first, then clean up any `slam-eval`-specific dependencies within this same spec. This minimizes risk and keeps the extraction straightforward.

### 5. Overall solution design

#### 5.1 High-level design

This is a straightforward package setup and extraction — no complex architecture needed. The flow is: copy files from `slam-eval` → fix imports → package with `pyproject.toml` → verify with tests.

#### 5.2 Core components

1. `slam_core/collections.py` — `EvalCaseCollection` base class (from `slam-eval`)
2. `slam_core/model.py` — `Model` base class (from `slam-eval`)
3. `slam_core/scorer.py` — `Scorer` base class (from `slam-eval`)
4. `slam-core/config/` — Hydra config directory with `user_settings/`
5. `slam-core/docs/` — Documentation directory

### 6. Implementation plan

#### 6.1 Todo list

1. [ ] Create package structure: `slam_core/`, `config/`, `docs/`, `tests/`, `pyproject.toml`, `requirements.txt`, `requirements_dev.txt`, `run_linters.sh`
2. [ ] Copy `collections.py`, `model.py`, `scorer.py` from `slam-eval` into `slam_core/`
3. [ ] Fix imports in the copied files to remove `slam-eval` references
4. [ ] Create `config/user_settings/user_settings.yaml` (based on `slam-eval` example)
5. [ ] Write tests for class imports and basic functionality
6. [ ] Fill `docs/` with package documentation
7. [ ] Run linters and fix any issues
8. [ ] Verify `pip install -e .` works

#### 6.2 Modification summary

| File | Action |
|------|--------|
| `slam-core/pyproject.toml` | New |
| `slam-core/requirements.txt` | New |
| `slam-core/requirements_dev.txt` | New |
| `slam-core/run_linters.sh` | New |
| `slam-core/slam_core/__init__.py` | New |
| `slam-core/slam_core/collections.py` | New (from `slam-eval`) |
| `slam-core/slam_core/model.py` | New (from `slam-eval`) |
| `slam-core/slam_core/scorer.py` | New (from `slam-eval`) |
| `slam-core/config/user_settings/user_settings.yaml` | New |
| `slam-core/docs/` | New |
| `slam-core/tests/__init__.py` | New |
| `slam-core/tests/test_imports.py` | New |
