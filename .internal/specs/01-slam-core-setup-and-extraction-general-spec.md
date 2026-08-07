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
2. `slam-core` must follow the Hydra repo template: `config/`, `pyproject.toml`, `requirements.txt`, `requirements_dev.txt`, `tests/`, `run_linters.sh`, and the `slam_core/` source directory
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

[List alternative ideas describing what should be implemented to satisfy these requirements. Describe at least two ideas and explain why you choose a particular idea]

### 5. Overall solution design

#### 5.1 High-level design

[Mermaid diagram (typically, flowchart, but it is ultimately up to a planner) describing the high-level design]

#### 5.2 Core components

[List all the core components of the solution]

### 6. Implementation plan

#### 6.1 Todo list

[Write a todo list with all the steps necessary to create an implementation]

#### 6.2 Modification summary

[Fill the table below specifying which files are going to be modified and which are going to be created]

| File | Action |
|------|--------|
| ... | Modified: add X, modify Y, etc. |
| ... | New |
