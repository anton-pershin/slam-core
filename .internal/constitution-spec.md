## Slam ecosystem constitution spec

### 1. Executive summary

#### 1.1 Project description 

The `slam` ecosystem is a suite of modular, composable libraries for the full lifecycle of foundational models (LLMs, vision, time series, etc.): data generation (`slam-datagen`), supervised training (`slam-train`), reinforcement learning (`slam-rl`), evaluation (`slam-eval`), and experiment monitoring (`slam-monitoring`). All repos share a common core (`slam-core`) that provides unified abstractions and training extensions, ensuring seamless data flow and consistent contracts across the entire pipeline.

#### 1.2 Project motivation

The goal is to provide a simple, controllable, and flexible ecosystem for all research projects related to foundational models, suitable for both educational and research purposes.

### 2. Requirement analysis

#### 2.1 Functional requirements

1. **Data generation**: Produce structured datasets for training and evaluation of various model types (LLMs, vision, time series, etc.)
2. **Model training**: Support supervised fine-tuning and reinforcement learning with pluggable backends
3. **Model evaluation**: Evaluate models on any dataset using configurable scorers and configurable inference engines
4. **Monitoring**: Aggregate training and evaluation results into structured reports with pluggable frontends
5. **Shared core**: Provide unified abstractions and training extensions that all repos depend on, eliminating duplication

#### 2.2 Non-functional requirements

1. **Simplicity**: Easy to understand, set up, and extend
2. **Flexibility**: Pluggable engines (e.g., swap `trl` for `unsloth` or custom loops via config), support for diverse model types
3. **Consistent configuration**: All repos use Hydra with a unified config structure
4. **Modularity**: Each repo is independently maintainable; adding a new component doesn't require changing existing ones
5. **Spec-driven development**: All repos follow SDD for project management

### 3. Acceptance criteria

1. **Shared dependency**: `slam-core` is installable as a package dependency by all ecosystem repos (`slam-datagen`, `slam-eval`, `slam-monitoring`, `slam-train`, `slam-rl`)
2. **End-to-end data flow**: A dataset produced by `slam-datagen` can be consumed by both `slam-train` for training and `slam-eval` for evaluation without manual conversion
3. **Train-eval handoff**: A model trained by `slam-train` can be loaded and evaluated by `slam-eval` using the same collection and scorer definitions
4. **Monitoring integration**: Evaluation results from `slam-eval` and training logs from `slam-train` can be ingested by `slam-monitoring` to produce reports
5. **Config consistency**: All repos use Hydra with a compatible config structure (shared defaults tree, consistent user settings pattern)
6. **Engine pluggability**: The evaluation and training pipelines allow swapping the backend engine via a single config change (e.g., `engine: trl` → `engine: unsloth`)
7. **SDD compliance**: All repos follow the SDD methodology with constitution and general/kiss specs in `.internal/`

### 4. Insight

**Option 1: Centralized monorepo.** Keep all components (`datagen`, `train`, `rl`, `eval`, `monitoring`) in a single repository with subdirectories. 
- Pros: Simple dependency management, easy cross-component changes, single spec location.
- Cons: Bloated repo, hard to maintain independent versioning, students must navigate a large codebase.

**Option 2: Independent repos with shared core.** Each component is a separate repository depending on `slam-core` for shared abstractions.
- Pros: Clean separation of concerns, independent versioning, focused scope for each student, reusable core.
- Cons: Requires package management across repos, more repos to maintain.

**Option 3: Independent repos with no shared core.** Each component reimplements its own abstractions.
- Pros: Maximum independence, no cross-repo dependencies.
- Cons: Duplication, divergent interfaces, brittle integration between training and evaluation.

**Decision:** Option 2. The modularity and separation benefits far outweigh the overhead of managing a shared package. The `slam-core` repo serves as both the shared library and the SDD management repo for the ecosystem.

### 5. Overall solution design

#### 5.1 High-level design

```mermaid
flowchart LR
    subgraph "Management & Core"
        A[slam-core]
    end

    subgraph "Pipeline"
        B[slam-datagen]
        C[slam-train]
        D[slam-rl]
        E[slam-eval]
        F[slam-monitoring]
    end

    A -.->|shared abstractions| B
    A -.->|shared abstractions| C
    A -.->|shared abstractions| D
    A -.->|shared abstractions| E
    A -.->|shared abstractions| F

    B ==>|datasets| C
    B ==>|datasets| E
    C ==>|trained model| E
    D ==>|trained model| E
    C ==>|logs| F
    E ==>|results| F
```

#### 5.2 Core components

1. **`slam-core` package** — Shared abstractions:
   - `EvalCaseCollection`: Dataset abstraction used by both training and evaluation
   - `Model`: Inference interface, extended via mixins for training (`CausalLMMixin`, `ClassificationMixin`, etc.)
   - `Scorer`: Metric computation interface, reused for training validation and evaluation
2. **`slam-core` management layer** — SDD specs in `.internal/`, constitution spec governing the ecosystem, general/kiss specs for individual components
3. **Hydra config structure** — Unified config tree with shared defaults, consistent `user_settings` pattern across all repos
4. **Engine registry pattern** — Pluggable backend abstraction (training engines, inference engines, monitoring frontends)

### 6. Implementation plan

#### 6.1 Todo list

1. [ ] Set up `slam-core` as an installable Python package with `pyproject.toml`
2. [ ] Extract `EvalCaseCollection` base class from `slam-eval` and move to `slam-core`
3. [ ] Extract `Model` base class and `Scorer` base class from `slam-eval` and move to `slam-core`
4. [ ] Add training mixins to `slam-core` (`CausalLMMixin`, `ClassificationMixin`, `RegressionMixin`)
5. [ ] Update `slam-eval` to depend on `slam-core` and import shared classes
6. [ ] Update `slam-datagen` to depend on `slam-core` and use `EvalCaseCollection`
7. [ ] Update `slam-monitoring` to depend on `slam-core`
8. [ ] Create `slam-train` repo with Hydra config structure and engine registry skeleton
9. [ ] Establish shared Hydra defaults and user_settings pattern across all repos
10. [ ] Add constitution specs to each implementation repo (or use `slam-core` as central management)
11. [ ] Verify end-to-end data flow: datagen → train → eval → monitoring
