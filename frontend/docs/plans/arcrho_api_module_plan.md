ArcRho Custom Python Module (`arcrho`) Subproject Plan
Version: v0.2
Last updated: 2026-02-17

---

1. Subproject Goals

Provide a business module that can be imported directly from the Scripting Console:

```python
import arcrho
```

The goal is to give business users a stable, easy-to-use, object-oriented API and reduce direct dependency on lower-level file structures, HTTP routes, and internal data formats.

Core goals:
1. Provide a unified entry point: `arcrho.application`.
2. Provide an object-oriented access path from Project -> ReservingClass -> Dataset/Dfm.
3. Support common read and write operations, starting with read support and landing write support in phases.
4. Preserve compatibility with existing scripting capabilities and avoid breaking existing `/scripting/*` behavior.

---

2. Constraints and Design Principles

Required constraints:
1. The root directory must come from `workspace_paths.json` / `app_server.config`; do not hard-code `E:\\ArcRho`.
2. Preserve the `router -> service -> config/schema` layering; do not put business logic into routers.
3. External interfaces are backward-compatible by default. Add new capabilities incrementally instead of directly refactoring old behavior.
4. Preserve scripting session isolation semantics. Variables from different sessions must not leak into each other.
5. Errors must be readable and distinguish parameter errors, missing resources, permission restrictions, and runtime errors.

Recommended strategy:
1. `arcrho` should primarily wrap existing service-layer capabilities and should not directly couple to frontend page logic.
2. Start write operations with the smallest complete workflow, then expand gradually to avoid a large one-shot change.

---

3. Target API Draft (MVP)

3.1 Top-Level Entry Point

```python
import arcrho

arcrho.application.version
arcrho.application.root
arcrho.projects()
arcrho.project("my_project")
```

3.2 Object Hierarchy

```text
Application
  └─ Project
      └─ ReservingClass
          ├─ Dataset
          └─ Dfm
```

3.3 Objects and Methods (Initial Recommendation)

`Application`
1. `version: str`
2. `root: str`
3. `projects() -> list[Project]`
4. `project(name: str) -> Project`

`Project`
1. `name: str`
2. `folder: str`
3. `reserving_class(path: str) -> ReservingClass`
4. `reserving_classes() -> list[ReservingClass]` (optional)

`ReservingClass`
1. `path: str`
2. `dataset(name_or_id: str) -> Dataset`
3. `datasets() -> list[Dataset]` (optional)
4. `dfm(name: str) -> Dfm`
5. `DFM(name: str) -> Dfm` (compatibility alias; not recommended for new code)

`Dfm`
1. `name: str`
2. `data.values(row: int, col: int) -> Any`
3. `ratios.values(row: int | None = None, col: int | None = None) -> Any`
4. `ratios.selected(col: int) -> Any`
5. `ratios.set_selected(col: int, avg_formula_name: str) -> None`
6. `ratios.set_user_entry(col: int, value: float) -> None`

---

4. Example (Public Documentation Wording)

```python
import arcrho

print(arcrho.application.version)
print(arcrho.application.root)

project = arcrho.project("my_project")
print(project.name, project.folder)

reserving_class = project.reserving_class("my_path")
dfm = reserving_class.dfm("my_dfm")

value_1 = dfm.data.values(1, 2)
value_2 = dfm.ratios.values(row=3, col=4)
value_3 = dfm.ratios.selected(col=1)

dfm.ratios.set_selected(col=1, avg_formula_name="simple_3")
dfm.ratios.set_user_entry(col=1, value=1.2345)
```

---

5. Implementation Scope and Non-Scope

5.1 In Scope
1. Make `import arcrho` available in the scripting execution environment.
2. Complete the minimal `Application/Project/ReservingClass/Dfm` object model.
3. Support core read capabilities: root, projects, object lookup, and basic `values` / `selected` access.
4. Support minimal write capabilities: `set_selected` and `set_user_entry`.
5. Provide an API help documentation entry point, which may connect to `/scripting/api-help`.

5.2 Out of Scope
1. Covering all advanced Dfm editing capabilities in one pass.
2. Providing complex rich-output rendering in this phase, such as images, HTML, or interactive controls.
3. Refactoring existing scripting route protocols or session mechanisms.

---

6. Phased Plan and Milestones

Phase 0: Requirement Freeze and Technical Design (0.5-1 day)
1. Confirm object names, method signatures, and error semantics.
2. Confirm which write operations are included in the first phase.
3. Deliverable: interface checklist and acceptance checklist.

Phase 1: Module Skeleton and Read-Only Capabilities (1-2 days)
1. Create the `arcrho` package structure and entry point.
2. Implement `application.root/version/projects/project(...)`.
3. Connect to the scripting session namespace while preserving multi-window session isolation.
4. Deliverable: runnable read-only demo.

Phase 2: Minimal Dfm Read/Write Workflow (2-3 days)
1. Implement `Dfm.data.values` and `ratios.values/selected`.
2. Implement `ratios.set_selected` and `ratios.set_user_entry`.
3. Complete exception classification for parameter errors, missing resources, permission errors, and runtime errors.
4. Deliverable: end-to-end verifiable read/write workflow.

Phase 3: Documentation, Tests, and Release (1-2 days)
1. Add unit tests, integration tests, and a manual regression script.
2. Add API help documentation and an example notebook.
3. Add release notes covering the new API, known limitations, and rollback path.

---

7. Task Breakdown (Execution Checklist)

App Server
1. Add the `arcrho` module implementation, preferably near `app_server/services` or as an independent package.
2. Wrap data reads and writes through the service layer without bypassing config rules.
3. Add any necessary schemas if dedicated new interfaces are introduced.

Scripting
1. Inject `arcrho` import support during session initialization.
2. Keep existing `/scripting/run`, `/scripting/run-stream`, and `/scripting/interrupt` behavior compatible.
3. Expose common method documentation for the new module through `/scripting/api-help`.

Docs
1. Update related backend/frontend MANUAL documentation if behavior or contracts are affected.
2. Add `arcrho_api_module` examples and error documentation.
3. Run the documentation index build and check.

QA
1. Add a minimal regression script covering import, read, write, exceptions, and concurrent sessions.
2. Cover path-change scenarios, including whether changes to `workspace_paths.workspace_root` take effect.
3. Cover permission scenarios, including error messages for disallowed write directories.

---

8. Acceptance Criteria (Definition of Done)

Functional acceptance:
1. `import arcrho` succeeds in the scripting console.
2. `arcrho.application.root` is consistent with the `workspace_paths` configuration.
3. `arcrho.projects()` returns a list of project objects and does not depend on hard-coded paths.
4. `Dfm` read and write methods work on the target sample project.

Compatibility acceptance:
1. Existing scripting notebook execution, save, and load behavior has no regression.
2. Existing `/scripting/*` route contracts are not broken.
3. Existing session isolation semantics remain unchanged.

Quality acceptance:
1. Key APIs have unit tests and failure-scenario tests.
2. Error messages are readable and identify the relevant object, field, or parameter.
3. Documentation examples can run directly.

---

9. Risks and Mitigations

Risk 1: Path logic is hard-coded, causing environment migration failures.
Mitigation: Resolve root paths uniformly through `app_server.config` and add regression coverage for path changes.

Risk 2: The object API does not match existing service semantics.
Mitigation: Define a service adapter layer first and avoid direct file read/write assembly in the object layer.

Risk 3: Write operations break data consistency.
Mitigation: Start with the smallest write scope and use temporary files plus atomic replacement where needed.

Risk 4: Cross-session contamination through shared variables or state.
Mitigation: Reuse the existing `session_id` isolation mechanism strictly and avoid global mutable singleton caches for business objects.

---

10. Rollback Strategy

1. Keep the original scripting helper API and do not remove legacy entry points.
2. Allow the new `arcrho` functionality to be disabled through configuration, preferably via a new feature flag.
3. If issues appear after release, roll back the `arcrho` injection step first without affecting the base scripting execution path.

---

11. Open Questions to Confirm Before Implementation

1. Authoritative data source for `Dfm`
- Decision: Use persisted data files as the authoritative source. Caches are only for faster reads.
- Behavior: If the target `Dfm` does not exist during read, return a clear "not found" error instead of `None`.
- Creation: Support `add_dfm()` to create an in-memory object and persist it after `save()` is called.

2. Persistence semantics for `set_selected` and `set_user_entry`
- Decision: Both methods only modify the in-memory object and mark it dirty; they do not immediately write to disk.
- Persistence: Use explicit `save()` to atomically write local JSON, including locking and concurrency protection.

3. First-phase `Dataset` API scope
- Decision: Include a minimal read-only `Dataset` API in the first phase: listing, basic metadata, and coordinate-based value lookup.
- Note: `Dataset` write APIs are not included in the first phase. Prioritize completing the `Dfm` workflow first.
