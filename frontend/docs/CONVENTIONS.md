# Documentation Conventions

## Purpose
This project uses a semi-automatic documentation system for code agents.

## Required Section Template
Module/submodule index files should use these sections in order:
1. `Purpose`
2. `Entry Points`
3. `Key Files`
4. `External Interfaces`
5. `Data/State/Caches`
6. `Common Change Tasks`
7. `Known Risks`

## Marker Contract
AUTO-GEN blocks are managed by `tools/docs_index_builder.py`:

```md
<!-- AUTO-GEN:BEGIN label -->
...
<!-- AUTO-GEN:END -->
```

MANUAL blocks are hand-maintained:

```md
<!-- MANUAL:BEGIN -->
...
<!-- MANUAL:END -->
```

Rule:
- The script may update only AUTO-GEN blocks.
- The script must not rewrite MANUAL blocks.
- Frontend module `Purpose` sections should stay under 6 nonblank lines and 900 characters; move behavior details to focused sections or source-specific docs.

## Naming and Placement
- All docs live under `docs/`.
- Frontend indexes: `docs/ui/`.
- App-server indexes: `docs/app_server/` and `docs/app_server/domains/`.
- Runtime/config indexes: `docs/runtime/`.
- Build indexes: `docs/build/`.
- Generated inventories: `docs/generated/`.
- Contract docs: `docs/contracts/`.
- Architecture guardrails: `docs/architecture/`.

## Priority Contract Docs
Read these before behavior/logic/architecture changes:
1. `docs/contracts/frontend_behavior_contract.md`
2. `docs/contracts/business_logic_contract.md`
3. `docs/architecture/architecture_guardrails.md`

## Update Workflow
1. `python tools/docs_index_builder.py --scaffold-missing`
2. `python tools/docs_index_builder.py --write`
3. `python tools/docs_index_builder.py --check`
