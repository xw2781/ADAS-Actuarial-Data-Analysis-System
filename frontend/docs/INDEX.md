# ArcRho Documentation Index

## Purpose
<!-- MANUAL:BEGIN -->
This is the top-level navigation hub for code agents.

System map:
- Electron host/runtime: `electron/main.js`, `electron/preload.js`, `app_shell.py`.
- Frontend pages/features: shell + dataset + DFM + workflow + project settings + pop-out.
- App-server API: FastAPI app in `app_server/main.py` with domain routers in `app_server/api`.
- Runtime/config state: path resolution and cache constants in `app_server/config.py`.
<!-- MANUAL:END -->

## Entry Points
<!-- MANUAL:BEGIN -->
| Question | Where to start |
| --- | --- |
| Add or modify an app-server API endpoint | [`app_server/INDEX.md`](app_server/INDEX.md) |
| Trace a page to app-server endpoints | [`frontend/INDEX.md`](frontend/INDEX.md) |
| Update path/config behavior | [`runtime/config_paths.md`](runtime/config_paths.md) |
| Troubleshoot packaging/build | [`build/packaging.md`](build/packaging.md) |
| Check mandatory behavior/logic/architecture rules | [`contracts/frontend_behavior_contract.md`](contracts/frontend_behavior_contract.md), [`contracts/business_logic_contract.md`](contracts/business_logic_contract.md), [`architecture/architecture_guardrails.md`](architecture/architecture_guardrails.md) |
| Inspect machine-generated inventories | [`generated/app_server_routes.md`](generated/app_server_routes.md), [`generated/frontend_entrypoints.md`](generated/frontend_entrypoints.md), [`generated/file_manifest.md`](generated/file_manifest.md) |
<!-- MANUAL:END -->

## Key Files
<!-- AUTO-GEN:BEGIN root.key_files -->
- [`docs/frontend/INDEX.md`](frontend/INDEX.md) - Frontend module index.
- [`docs/app_server/INDEX.md`](app_server/INDEX.md) - App-server domain index.
- [`docs/runtime/config_paths.md`](runtime/config_paths.md) - Runtime config and path index.
- [`docs/runtime/data_cache_files.md`](runtime/data_cache_files.md) - Runtime cache/data file index.
- [`docs/build/packaging.md`](build/packaging.md) - Build and packaging index.
- [`docs/generated/app_server_routes.md`](generated/app_server_routes.md) - Generated route inventory.
- [`docs/generated/frontend_entrypoints.md`](generated/frontend_entrypoints.md) - Generated frontend entrypoint inventory.
- [`docs/generated/file_manifest.md`](generated/file_manifest.md) - Generated repository file manifest.
<!-- AUTO-GEN:END -->

## Non-Negotiable Contracts
<!-- MANUAL:BEGIN -->
Read and follow these before making behavior or architecture changes:
1. [`contracts/frontend_behavior_contract.md`](contracts/frontend_behavior_contract.md)
2. [`contracts/business_logic_contract.md`](contracts/business_logic_contract.md)
3. [`architecture/architecture_guardrails.md`](architecture/architecture_guardrails.md)

Execution rule:
- If a change affects these contracts, update the relevant MANUAL doc sections in the same change.
<!-- MANUAL:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
Tooling interfaces introduced by this documentation system:
- CLI: `python tools/docs_index_builder.py --scaffold-missing|--write|--check`
- Marker contract:
  - `<!-- AUTO-GEN:BEGIN ... --> ... <!-- AUTO-GEN:END -->`
  - `<!-- MANUAL:BEGIN --> ... <!-- MANUAL:END -->`
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
Runtime/cache references are centralized in:
- [`runtime/config_paths.md`](runtime/config_paths.md)
- [`runtime/data_cache_files.md`](runtime/data_cache_files.md)
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
High-frequency workflows:
1. Add/modify API endpoint: [`app_server/INDEX.md`](app_server/INDEX.md) -> target domain file under `app_server/domains/`.
2. Trace page -> API -> service: [`frontend/INDEX.md`](frontend/INDEX.md) then follow linked app-server domain files.
3. Update config/path behavior: [`runtime/config_paths.md`](runtime/config_paths.md).
4. Package/build troubleshooting: [`build/packaging.md`](build/packaging.md).
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Documentation can drift if `--write` is not run after route/page changes.
- AUTO-GEN blocks are deterministic; manual edits must stay inside MANUAL blocks.
<!-- MANUAL:END -->
