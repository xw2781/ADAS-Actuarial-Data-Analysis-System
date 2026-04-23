# Architecture Guardrails

## Purpose
Capture critical architectural decisions that agents must preserve unless explicitly instructed otherwise.

## AR-1 Layer Boundaries
MUST:
- Keep backend layering as:
  - router: transport and validation orchestration
  - service: business logic and persistence flow
  - schema: request/response model definitions
- Keep routers thin; move non-trivial logic to services.

MUST NOT:
- Embed substantial business logic directly in `backend/api/*`.
- Bypass schemas for typed payload paths where schemas already exist.

## AR-2 Config as Single Source of Truth
MUST:
- Use `backend/config.py` for runtime path constants and path helper functions.
- Keep environment/path derivation centralized.

MUST NOT:
- Duplicate path resolution logic ad hoc in multiple services.
- Introduce hardcoded project-root assumptions in domain services.

## AR-3 Frontend Host Architecture
MUST:
- Keep `ui_shell.js` as tab/iframe orchestration layer.
- Keep feature pages (`dataset`, `workflow`, `DFM`, `project_settings`) isolated in iframe contexts.
- Keep cross-context communication via explicit `adas:*` messages.

MUST NOT:
- Collapse all feature logic into shell context.
- Introduce hidden implicit coupling between iframes without message contract.

## AR-4 Stable Public Contracts
MUST:
- Treat route paths and high-traffic message names as stable contracts.
- Document any contract changes in MANUAL docs and contract files.
- Preserve backward compatibility by default.

MUST NOT:
- Rename/reshape contracts silently.

## AR-5 Persistence and Cache Discipline
MUST:
- Keep cache file names/locations managed through config constants.
- Keep lock-aware file write patterns for shared cache/preference files.
- Keep refresh endpoints explicit about side effects.

MUST NOT:
- Perform unsafe concurrent writes without lock strategy.
- Hide cache invalidation behavior from route contracts.

## AR-6 Change Impact Discipline
MUST:
- Evaluate blast radius across:
  - frontend consumers
  - backend routes/services
  - persisted files and compatibility
- Update corresponding docs in same change.

MUST NOT:
- Ship architecture-affecting code changes without doc updates.

## AR-7 Packaging and Runtime Separation
MUST:
- Keep packaging/build concerns in `package.json`, `server.spec`, installer scripts.
- Keep runtime behavior independent from build-only directories.

MUST NOT:
- Introduce runtime dependency on transient build artifacts.

## AR-8 Exception Process
When a requested change conflicts with guardrails:
1. Identify exact violated AR-* rule.
2. Describe risk and alternatives.
3. Proceed only with explicit approval for a deliberate exception.

## Architectural Decision Checklist
Before finishing architecture-impacting changes:
1. Mark affected AR-* rules.
2. Update relevant docs (`docs/INDEX.md`, module indexes, domain indexes, contracts).
3. Run `python tools/docs_index_builder.py --write`.
4. Run `python tools/docs_index_builder.py --check`.
