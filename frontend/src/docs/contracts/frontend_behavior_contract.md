# Frontend Behavior Contract

## Purpose
Define non-negotiable frontend interaction behaviors for shell, iframe pages, and cross-window usage.

## Scope
This contract applies to:
- `index.html`
- `ui_shell.js`
- `dataset_viewer.html`
- `dataset_main.js`
- `workflow.html`
- `workflow_main.js`
- `DFM.html`
- `DFM_main.js`
- `dfm_*.js`
- `project_settings.html`
- `project_settings.js`
- `popout_shell.html`
- `popout_bridge.js`
- `electron_preload.js`

## FB-1 Shell and Iframe Lifecycle
When:
- Editing tab management, page activation, or shell rendering logic.

MUST:
- Keep one persistent shell host and one persistent iframe host.
- Keep one iframe instance per non-home tab and switch visibility via show/hide.
- Preserve non-home tab state across tab switches.

MUST NOT:
- Destroy and recreate all tab iframes on every tab switch.
- Replace shell content in a way that invalidates existing iframe references.

Verify:
1. Open Dataset tab, switch to Home, then back: no forced full reload.
2. Open Workflow tab and DFM tab, switch among tabs: state preserved.

## FB-2 Message Protocol Stability
When:
- Editing cross-frame communication.

MUST:
- Keep `adas:*` message names stable unless a coordinated migration is implemented.
- Update all known producers and consumers in the same change when message payload changes.
- Document changed message schema in `docs/frontend/*.md` MANUAL sections.

MUST NOT:
- Introduce one-sided message payload changes.
- Rename high-frequency signals (`adas:workflow-dirty`, `adas:workflow-saved`, `adas:update-active-tab-title`) without migration.

Verify:
1. Workflow dirty/saved state still updates shell tab visual state.
2. Dataset/DFM tab title updates still propagate to shell.

## FB-3 Dirty-State and Close Semantics
When:
- Editing tab close logic, context menu actions, or save flows.

MUST:
- Keep confirmation for closing dirty tabs.
- Keep close/close others/close all behavior consistent with dirty-state prompts.
- Clear dirty state only on successful save acknowledgment.

MUST NOT:
- Clear dirty state before confirmed save result.
- Bypass close confirmation for dirty tabs.

Verify:
1. Mark workflow tab dirty, close tab: confirmation appears.
2. Save workflow, dirty indicator clears.

## FB-4 Hotkey and Menu Ownership
When:
- Editing keyboard shortcuts or menu action dispatching.

MUST:
- Keep hotkey interception disabled while typing in `input/textarea/select/contenteditable`.
- Keep shell-level and iframe-level shortcut responsibilities explicit.
- Keep safety actions (`close tab`, `shutdown`, `restart`) behind current guard behavior.

MUST NOT:
- Capture global shortcuts while user is typing.
- Duplicate conflicting handlers without ordering.

Verify:
1. Typing in editable field does not trigger shell hotkeys.
2. Existing primary shortcuts still work (`Ctrl+S`, `Ctrl+Shift+S`, `Alt+W`, etc.).

## FB-5 Pop-Out and Dock-Back Continuity
When:
- Editing pop-out lifecycle or channel bridge.

MUST:
- Preserve tab identity (`inst`, `type`, `datasetId`/`ds`, tab metadata) when undocking/docking.
- Ensure pop-out close notifies main shell for restoration.
- Keep message relay directions explicit (shell -> iframe, iframe -> shell).

MUST NOT:
- Lose tab state on dock-back.
- Keep stale channel instances after window close.

Verify:
1. Pop out a tab and dock back: tab is restored with expected state.
2. No duplicate restored tabs appear.

## FB-6 User Feedback Integrity
When:
- Editing title/status propagation.

MUST:
- Keep active tab title updates propagated to shell where applicable.
- Keep status messaging channel (`adas:status`) functional.
- Keep error/confirmation flows explicit for destructive actions.

MUST NOT:
- Swallow failures silently for save/import/export/reload paths.

Verify:
1. Title changes from child frames update shell tab title as expected.
2. Status bar or equivalent status updates still visible.

## FB-7 Performance Guardrails
When:
- Editing refresh/reload/caching behavior.

MUST:
- Prefer targeted refresh for active tab/page over whole-app reload.
- Keep one-time or guarded refresh flows (avoid infinite auto-refresh loops).
- Keep expensive operations behind explicit user action or clear trigger.

MUST NOT:
- Trigger unconditional full reload for routine tab activation.
- Add repeating timers/listeners without cleanup.

Verify:
1. Switching tabs does not trigger full app reload.
2. Auto-refresh logic runs at intended frequency only.

## Change Checklist
Before finishing a frontend behavior change:
1. State which FB-* rules are affected.
2. Update relevant MANUAL sections in `docs/frontend/*.md`.
3. Run `python tools/docs_index_builder.py --write`.
4. Run `python tools/docs_index_builder.py --check`.
