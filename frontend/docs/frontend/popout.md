# Frontend: Popout

## Purpose
<!-- MANUAL:BEGIN -->
Standalone pop-out window to host one tab outside main shell.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.popout.entry_points -->
- `ui/shell/popout_shell.html`: external scripts _none_; inline imports `/ui/shell/popout_bridge.js`.

Detected `fetch(...)` targets in key JS files:
- `/`
- `/app/restart`
- `/app/restart_electron`
- `/app/shutdown`
- `/restart`
- `/workflow/default_dir`
- `/workflow/load`
- `/workspace_paths`

Detected `arcrho:*` message types in key JS files:
- `arcrho:autosave-toggle`
- `arcrho:browsing-history-updated`
- `arcrho:dfm-tab-activated`
- `arcrho:force-rebuild-toggle`
- `arcrho:open-path-result`
- `arcrho:set-app-font`
- `arcrho:set-zoom`
- `arcrho:tab-activated`
- `arcrho:workflow-load`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.popout.key_files -->
- [`ui/shell/popout_shell.html`](../../ui/shell/popout_shell.html) - Secondary window shell hosting one iframe tab.
- [`ui/shell/popout_bridge.js`](../../ui/shell/popout_bridge.js) - BroadcastChannel connection between windows.
- [`ui/shell/ui_shell.js`](../../ui/shell/ui_shell.js) - Main-window pop-out/dock-back orchestration.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Uses `BroadcastChannel` for shell <-> popout message relays.
- Forwards `arcrho:*` iframe messages back to shell.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Carries tab identity/state via URL params and channel payloads.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add popout tab type support: update src construction + dock-back state mapping.
2. Change popout controls: adjust `popout_shell.html` and bridge handling.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Channel lifecycle race conditions during window close/dock-back.
- Mismatched tab instance IDs can orphan popout state.
<!-- MANUAL:END -->
