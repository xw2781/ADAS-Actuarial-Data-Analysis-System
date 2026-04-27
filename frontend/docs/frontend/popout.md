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
- `/ui_config`
- `/workflow/default_dir`
- `/workflow/load`

Detected `adas:*` message types in key JS files:
- `adas:autosave-toggle`
- `adas:browsing-history-updated`
- `adas:dfm-tab-activated`
- `adas:force-rebuild-toggle`
- `adas:open-path-result`
- `adas:set-app-font`
- `adas:set-zoom`
- `adas:tab-activated`
- `adas:workflow-load`
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
- Forwards `adas:*` iframe messages back to shell.
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
