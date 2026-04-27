# Backend Domain: app_control

## Purpose
<!-- MANUAL:BEGIN -->
Application lifecycle control domain (restart/shutdown flags) coordinated between backend routes and Electron host startup/shutdown flow.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.app_control.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/app/restart` | `app_restart` | - | - | - |
| `POST` | `/app/restart_electron` | `app_restart_electron` | - | - | - |
| `POST` | `/app/shutdown` | `app_shutdown` | - | - | - |
| `POST` | `/app/shutdown_electron` | `app_shutdown_electron` | - | - | - |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.app_control.key_files -->
- [`backend/api/app_control_router.py`](../../../backend/api/app_control_router.py) - Restart/shutdown control endpoints.
- [`backend/config.py`](../../../backend/config.py) - Flag-file paths for app control.
- [`app_launcher.py`](../../../app_launcher.py) - Launcher process watching control flags.
- [`electron_main.js`](../../../electron_main.js) - Electron host restart/shutdown integration.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Called by shell app control actions.
- Coordinated with launcher/electron host watchers.
- Electron host startup clears stale lifecycle flags before spawning backend; shutdown paths post `/app/shutdown` then wait before force-kill fallback.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Uses lifecycle flag files under project root: `.restart_app`, `.shutdown_app`, `.restart_electron`, `.shutdown_electron`.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add lifecycle action: define flag contract and watcher handling in launcher/host.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Incorrect flag behavior can cause app restart loops.
- Stale `.shutdown_app` plus premature process kill can cause next-launch startup timeout.
<!-- MANUAL:END -->
