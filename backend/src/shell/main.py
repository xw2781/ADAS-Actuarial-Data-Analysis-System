from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
import subprocess

def install_startup_shortcut(
    exe_path: str | os.PathLike,
    shortcut_name: Optional[str] = None,
    args: str = "",
    work_dir: Optional[str | os.PathLike] = None,
    icon_path: Optional[str | os.PathLike] = None,
    description: str = "",
) -> Path:
    """
    Create (or overwrite) a .lnk shortcut in the current user's Startup folder.

    Requirements:
      pip install pywin32

    Args:
      exe_path: Full path to the .exe.
      shortcut_name: Shortcut file name (without .lnk). Defaults to exe file stem.
      args: Command-line arguments for the target.
      work_dir: Working directory. Defaults to exe folder.
      icon_path: Path to .ico or .exe to use as icon. Defaults to exe_path.
      description: Shortcut description.

    Returns:
      Path to the created shortcut (.lnk).
    """
    exe = Path(exe_path).expanduser().resolve()
    if not exe.exists():
        raise FileNotFoundError(f"EXE not found: {exe}")
    if exe.suffix.lower() != ".exe":
        raise ValueError(f"Target must be an .exe: {exe}")

    name = (shortcut_name or exe.stem).strip()
    if not name:
        raise ValueError("shortcut_name is empty")

    startup_dir = Path(os.environ["APPDATA"]) / r"Microsoft\Windows\Start Menu\Programs\Startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    lnk_path = startup_dir / f"{name}.lnk"

    wd = Path(work_dir).expanduser().resolve() if work_dir else exe.parent
    ico = Path(icon_path).expanduser().resolve() if icon_path else exe

    # Create shortcut via WScript.Shell
    try:
        import win32com.client  # type: ignore
    except ImportError as e:
        raise ImportError("pywin32 is required: pip install pywin32") from e

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(str(lnk_path))
    shortcut.TargetPath = str(exe)
    shortcut.Arguments = args or ""
    shortcut.WorkingDirectory = str(wd)
    shortcut.Description = description or ""
    shortcut.IconLocation = str(ico)  # can be exe or ico
    shortcut.Save()

    return lnk_path


# lnk = install_startup_shortcut(
#     r"E:\ADAS\core\ADAS Master\dist\ADAS Master\ADAS Master.exe",
#     shortcut_name="ADAS Master",
#     args="--silent",
#     description="Launch ADAS Master at login",
# )
# print("\n> App Installed/Updated:", lnk); time.sleep(0.5)


# lnk = install_startup_shortcut(
#     r"E:\ResQ\Excel Add-ins\URA master\dist\URA master.exe",
#     shortcut_name="URA master",
#     args="--silent",
#     description="Launch URA master at login",
# )
# print("\n> App Installed/Updated:", lnk); time.sleep(0.5)


lnk = install_startup_shortcut(
    r"E:\ADAS\core\ADAS Shell\dist\ADAS Shell\ADAS Shell.exe",
    shortcut_name="ADAS Shell",
    args="--silent",
    description="Launch ADAS Shell at login",
)

print("\n> Shortcut created:", lnk); time.sleep(0.5)

print('\n> Start Applications ...')

# subprocess.Popen([r"E:\ADAS\core\ADAS Master\dist\ADAS Master\ADAS Master.exe"], close_fds=True)
# subprocess.Popen([r"E:\ResQ\Excel Add-ins\URA master\dist\URA master.exe"], close_fds=True)

os.startfile(r"E:\ADAS\core\ADAS Master\dist\ADAS Master\ADAS Master.exe")
os.startfile(r"E:\ResQ\Excel Add-ins\URA master\dist\URA master.exe")

# time.sleep(0.5)

print('\n> Done'); time.sleep(2)

sys.exit(0)
