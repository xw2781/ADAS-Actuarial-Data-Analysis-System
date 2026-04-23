"""Shared configuration, paths, constants, locks, and mutable globals.

This module is the foundation of the backend — it has zero imports from other
backend modules to prevent circular dependencies.  Every other backend module
may freely ``from backend import config`` and reference ``config.DATA_DIR``,
``config._AUDIT_LOG_LOCK``, etc.
"""
from __future__ import annotations

import os
import re
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "ui_config.json").exists() or (candidate / "index.html").exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = _resolve_project_root()

# ---------------------------------------------------------------------------
# Config — Load from ui_config.json
# ---------------------------------------------------------------------------

UI_CONFIG_PATH = str(PROJECT_ROOT / "ui_config.json")
DEFAULT_ROOT_PATH = r"E:\ADAS"


def load_ui_config() -> Dict[str, Any]:
    """Load configuration from ui_config.json."""
    try:
        with open(UI_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    if not isinstance(raw, dict):
        raw = {}

    root_path = raw.get("root_path")
    if not isinstance(root_path, str) or not root_path.strip():
        root_path = DEFAULT_ROOT_PATH

    paths = raw.get("paths")
    if not isinstance(paths, dict):
        paths = {}

    return {"root_path": root_path, "paths": paths}


def get_root_path() -> str:
    """Get the root path from config."""
    return load_ui_config()["root_path"]


def get_path(subpath: str) -> str:
    """Get a full path by joining root_path with subpath."""
    return os.path.join(get_root_path(), subpath)


# Derived paths (loaded dynamically from config)
def _get_data_dir() -> str:
    cfg = load_ui_config()
    return get_path(cfg.get("paths", {}).get("web_ui", "Web UI"))


def _get_project_map_dir() -> str:
    cfg = load_ui_config()
    return get_path(cfg.get("paths", {}).get("project_map", "projects"))


def _get_virtual_projects_dir() -> str:
    cfg = load_ui_config()
    return get_path(cfg.get("paths", {}).get("virtual_projects", "Virtual Projects"))


def _get_data_base() -> str:
    cfg = load_ui_config()
    return get_path(cfg.get("paths", {}).get("data", "data"))


def _get_requests_dir() -> str:
    cfg = load_ui_config()
    return get_path(cfg.get("paths", {}).get("requests", "requests"))


def _get_workflow_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Documents", "ArcRho", "workflows")


def _get_scripting_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Documents", "ArcRho", "scripts")


# Project settings JSON files (on shared network drive)
PROJECT_SETTINGS_SOURCES = {
    "project_map": "map.json",
    # Add more sources here as needed
}

WORKFLOW_EXT = ".arcwf"

# ---------------------------------------------------------------------------
# Mutable runtime paths — refreshed from config
# ---------------------------------------------------------------------------

DATA_DIR: str = ""
PROJECT_SETTINGS_DIR: str = ""
PROJECT_BOOK: str = ""
WORKFLOW_DIR: str = ""
SCRIPTING_DIR: str = ""
ALLOWED_BOOK_DIRS: List[Path] = []
REQUEST_DIR: str = ""
DATA_BASE: str = ""


def refresh_runtime_paths() -> None:
    """Refresh runtime directories from ui_config.json."""
    global DATA_DIR, PROJECT_SETTINGS_DIR, PROJECT_BOOK, WORKFLOW_DIR, SCRIPTING_DIR
    global ALLOWED_BOOK_DIRS, REQUEST_DIR, DATA_BASE
    DATA_DIR = _get_data_dir()
    PROJECT_SETTINGS_DIR = _get_project_map_dir()
    PROJECT_BOOK = os.path.join(
        PROJECT_SETTINGS_DIR,
        PROJECT_SETTINGS_SOURCES.get("project_map", "map.json"),
    )
    WORKFLOW_DIR = _get_workflow_dir()
    SCRIPTING_DIR = _get_scripting_dir()
    ALLOWED_BOOK_DIRS = [
        Path(_get_virtual_projects_dir()).resolve(),
        # add more if needed
    ]
    REQUEST_DIR = _get_requests_dir()
    DATA_BASE = _get_data_base()


# In-memory dataset cache
DATASETS: Dict[str, str] = {}

# Initialise paths on first import
refresh_runtime_paths()

# ---------------------------------------------------------------------------
# File-name constants
# ---------------------------------------------------------------------------

FOLDER_STRUCTURE_FILE = "folder_structure.json"
FIELD_MAPPING_FILE = "field_mapping.json"
FIELD_MAPPING_SIGNIFICANCES = {
    "Reserving Class",
    "Origin Date",
    "Development Date",
    "Dataset",
}
RESERVING_CLASS_VALUES_FILE = "reserving_class_values.json"
RESERVING_CLASS_COMBINATIONS_FILE = "reserving_class_combinations_cache.json"
RESERVING_CLASS_TYPES_FILE = "reserving_class_types.json"
RESERVING_CLASS_PATH_TREE_FILE = "reserving_class_path_tree_cache.json"
RESERVING_CLASS_PATH_TREE_MAX_GENERATED = 250000
RESERVING_CLASS_HIDDEN_PATHS_PREF_FILE = "reserving_class_hidden_paths.json"
RESERVING_CLASS_FILTER_SPEC_PREF_FILE = "reserving_class_filter_spec.json"
SCRIPTING_PREFS_FILE = "scripting_prefs.json"
DATASET_TYPES_FILE = "dataset_types.json"
PROJECT_SETTINGS_XLSX_FILE = "settings.xlsx"
RESERVING_CLASS_TYPES_SHEET_NAME = "Reserving Class Types"
RESERVING_CLASS_TYPES_COLUMNS = ["Name", "Level", "Formula", "EEX Formula"]
RESERVING_CLASS_TYPES_FILE_COLUMNS = ["Name", "Level", "Formula", "EEX Formula", "Source"]
DATASET_TYPES_COLUMNS = ["Name", "Data Format", "Category", "Calculated", "Formula"]
DATASET_TYPES_FILE_COLUMNS = ["Name", "Data Format", "Category", "Calculated", "Formula", "Source"]
AUDIT_LOG_FILE = "audit_log.json"
AUDIT_LOG_MAX_ENTRIES = 5000
GENERAL_SETTINGS_FILE = "general_settings.json"

# ---------------------------------------------------------------------------
# Thread locks
# ---------------------------------------------------------------------------

_AUDIT_LOG_LOCK = threading.Lock()
_RESERVING_CLASS_PATH_TREE_LOCK = threading.Lock()
_RESERVING_CLASS_HIDDEN_PATHS_LOCK = threading.Lock()
_RESERVING_CLASS_FILTER_SPEC_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# App-control flag paths
# ---------------------------------------------------------------------------

BASE_DIR = PROJECT_ROOT
RESTART_FLAG = BASE_DIR / ".restart_app"
SHUTDOWN_FLAG = BASE_DIR / ".shutdown_app"
ELECTRON_RESTART_FLAG = BASE_DIR / ".restart_electron"
ELECTRON_SHUTDOWN_FLAG = BASE_DIR / ".shutdown_electron"

# ---------------------------------------------------------------------------
# Path-resolver helpers
# ---------------------------------------------------------------------------

def _sanitize_folder_name(name: str) -> str:
    invalid = [":", "*", "?", '"', "<", ">", "|"]
    out = name or ""
    for ch in invalid:
        out = out.replace(ch, "_")
    return out


def _sanitize_project_dir_name(name: str) -> str:
    out = (name or "").strip()
    for ch in ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]:
        out = out.replace(ch, "_")
    return out


def _infer_project_name_from_table_path(table_path: str) -> str:
    """Infer project name from table file name like <project_name>_YYYYMM.csv."""
    stem = Path(table_path).stem
    m = re.match(r"^(?P<name>.+)_\d{6}$", stem)
    project_name = (m.group("name") if m else stem).strip()
    return _sanitize_folder_name(project_name) or _sanitize_folder_name(stem) or "project"


def _find_existing_project_dir(project_name: str) -> Optional[str]:
    """Find an existing project folder under E:\\ADAS\\projects by name (case-insensitive)."""
    target = _sanitize_folder_name(project_name or "").strip()
    if not target:
        return None

    direct = os.path.join(PROJECT_SETTINGS_DIR, target)
    if os.path.isdir(direct):
        return direct

    try:
        target_l = target.lower()
        with os.scandir(PROJECT_SETTINGS_DIR) as it:
            for entry in it:
                if entry.is_dir() and entry.name.strip().lower() == target_l:
                    return entry.path
    except Exception:
        return None
    return None


def get_cache_path(csv_path: str, project_name: Optional[str] = None) -> str:
    """Get table-summary cache path under an existing project folder."""
    chosen_name = (project_name or "").strip() or _infer_project_name_from_table_path(csv_path)
    project_dir = _find_existing_project_dir(chosen_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {chosen_name}")
    return os.path.join(project_dir, "table_summary.json")


def get_field_mapping_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, FIELD_MAPPING_FILE)


def get_dataset_types_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, DATASET_TYPES_FILE)


def get_reserving_class_values_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, RESERVING_CLASS_VALUES_FILE)


def get_reserving_class_combinations_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, RESERVING_CLASS_COMBINATIONS_FILE)


def get_reserving_class_types_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, RESERVING_CLASS_TYPES_FILE)


def get_reserving_class_path_tree_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, RESERVING_CLASS_PATH_TREE_FILE)


def _get_user_appdata_cache_dir() -> str:
    appdata = str(os.environ.get("APPDATA") or "").strip()
    if not appdata:
        appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    return os.path.join(appdata, "ArcRho", "WebUI", "cache")


def get_reserving_class_hidden_paths_pref_path() -> str:
    return os.path.join(_get_user_appdata_cache_dir(), RESERVING_CLASS_HIDDEN_PATHS_PREF_FILE)


def get_reserving_class_filter_spec_pref_path() -> str:
    return os.path.join(_get_user_appdata_cache_dir(), RESERVING_CLASS_FILTER_SPEC_PREF_FILE)


def get_scripting_prefs_path() -> str:
    return os.path.join(_get_user_appdata_cache_dir(), SCRIPTING_PREFS_FILE)


def get_project_settings_workbook_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, PROJECT_SETTINGS_XLSX_FILE)


def get_audit_log_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, AUDIT_LOG_FILE)


def get_general_settings_path(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, GENERAL_SETTINGS_FILE)


def get_project_data_dir(project_name: str) -> str:
    project_dir = _find_existing_project_dir(project_name)
    if not project_dir:
        raise ValueError(f"Project folder not found under projects: {project_name}")
    return os.path.join(project_dir, "data")
