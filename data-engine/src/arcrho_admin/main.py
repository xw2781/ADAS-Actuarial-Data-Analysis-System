import argparse
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PRODUCT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

DEFAULT_PORT = 8765
STALE_AFTER_SECONDS = 60
PROJECT_ROOT_NAMES = ("ArcRho Server", "ArcRho", "ADAS")
COMPONENT_DIRS = {
    "engine": ("arcrho_engine", "ArcRho Engine", "ADAS Agent"),
    "orchestrator": ("arcrho_orchestrator", "ArcRho Orchestrator", "ADAS Master"),
}


def find_project_root():
    starts = [Path(__file__).resolve().parent, Path(sys.executable).resolve().parent]
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate.name in PROJECT_ROOT_NAMES:
                return candidate
            if ((candidate / "config" / "config.json").exists() or (candidate / "core" / "config.json").exists()):
                return candidate
    return PRODUCT_ROOT


PROJECT_ROOT = find_project_root()
_CONFIG_ENV = os.environ.get("ARCRHO_CONFIG") or os.environ.get("ADAS_CONFIG")
_DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
_LEGACY_CONFIG_FILE = PROJECT_ROOT / "core" / "config.json"


def resolve_config_file():
    if _DEFAULT_CONFIG_FILE.exists():
        return _DEFAULT_CONFIG_FILE
    if _CONFIG_ENV:
        return Path(_CONFIG_ENV)
    if _LEGACY_CONFIG_FILE.exists():
        return _LEGACY_CONFIG_FILE
    return _DEFAULT_CONFIG_FILE


CONFIG_FILE = resolve_config_file()


def resource_path(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def load_config():
    try:
        with open(CONFIG_FILE, mode="r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {
            "config_version": "1.0",
            "root": str(PRODUCT_ROOT),
            "apps": {
                "engine": {"kill_all": False},
                "orchestrator": {
                    "kill_all": False,
                    "auto_create_workers": True,
                    "max_workers": 5,
                },
            },
        }


def save_config(config):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CONFIG_FILE.with_name(f"{CONFIG_FILE.name}.{os.getpid()}.tmp")
    with open(temp_path, mode="w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    os.replace(temp_path, CONFIG_FILE)


def set_nested_value(data, key_path, value):
    parts = key_path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def instance_sources():
    return {
        "engine": ("Engine", resolve_app_path("engine", "instances")),
        "orchestrator": ("Orchestrator", resolve_app_path("orchestrator", "instances")),
    }


def resolve_app_path(role, *parts):
    normalized_parts = tuple(str(part) for part in parts)
    if normalized_parts and normalized_parts[0].lower() == "instances":
        return PROJECT_ROOT.joinpath("runtime", "instances", COMPONENT_DIRS[role][0], *normalized_parts[1:])
    for dirname in COMPONENT_DIRS[role]:
        candidate = PROJECT_ROOT / "core" / dirname
        if candidate.exists():
            return candidate.joinpath(*parts)
    return (PROJECT_ROOT / "core" / COMPONENT_DIRS[role][0]).joinpath(*parts)


def instance_age(last_seen):
    if not last_seen:
        return None
    try:
        seen = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max(0, int((datetime.now() - seen).total_seconds()))


def instance_user(server_name):
    parts = str(server_name or "").split("@")
    return parts[1] if len(parts) >= 3 and parts[1] else ""


def instance_created(server_name, path):
    token = str(server_name or "").split("@")[-1]
    try:
        created = datetime.strptime("-".join(token.split("-")[:2]), "%y%m%d-%H%M%S")
    except ValueError:
        try:
            created = datetime.fromtimestamp(path.stat().st_ctime)
        except OSError:
            return ""
    return created.strftime("%Y-%m-%d %H:%M:%S")


def read_instance_file(path):
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def list_instances():
    rows = []
    for role_key, (role_label, folder) in instance_sources().items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            data = read_instance_file(path)
            last_seen = data.get("Last seen", "")
            server = data.get("Server") or path.stem
            age = instance_age(last_seen)
            rows.append(
                {
                    "role": role_key,
                    "role_label": role_label,
                    "name": path.name,
                    "server": server,
                    "user": data.get("User") or instance_user(server),
                    "created": data.get("Created") or instance_created(server, path),
                    "last_seen": last_seen,
                    "age_seconds": age,
                    "status": "Active" if age is None or age <= STALE_AFTER_SECONDS else "Stale",
                }
            )
    return rows


def remove_instance(role, name):
    sources = instance_sources()
    if role not in sources:
        raise ValueError(f"Unknown role: {role}")
    if not name.lower().endswith(".json") or Path(name).name != name:
        raise ValueError("Invalid instance file name")

    path = sources[role][1] / name
    folder = sources[role][1].resolve()
    resolved = path.resolve()
    if folder not in resolved.parents:
        raise ValueError("Instance path is outside the expected folder")
    if resolved.exists():
        resolved.unlink()
        return True
    return False


class AdminHandler(BaseHTTPRequestHandler):
    server_version = "ArcRhoAdmin/1.0"

    def log_message(self, _format, *_args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_file(resource_path("index.html"), "text/html; charset=utf-8")
        elif parsed.path == "/api/config":
            self.send_json({"path": str(CONFIG_FILE), "config": load_config()})
        elif parsed.path == "/api/instances":
            self.send_json({"instances": list_instances(), "stale_after_seconds": STALE_AFTER_SECONDS})
        else:
            self.send_error(404, "Not found")

    def do_PATCH(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/config":
            self.send_error(404, "Not found")
            return

        payload = self.read_json_body()
        key_path = payload.get("path")
        if not key_path:
            self.send_error(400, "Missing config path")
            return

        config = load_config()
        set_nested_value(config, key_path, payload.get("value"))
        save_config(config)
        self.send_json({"ok": True, "path": str(CONFIG_FILE), "config": config})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/instances":
            self.send_error(404, "Not found")
            return

        query = parse_qs(parsed.query)
        try:
            removed = remove_instance(query.get("role", [""])[0], query.get("name", [""])[0])
            self.send_json({"ok": True, "removed": removed, "instances": list_instances()})
        except ValueError as exc:
            self.send_error(400, str(exc))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/open-config-folder":
            os.startfile(CONFIG_FILE.parent)
            self.send_json({"ok": True})
        elif parsed.path == "/api/shutdown":
            self.send_json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(404, "Not found")

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_file(self, path, content_type):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Missing file: {path}")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def make_server(port):
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidate), AdminHandler)
        except OSError:
            continue
    raise OSError(f"No available port found from {port} to {port + 19}")


def main():
    parser = argparse.ArgumentParser(description="Run ArcRho Admin Control in a local browser.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = make_server(args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"ArcRho Admin Control: {url}")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()



