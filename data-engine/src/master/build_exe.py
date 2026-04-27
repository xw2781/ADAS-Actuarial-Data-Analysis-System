import subprocess
from pathlib import Path
import sys
import shutil

BASE_DIR = Path(__file__).resolve().parent

VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
REQ_FILE = BASE_DIR / "requirements.txt"

ENTRY_PY = BASE_DIR / "main.py"
ICON = r"E:\ADAS\library\icon\ADASV7.ico"

DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"

try:
    shutil.rmtree(DIST_DIR)
except:
    pass

try:
    shutil.rmtree(BUILD_DIR)
except:
    pass

shutil.rmtree(r"E:\ADAS\core\ADAS Master\spec")

def run(cmd, check=True):
    print("\n>>>", " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), check=check)


def ensure_venv():
    if VENV_PYTHON.exists():
        return

    print("\n>>> Creating virtual environment (.venv)")
    run([sys.executable, "-m", "venv", BASE_DIR / ".venv"])

    if not VENV_PYTHON.exists():
        raise RuntimeError("Failed to create virtual environment")


def ensure_venv_python():
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Venv python not found: {VENV_PYTHON}")


def install_requirements():
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found: {REQ_FILE}")

    # Upgrade pip tooling first (more reliable installs)
    run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install requirements
    run([VENV_PYTHON, "-m", "pip", "install", "-r", REQ_FILE])


def build_exe():
    cmd = [
        VENV_PYTHON,
        "-m", "PyInstaller",
        "--specpath", BASE_DIR / "spec",
        "--noconfirm",   
        f"--icon={ICON}",
        "--add-data", f"{ICON};.",
        "--noconsole",
        "--clean",
        "--name", "ADAS Master",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        ENTRY_PY,
    ]

    print("\nRunning PyInstaller:")
    print(" ".join(map(str, cmd)))
    run(cmd)


def main():
    ensure_venv()
    ensure_venv_python()
    install_requirements()
    build_exe()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
