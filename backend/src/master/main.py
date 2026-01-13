import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

instance_path = r"E:\ADAS\core\ADAS Agent\instances"


def kill_extra_python_processes():
    # collect python processes
    py_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'exe', 'create_time']):
        try:
            if proc.info['name'] and 'ADAS Master.exe' in proc.info['name'].lower():
                py_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # if only one, do nothing
    if len(py_procs) <= 1:
        return 0

    # here we kill everything
    killed = 0
    for proc in py_procs[:-1]:
        try:
            proc.terminate()
            killed += 1
        except Exception:
            pass

    return killed

try:
    kill_extra_python_processes()
except Exception as e:
    print(e)


def read_txt(txt_file, retries=50, delay=0.02):
    """
    Reads key=value lines safely with retries.
    Supports values that contain '='.
    Ignores blank / malformed lines.
    """

    # ---- 1. Wait until file is available ----
    for _ in range(retries):
        try:
            with open(txt_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
            break
        except PermissionError:
            time.sleep(delay)
    else:
        raise PermissionError(f"Cannot open {txt_file}")

    # ---- 2. Parse key=value ----
    arg_dict = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Only split at first '='
        if '= ' in line:
            key, value = line.split(' = ', 1)
        elif '=' in line:
            key, value = line.split('=', 1)
        else:
            continue

        arg_dict[key.strip()] = value.strip()

    return arg_dict


def remove_old_instances():
    FOLDER = instance_path
    AGE_SECONDS = 5 * 60  # 5 minutes
    now = time.time()

    for name in os.listdir(FOLDER):
        path = os.path.join(FOLDER, name)
        # only remove files
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
            if now - mtime > AGE_SECONDS:
                os.remove(path)
        except Exception:
            # ignore locked / race-condition files
            pass


def cmd(name):
    '''
    KILL_ALL_AGENTS = 0
    KILL_ALL_MASTER = 0
    AUTO_CREATE_WORKERS = 1
    MAX_WORKERS = 4
    '''
    txt = read_txt(r"E:\ADAS\core\ADAS Master\command.txt")[name]

    if 'MAX' in name:
        return int(txt)
    else:
        if txt.upper() in ['TRUE', '1']:
            return True
        elif txt.upper() in ['FALSE', '0']:
            return False
        

def file_counts():
    FOLDER = instance_path
    file_count = sum(
        1 for name in os.listdir(FOLDER)
        if os.path.isfile(os.path.join(FOLDER, name))
    )
    return file_count


while True:
    try:
        if cmd('KILL_ALL_MASTER'):
            break

        remove_old_instances()

        if cmd('AUTO_CREATE_WORKERS') and file_counts() < cmd('MAX_WORKERS'):
            num_workers_to_add = cmd('MAX_WORKERS') - file_counts()

            for i in range(1, num_workers_to_add+1):
                time.sleep(1)

                exe = Path(r"E:\ADAS\core\ADAS Agent\dist\ADAS Agent\ADAS Agent.exe")

                subprocess.Popen(
                    [str(exe)],
                    close_fds=True
                )
                
        time.sleep(30)
    except:
        time.sleep(10)





