import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve product root from __file__: main.py -> engine app -> core -> project root
_PRODUCT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PRODUCT_ROOT not in sys.path:
    sys.path.insert(0, _PRODUCT_ROOT)

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core.utils import get_config_value, get_project_root, normalize_function_name
from core.arcrho_engine.data_processing import (
    BASE_DICT,
    PROJECT_CONFIG,
    PROJECT_CONFIG_LOCK,
    ProjectSettingsError,
    UDF_ADASHeaders,
    UDF_ADASProjectSettings,
    UDF_ADASTri,
    _get_vps_last_modified_time,
    debug_mode,
    id_path,
    load_BASE_DICT,
    load_to_PROJECT_CONFIG,
    project_map_path,
    remove_old_instances,
    robot_id,
)
from core.arcrho_engine.general_utils import (
    DLOOKUP,
    convert_dict,
    get_current_time,
    read_json,
    read_txt,
    safe_remove,
    write_json,
    write_lists_to_csv,
)


class RequestHandler(FileSystemEventHandler):

    def on_moved(self, event):
        if event.is_directory:
            return
        if not event.dest_path.lower().endswith(".txt"):
            return

        file_path = event.dest_path

        # Process request immediately in the watchdog event thread
        self.process_file(file_path)

    def process_file_debug(self, file_path):
        if debug_mode == 0:
            try:
                self.process_file(file_path)
            except Exception as e:
                print(e)
        else:
            self.process_file(file_path)

    def process_file(self, file_path):
        try:
            arg = convert_dict(read_txt(file_path))
            print(arg)
        except:
            # print(f'\n* request sent to another agent')
            return

        try:
            project_name = arg['ProjectName']
            DLOOKUP(BASE_DICT['Project Map'], project_name, 'Project Name', 'Table Path')
        except:
            write_lists_to_csv(arg['DataPath'], [[f'(project not found: {project_name})']])
            return

        try:
            safe_remove(file_path)
        except: # Already removed by another agent
            return

        if debug_mode == 1:
            print(arg)

        print(f"\n> {get_current_time()} \n> new request # {robot_id} # user [{arg['UserName']}]")

        # Check VPS Updates (guarded)
        with PROJECT_CONFIG_LOCK:
            if project_name + " - Version" in PROJECT_CONFIG:
                vps_last_modified_time = _get_vps_last_modified_time(project_name)
                if PROJECT_CONFIG[project_name + " - Version"] < vps_last_modified_time:
                    load_to_PROJECT_CONFIG(project_name)
                    print(f">>> Virtual Project Settings Updated -> [{project_name} JSON]\n")
            # If missing, _get_df() will load it later; or you can proactively load it here.

        # Go to Functions
        try:
            function_name = normalize_function_name(arg.get('Function'))
            if function_name in ['ADASTri', 'ADASVec']:
                UDF_ADASTri(arg)
            elif function_name == 'ADASProjectSettings':
                UDF_ADASProjectSettings(arg)
            elif function_name == 'ADASHeaders':
                UDF_ADASHeaders(arg)
            else:
                write_lists_to_csv(arg['DataPath'], [['(invalid function name)']])
        except ProjectSettingsError as e:
            print(str(e))
            write_lists_to_csv(arg['DataPath'], [['project settings not defined']])
            return
        except Exception as e:
            if debug_mode:
                import traceback
                traceback.print_exc()
                print(arg)
            else:
                err_msg = f"(error: {str(e).upper()})"
                print(err_msg)
            write_lists_to_csv(arg['DataPath'], [[0]])
            return

        print(f"> request completed @ {get_current_time().split(' ')[1]}")


def start_monitoring(path):
    event_handler = RequestHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print('Server ID: ' + robot_id + '\n')

    remove_old_instances()
    load_BASE_DICT()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_json(id_path, {'Server': robot_id, 'Last seen': current_time})

    try:
        while True:

            if not os.path.exists(id_path):
                observer.stop(); break

            if get_config_value('apps.engine.kill_all'):
                os.remove(id_path)
                observer.stop(); break

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update Status
            arg_1 = read_json(id_path)
            arg_1['Last seen'] = current_time
            write_json(id_path, arg_1)

            # Check Base Settings (New Version Available?)
            if BASE_DICT["Project Map - Version"] < datetime.fromtimestamp(os.path.getmtime(project_map_path)):
                load_BASE_DICT()
                print(">>> Project Map Updated\n")

            time.sleep(5)

    except KeyboardInterrupt:
        observer.stop()

    observer.join()


start_monitoring(str(get_project_root() / "requests"))
