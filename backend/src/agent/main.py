import os
import re
import time
import uuid
import csv
import numpy as np
import calendar
import threading
from pathlib import Path
from threading import Lock
from datetime import date, datetime
import pandas as pd
from utils import File
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


MAX_WORKERS = 1
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

debug_mode = 0
team_profile_path = r"E:\ADAS\Team Profile\Actuarial_NJ.xlsm"
command_path = r"E:\ADAS\core\ADAS Master\command.txt"
robot_id = datetime.now().strftime(f"%y%m%d-%H%M%S") + f'NE7SASWPN02@' + os.getlogin()
id_folder = r"E:\ADAS\core\ADAS Agent\instances"
id_path = id_folder + '\\' + robot_id + '.txt'


BASE_DICT = {}  # Base Settings Table
VPS_DICT = {}   # Virtual Project Setting Files
DATA_DICT = {}  # CSV Data Table Files
DATA_DICT_LOCK = Lock() # for atomic swap
VPS_DICT_LOCK = Lock()
BASE_DICT_LOCK = Lock()


def remove_old_instances():
    today = date.today()
    for f in Path(id_folder).iterdir():
        if f.is_file():
            modified_date = datetime.fromtimestamp(f.stat().st_mtime).date()
            if modified_date < today:
                f.unlink()


def load_BASE_DICT():
    with open(team_profile_path, "rb") as f:
        bio = BytesIO(f.read())
    with pd.ExcelFile(bio, engine="openpyxl") as xls:
        BASE_DICT['Team Profile'] = pd.read_excel(xls, sheet_name='Virtual Projects', header=0, index_col=None).fillna('')
        BASE_DICT['Team Profile - Version'] = datetime.now()


def DLOOKUP(df, lookup_value, lookup_col, return_col):
    """
    Lookup a value in DataFrame
    """
    return df[df[lookup_col]==lookup_value][return_col].iloc[0]


def to_upper_case(df):
    return df.apply(lambda col: col.str.upper() if col.dtype == "object" else col)


def get_current_time():
    now = datetime.now()
    milliseconds = now.microsecond // 1000  # Convert microseconds to milliseconds
    formatted_date_time = now.strftime(f"%m/%d %H:%M:%S ({milliseconds})")
    return formatted_date_time


def load_to_VPS_DICT(project_name, settings_file):
    print(f"Loading {settings_file} @ {get_current_time()}")
    with open(settings_file, "rb") as f:
        bio = BytesIO(f.read())
    with pd.ExcelFile(bio, engine="openpyxl") as xls:
        VPS_DICT[project_name] = {}
        VPS_DICT[project_name + " - Version"] = datetime.now()
        for page_name in ['Source Table', 'Dataset Types', 'Reserving Class Types']:
            VPS_DICT[project_name][page_name] = pd.read_excel(xls, sheet_name=page_name, header=0, index_col=None).fillna('')


def load_to_DATA_DICT(csv_path):
    print(f"Loading Data Table {csv_path} @ {get_current_time()}")
    key = os.path.basename(csv_path)
    DATA_DICT[key] = pd.read_csv(csv_path)
    DATA_DICT[key + " - Version"] = datetime.now()
    print(f"Data Table Loaded @ {get_current_time()}")


def load_dataframe(data_csv_path):
    '''
    Add a new table to DATA_DICT
    '''
    print(get_current_time())
    print(f'Loading Data Table -- [{os.path.basename(data_csv_path)}]')
    df = pd.read_csv(data_csv_path) # build off-thread
    with DATA_DICT_LOCK:
        DATA_DICT[os.path.basename(data_csv_path).replace('.csv', '')] = df

    print(get_current_time())
    print(f'Data Table Loaded -- [{os.path.basename(data_csv_path)}]')


def load_dataframe_in_thread(data_csv_path):
    t = threading.Thread(target=load_dataframe, args=(data_csv_path,), daemon=True)
    t.start()


def strip_outer_quotes(s: str):
    return re.sub(r'^\s*"|"(\s*)$', '', s).strip()


def split_formula_with_ops(s: str):
    """
    Return:
      - items: dataset names
      - ops:   operator BEFORE each item ('+' or '-')
    """

    token_pattern = re.compile(
        r'''
        (?P<op>[+-]?)\s*                  # optional leading operator
        (?:
            "(?P<quoted>[^"]*)"           # quoted token
            |
            (?P<unquoted>[^"+*/()\-\s]+   # unquoted token
                (?:\s+[^"+*/()\-\s]+)*)
        )
        ''',
        re.VERBOSE
    )

    items = []
    ops = []

    for m in token_pattern.finditer(s):
        op = m.group('op') or '+'
        token = m.group('quoted') or m.group('unquoted')

        token = token.strip()
        if not token:
            continue

        items.append(token)
        ops.append(op)

    return items, ops


def split_formula(s: str):
    return split_formula_with_ops(s)[0]


def split_formula_opts(s: str):
    return split_formula_with_ops(s)[1]


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


def write_txt(txt_file, arg):
    content = ''
    for item in arg.items():
        content += item[0] + ' = ' + item[1] + '\n'
    with open(txt_file, "w") as file:
        file.write(content)


def time_diff(time_str_1, time_str_2 = 'Current Time'):
    time_1 = datetime.strptime(time_str_1, "%Y-%m-%d %H:%M:%S")
    if time_str_2 == 'Current Time':
        time_difference = (datetime.now() - time_1).total_seconds()
    else:
        time_difference = (datetime.strptime(time_str_2, "%Y-%m-%d %H:%M:%S") - time_1).total_seconds()
    return time_difference
    

def write_lists_to_csv(csv_path, lists, overwrite=True):
    folder_path = os.path.dirname(csv_path)
    tmp_folder = folder_path + '\\tmp'
    tmp_csv_path = tmp_folder + '\\' + os.path.basename(csv_path)
   
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    if os.path.exists(tmp_csv_path):
        safe_remove(tmp_csv_path)

    with open(tmp_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for lst in lists:
            writer.writerow(lst)

    if os.path.exists(csv_path):
        safe_remove(csv_path)

    time.sleep(0.05)
    os.rename(tmp_csv_path, csv_path)


def _calc_age(acc_yrmo, sys_yrmo):
    acc_yr = acc_yrmo//100
    sys_yr = sys_yrmo//100
    acc_mo = acc_yrmo % 100
    sys_mo = sys_yrmo % 100

    return 12*(sys_yr-acc_yr) + sys_mo-acc_mo + 1


def _get_org_label(yyyymm, org_len):
    year = int(yyyymm // 100)
    month = int(yyyymm % 100)

    if org_len == 1:
        return yyyymm
        # return "'" + datetime.strptime(str(yyyymm), "%Y%m").strftime("%b %Y")
    
    elif org_len == 3:  
        return f"{year} Q{(month+2)//3}"
    elif org_len == 6:  
        return f"{year} H{(month+5)//6}"
    elif org_len == 12: 
        return year


def safe_remove(file_path, attempts=5, delay=0.1):
    """Attempt to remove a file with retries on permission error."""
    for _ in range(attempts):
        try:
            tmp = f"{file_path}.{uuid.uuid4()}.deleting"
            os.replace(file_path, tmp)  # atomic
            os.remove(tmp)
            return True
        except PermissionError:
            time.sleep(delay)

    return False


def smart_convert(value: str):
    """
    Convert a string into int, bool, or str.
    """
    v = value.strip()

    # Try boolean first
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # Try integer
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)

    # Fallback: string
    return v


def convert_dict(input_dict: dict) -> dict:
    """
    Apply smart_convert to all values in a dictionary.
    """
    return {k: smart_convert(str(v)) for k, v in input_dict.items()}


def vector_to_triangle(df: pd.Series | pd.DataFrame, colnames=None) -> pd.DataFrame:
    """
    Convert a vector (Series or 1-col DataFrame) to a triangle DataFrame.
    If the input is already square (n×n), return it unchanged.
    
    colnames: optional list/Index to use as column names.
              If None, defaults to using the row index.
    """
    # Case 1: Already a square DataFrame → do nothing
    if isinstance(df, pd.DataFrame) and df.shape[0] == df.shape[1]:
        return df

    # Convert to Series for uniform processing
    if isinstance(df, pd.DataFrame):
        if df.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column or be square.")
        s = df.iloc[:, 0]
    elif isinstance(df, pd.Series):
        s = df
    else:
        raise TypeError("Input must be a pandas Series or 1-column DataFrame.")

    # Default column names = index
    if colnames is None:
        colnames = s.index
    else:
        # if len(colnames) != len(s): raise ValueError("Length of colnames must match length of vector.")
        pass

    # Expand vector → row-constant matrix
    idx = s.index
    arr = np.repeat(s.values.reshape(-1, 1), len(colnames), axis=1)

    return pd.DataFrame(arr, index=idx, columns=colnames, dtype=float)


def eval_triangle_formula(triangles: dict[str, pd.DataFrame],
                          formula: str,
                          div0_to_zero: bool = True) -> pd.DataFrame:
    """
    triangles: dict like {'A': tri_A, 'B': tri_B, ...} where each value is a pivoted DF
    formula:   e.g. 'D = A/B*1000' or 'A/B*1000' or 'A + B*C'
    div0_to_zero: if True, convert inf/NaN from division-by-zero to 0
    """
    # allow 'D = A/B*1000' or just 'A/B*1000'
    rhs = formula.split('=', 1)[-1].strip()

    # safety: no builtins; variables come from triangles dict
    env = {"__builtins__": {}}

    # element-wise eval; pandas aligns on index & columns automatically
    result = eval(rhs, env, triangles)

    if div0_to_zero:
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ensure numeric dtype (optional)
    return result.astype(float)


def _get_df(project_name):
    table_path = DLOOKUP(BASE_DICT['Team Profile'], project_name, 'Project Name', 'Table Path')
    table_name = os.path.basename(table_path)

    # DATA table cache (guarded)
    with DATA_DICT_LOCK:
        need_load = (table_name not in DATA_DICT) or (DATA_DICT.get(table_name + " - Version") is None) \
                    or (DATA_DICT[table_name + " - Version"] < File(table_path).last_modified_time)

    if need_load:
        # build outside lock if you want, but simplest is just load here
        with DATA_DICT_LOCK:
            load_to_DATA_DICT(table_path)

    # VPS cache (guarded)
    with VPS_DICT_LOCK:
        if project_name not in VPS_DICT:
            VPS_path = DLOOKUP(BASE_DICT['Team Profile'], project_name, 'Project Name', 'Project Settings')
            load_to_VPS_DICT(project_name, VPS_path)

    return DATA_DICT[table_name]


def _get_dataset_info(arg):
    # This apply to both vector and triangle
    project_name = arg['ProjectName']
    path = arg['Path']
    dataset_name = arg['DatasetName']

    df = _get_df(project_name)

    # Set user defined name (ResQ) to actual SQL table col names
    df_info = VPS_DICT[project_name]['Dataset Types']
    
    if dataset_name in df_info['Name'].values:
        source = df_info.loc[df_info['Name'] == dataset_name, 'Source'].iloc[0]
    else:
        write_lists_to_csv(arg['DataPath'], [[f'(dataset name not defined: {dataset_name})']])
        return
    
    output_data_format = df_info.loc[df_info['Name'] == dataset_name, 'Data Format'].iloc[0]

    # find all required table and column names
    df_info = VPS_DICT[project_name]['Source Table']
    required_datasets = split_formula(source)
    rsv_cls_col_names = df_info.loc[df_info['Significances'].isin(['Reserving Class']), 'Column Name'].unique().tolist()

    date_cols = []
    date_cols.append(DLOOKUP(df_info, 'Origin Date', 'Significances', 'Column Name'))
    date_cols.append(DLOOKUP(df_info, 'Development Date', 'Significances', 'Column Name'))

    max_sys_yrmo = int(df[date_cols[1]].max())

    required_datasets = [c for c in required_datasets if c in df.columns]  # remove invalid dataset names
    required_datasets = list(set(required_datasets))                       # remove duplicates

    # determine the categorical values need to be included/adjusted in the calculation
    df_info = VPS_DICT[project_name]['Reserving Class Types']
    name_lookup = {str(v).lower(): v for v in df_info['Name'].dropna()}
    splited_path = path.split('\\')
    included_rsv_cls_types = []  # use original value
    excluded_rsv_cls_types = []  # use negative value
    adjusted_rsv_cls_types = []  # change values to zero for EEX calculations

    level = 1
    for rsv_cls_type in splited_path:  # loop through N levels of reserving class
        if level > len(rsv_cls_col_names):
            break
        rsv_cls_type = name_lookup[rsv_cls_type.lower()]
        included_rsv_cls_types.append([])
        excluded_rsv_cls_types.append([])
        adjusted_rsv_cls_types.append([])

        if rsv_cls_type in df_info['Name'].values:
            included_rsv_cls_types[level-1].append(rsv_cls_type) # always include the input value itself
            formula     = df_info.loc[df_info['Name'] == rsv_cls_type, 'Formula'].iloc[0]
            eex_formula = df_info.loc[df_info['Name'] == rsv_cls_type, 'EEX Formula'].iloc[0]
            if formula == '':
                pass
            else: 
                if eex_formula != '':
                    adjusted_rsv_cls_types_level_x = list(set(split_formula(formula)) - set(split_formula(eex_formula)))
                    adjusted_rsv_cls_types[level-1] = adjusted_rsv_cls_types_level_x
                
                name_list = split_formula(formula)
                opts_list = split_formula_opts(formula)

                for i in range(len(name_list)):
                    name = name_list[i]
                    opt = opts_list[i]
                    if opt == '-':
                        excluded_rsv_cls_types[level-1].append(name)
                    included_rsv_cls_types[level-1].append(name)
        elif rsv_cls_type == '':
            pass
        else: 
            write_lists_to_csv(arg['DataPath'], [[f'(reserving class type not defined: {rsv_cls_type})']])
            return

        level += 1

    return df, date_cols, required_datasets, rsv_cls_col_names, \
           included_rsv_cls_types, excluded_rsv_cls_types, adjusted_rsv_cls_types, \
           source, output_data_format, max_sys_yrmo


def UDF_ADASProjectSettings(arg):
    project_name = arg['ProjectName']
    df = _get_df(project_name)

    df_info = VPS_DICT[project_name]['Source Table']
    date_cols = []
    date_cols.append(DLOOKUP(df_info, 'Origin Date', 'Significances', 'Column Name'))
    date_cols.append(DLOOKUP(df_info, 'Development Date', 'Significances', 'Column Name'))

    origin_start = int(df[date_cols[0]].min())
    origin_end = int(df[date_cols[0]].max())
    dev_end = int(df[date_cols[1]].max())

    data_list = [
        ['Name', project_name], 
        ['Origin Type', 'Accident'], 
        ['Origin Start Date', date(origin_start // 100, origin_start % 100, 1)], 
        ['Origin End Date', date(origin_end // 100, origin_end % 100, calendar.monthrange(origin_end // 100, origin_end % 100)[1])], 
        ['Development End Date', date(dev_end // 100, dev_end % 100, calendar.monthrange(dev_end // 100, dev_end % 100)[1])], 
        ['Origin Length', 12], 
        ['Development Length', 12], 
        ['Folder', 'ADAS Virtual Project']
    ]
    write_lists_to_csv(arg['DataPath'], data_list)


def UDF_ADASHeaders(arg):
    # Calculate Age & Origin Labels
    project_name = arg['ProjectName']
    org_len = int(arg['PeriodLength'])
    dev_len = int(arg['PeriodLength'])
    period_type = int(arg['periodType'])

    df = _get_df(project_name)
    df_info = VPS_DICT[project_name]['Source Table']
    date_cols = []
    date_cols.append(DLOOKUP(df_info, 'Origin Date', 'Significances', 'Column Name'))
    date_cols.append(DLOOKUP(df_info, 'Development Date', 'Significances', 'Column Name'))
    
    if period_type == 0: # Origin Period

        acc_yrmo_all = list(map(int, sorted(df[date_cols[0]].unique())))
        org_index_grp = [tuple(acc_yrmo_all[i: i+org_len]) for i in range(0, len(acc_yrmo_all), org_len)]
            
        org_label = [_get_org_label(i[0], org_len) for i in org_index_grp]

        return write_lists_to_csv(arg['DataPath'], [org_label])
    
    elif period_type == 1: # Development Period

        if (dev_len == 'Default') or (org_len % dev_len != 0): 
            dev_len = org_len

        dev_cnt = round(len(df[date_cols[0]].unique())/dev_len)
        first_mon = int(df[date_cols[1]].max()%100)

        dev_label = list(range(first_mon, dev_cnt*dev_len+1, dev_len))

        for i in range(1, 999):
            prior_mon = first_mon - dev_len*i
            if prior_mon > 0:
                dev_label = [prior_mon] + dev_label
            else:
                break

        dev_label = list(map(lambda x:f"{x}m", dev_label))
        return write_lists_to_csv(arg['DataPath'], [dev_label])
    
    else:
        return write_lists_to_csv(arg['DataPath'], [['(invalid input: periodType)']])


def _filter_main_table(df, date_cols, rsv_cls_col_names, included_rsv_cls_types, required_datasets):
    """
    df: original DataFrame
    rsv_cls_col_names: list of df column names, length N
    included_rsv_cls_types: list of lists (may include empty lists)
    required_datasets: list of other columns to keep

    If included_rsv_cls_types[i] is empty, no filtering is applied for that column.
    """

    # if len(rsv_cls_col_names) != len(included_rsv_cls_types):
    #     raise ValueError("Lengths of rsv_cls_col_names and included_rsv_cls_types must match.")

    fixed_levels = len(rsv_cls_col_names)
    input_levels = len(included_rsv_cls_types)

    if fixed_levels < input_levels:
        included_rsv_cls_types = included_rsv_cls_types[:len(rsv_cls_col_names)]
    elif fixed_levels > input_levels:
        included_rsv_cls_types = included_rsv_cls_types + [''] * (fixed_levels-input_levels)

    # Start with full mask
    mask = True

    # Add filters dynamically
    for col, allowed_values in zip(rsv_cls_col_names, included_rsv_cls_types):
        # If empty list → skip filter for this level
        if allowed_values:
            mask &= df[col].isin(allowed_values)

    # Build final column list
    cols = date_cols + rsv_cls_col_names + required_datasets

    return df.loc[mask, cols]


def UDF_ADASTri(arg):
    org_len = arg['OriginLength']
    dev_len = arg['DevelopmentLength']
    cumulative = arg['Cumulative']

    # initialize
    if org_len == 'Default': org_len = 12

    # Get a subset dataframe based on a user's request
    df, date_cols, required_datasets, rsv_cls_col_names, \
    included_rsv_cls_types, excluded_rsv_cls_types, adjusted_rsv_cls_types, \
    source, output_data_format, max_sys_yrmo = _get_dataset_info(arg) 

    max_sys_month = max_sys_yrmo % 100

    df1 = _filter_main_table(df, date_cols, rsv_cls_col_names, included_rsv_cls_types, required_datasets)

    # Row Adjustments (Excluded Values) -- multiply value by -1
    num_cols = df1.select_dtypes(include=[np.number]).columns
    dataset_cols = [col for col in num_cols if col not in date_cols]  # all numerical field need to be adjusted

    for i in range(len(excluded_rsv_cls_types)):
        excluded_rsv_cls_types_level_x = excluded_rsv_cls_types[i]
        if excluded_rsv_cls_types_level_x == []: 
            continue
        for value in excluded_rsv_cls_types_level_x:
            df1.loc[df1[rsv_cls_col_names[i]].isin([value]), dataset_cols] *= -1

    # Row Adjustments (EEX aggregation) -- set value to 0
    if 'Earned_Exposure' in required_datasets:
        adjusted_rsv_cls_types_level_x = adjusted_rsv_cls_types[4]  # level 5: IBNRCAT
        for value in adjusted_rsv_cls_types_level_x:
            df1.loc[df1[rsv_cls_col_names[4]].isin([value]), ['Earned_Exposure']] *= 0

    # Calculate Age & Origin Labels
    df1['Age'] = df1.apply(lambda row: _calc_age(row[date_cols[0]], row[date_cols[1]]), axis=1)

    if (dev_len == 'Default') or (org_len % dev_len != 0): 
        dev_len = org_len
        
    dev_cnt = round(len(df[date_cols[0]].unique())/dev_len)
    first_mon = int(df[date_cols[1]].max()%100)
    
    dev_label = list(range(first_mon, dev_cnt*dev_len+1, dev_len))
    for i in range(1, 999):
        prior_mon = first_mon - dev_len*i
        if prior_mon > 0:
            dev_label = [prior_mon] + dev_label
        else:
            break
            
    acc_yrmo_all = list(map(int, sorted(df[date_cols[0]].unique())))
    org_index_grp = [tuple(acc_yrmo_all[i: i+org_len]) for i in range(0, len(acc_yrmo_all), org_len)]
    org_index_map = {val: group[0] for group in org_index_grp for val in group}
    org_label = [_get_org_label(i[0], org_len) for i in org_index_grp]
    
    # df1['Org*Grp'] = df1[date_cols[0]].apply(_get_org_label)
    df1['Org*Grp'] = df1.apply(lambda row: _get_org_label(row[date_cols[0]], org_len), axis=1)

    df1['Org*Start'] = df1[date_cols[0]].map(org_index_map)
    df1['Age*'] = df1.apply(lambda row: _calc_age(row['Org*Start'], row[date_cols[1]]), axis=1)  # The 'real' age for org_period grouping
    df1['Age*Grp'] = df1['Age*'].apply(lambda x: min([i for i in dev_label if i >= x]))

    df1 = df1.groupby(['Org*Grp', 'Age*Grp'])[required_datasets].sum().reset_index()

    # Create individual non-calculated triangles
    triangles  = {}
    
    for name in required_datasets:
        df2 = df1.pivot_table(
            index = df1['Org*Grp'], 
            columns = df1['Age*Grp'], 
            values = name,
            aggfunc = 'sum', 
            fill_value = 0
        )
        df2 = df2.reindex(index=org_label, columns=dev_label).fillna(0)

        if cumulative == True: 
            df2 = df2.cumsum(axis=1)

        data_format = DLOOKUP(VPS_DICT[arg['ProjectName']]['Dataset Types'], name, 'Source', 'Data Format')
        if data_format == 'Vector':
            df2 = vector_to_triangle(df2.iloc[:, [0]], dev_label)

        triangles[name] = df2
    
    # Calculated Triangle
    df2 = eval_triangle_formula(triangles, source)  

    # Clean Format
    n_rows = df2.shape[0]
    for i, acc in enumerate(df2.index):
        max_dev_age = (n_rows - i) * int((org_len/dev_len))

        if org_len == 3 and dev_len == 1:
            max_dev_age = max_dev_age - (12 - max_sys_month)

        if dev_len == 3:
            if max_sys_month in [1, 2, 3]:
                max_dev_age = max_dev_age - 3
            elif max_sys_month in [4, 5, 6]:
                max_dev_age = max_dev_age - 2
            elif max_sys_month in [7, 8, 9]:
                max_dev_age = max_dev_age - 1

        if dev_len == 6 and max_sys_month <= 6:
            max_dev_age = max_dev_age - 1

        if max_dev_age < 0:
            max_dev_age = 0

        df2.loc[acc, dev_label[max_dev_age:]] = np.nan

    if output_data_format == 'Vector' or arg['Function'] == 'ADASVec':
        df2 = df2.iloc[:, [0]]

    # Output
    _export_dataframe(df2, arg)


def _export_dataframe(df, arg):
    data_path = arg['DataPath']
    file_name = os.path.basename(data_path)
    folder = os.path.dirname(data_path)
    tmp_folder = folder + '\\tmp'
    tmp_data_path = tmp_folder + '\\' + file_name
    
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
    except:
        pass

    df.to_csv(tmp_data_path, index=False, header=False)

    if os.path.exists(data_path):
        os.remove(data_path)

    os.rename(tmp_data_path, data_path)


class RequestHandler(FileSystemEventHandler):

    def on_moved(self, event):
        if event.is_directory:
            return
        if not event.dest_path.lower().endswith(".txt"):
            return

        file_path = event.dest_path

        # Don’t block watchdog’s thread; queue to worker pool
        EXECUTOR.submit(self.process_file, file_path)
        
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
        except:
            # print(f'\n* request sent to another agent')
            return

        try:
            project_name = arg['ProjectName']
            settings_file = DLOOKUP(BASE_DICT['Team Profile'], project_name, 'Project Name', 'Project Settings')
        except:
            write_lists_to_csv(arg['DataPath'], [[f'(project not found: {project_name})']])
            return

        try:
            safe_remove(file_path)
        except: # Already removed by another agent
            return

        print(f"\n> {get_current_time()} \n> new request # {robot_id} # user [{arg['UserName']}]")

        # Check VPS Updates (guarded)
        with VPS_DICT_LOCK:
            if project_name + " - Version" in VPS_DICT:
                if VPS_DICT[project_name + " - Version"] < File(settings_file).last_modified_time:
                    load_to_VPS_DICT(project_name, settings_file)
                    print(f">>> Virtual Project Settings Updated -> [{settings_file}]\n")
            # If missing, _get_df() will load it later; or you can proactively load it here.

        # Go to Functions
        try:
            if arg['Function'] in ['ADASTri', 'ADASVec']:
                UDF_ADASTri(arg)
            elif arg['Function'] == 'ADASProjectSettings':
                UDF_ADASProjectSettings(arg)
            elif arg['Function'] == 'ADASHeaders':
                UDF_ADASHeaders(arg)
            else:
                write_lists_to_csv(arg['DataPath'], [['(invalid function name)']])
        except Exception as e:
            err_msg = f"(error: {str(e).upper()})"
            print(err_msg)
            # write_lists_to_csv(arg['DataPath'], [[err_msg]])
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
    preload_table_list = BASE_DICT['Team Profile'][BASE_DICT['Team Profile']['Preload'] == 'Y']['Table Path'].tolist()

    # inital load
    for csv_path in preload_table_list:
        project_name  = DLOOKUP(BASE_DICT['Team Profile'], csv_path, 'Table Path', 'Project Name')
        settings_file = DLOOKUP(BASE_DICT['Team Profile'], csv_path, 'Table Path', 'Project Settings')
        load_to_VPS_DICT(project_name, settings_file)
        load_to_DATA_DICT(csv_path)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_txt(id_path, {'Server': robot_id, 'Last seen': current_time})

    try:
        while True:

            if not os.path.exists(id_path):
                observer.stop(); break

            if read_txt(command_path)['KILL_ALL_AGENTS'] in ['True', '1']:
                File(id_path).delete()
                observer.stop(); break
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            # Update Status
            arg_1 = read_txt(id_path)
            arg_1['Last seen'] = current_time
            write_txt(id_path, arg_1)

            # Check Base Settings (New Version Available?)
            if BASE_DICT["Team Profile - Version"] < File(team_profile_path).last_modified_time:
                load_BASE_DICT()
                print(">>> Team Profile Updated\n")

            time.sleep(5)

    except KeyboardInterrupt:
        observer.stop()
        
    observer.join()


start_monitoring("E:\\ADAS\\requests")
