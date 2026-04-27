"""Table summary generation and caching."""
from __future__ import annotations

import os
import json
from typing import Any, Dict

import pandas as pd

from app_server import config


def is_cache_valid(csv_path: str, cache_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    csv_mtime = os.stat(csv_path).st_mtime
    cache_mtime = os.stat(cache_path).st_mtime
    return cache_mtime > csv_mtime


def generate_table_summary(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    file_size = st.st_size

    df = pd.read_csv(path)
    row_count = len(df)

    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_data = df[col].dropna()

        if "int" in dtype:
            friendly_type = "Integer"
            if len(col_data) > 0:
                min_val = int(col_data.min())
                max_val = int(col_data.max())
                values_str = f"Range: ({min_val:,}, {max_val:,})"
            else:
                values_str = "(empty)"
        elif "float" in dtype:
            friendly_type = "Float"
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                if abs(max_val) >= 1000 or abs(min_val) >= 1000:
                    values_str = f"Range: ({min_val:,.2f}, {max_val:,.2f})"
                else:
                    values_str = f"Range: ({min_val:.4f}, {max_val:.4f})"
            else:
                values_str = "(empty)"
        elif "object" in dtype:
                friendly_type = "String"
                distinct = col_data.unique().tolist()
                distinct_count = len(distinct)
                if distinct_count <= 10:
                    values_str = ", ".join(str(v) for v in sorted(distinct, key=str))
                else:
                    sample = sorted(distinct, key=str)[:10]
                    values_str = f"{distinct_count} distinct: {', '.join(str(v) for v in sample)}..."
        elif "datetime" in dtype:
            friendly_type = "DateTime"
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                values_str = f"Range: {min_val} - {max_val}"
            else:
                values_str = "(empty)"
        elif "bool" in dtype:
            friendly_type = "Boolean"
            values_str = "True, False"
        else:
            friendly_type = dtype
            values_str = "(unknown)"

        columns.append({
            "name": str(col),
            "dtype": dtype,
            "type": friendly_type,
            "values": values_str,
        })

    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"

    return {
        "ok": True,
        "path": path,
        "row_count": row_count,
        "column_count": len(columns),
        "file_size": file_size,
        "file_size_str": size_str,
        "columns": columns,
        "csv_mtime": st.st_mtime,
    }
