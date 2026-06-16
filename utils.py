"""工具函数。"""

import csv
import os
from typing import List, Dict


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def save_train_log(log_path: str, records: List[Dict]) -> None:
    """保存训练日志到 CSV 文件。"""
    ensure_dir(os.path.dirname(log_path))
    if not records:
        return

    fieldnames = records[0].keys()
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def append_train_log(log_path: str, record: Dict) -> None:
    """追加一条训练记录到 CSV。"""
    ensure_dir(os.path.dirname(log_path))
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
