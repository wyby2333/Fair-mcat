"""Console and JSON reporting helpers for Fair-MCAT experiments."""

import json


def mark_best(value, best_value):
    if value == best_value:
        return f"{value:<15.4f} ★"
    return f"{value:<15.4f}"


def save_json_result(result_file, results):
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
