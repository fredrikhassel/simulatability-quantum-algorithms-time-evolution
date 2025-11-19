#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple


def parse_index_and_suffix(name: str, prefix: str) -> Tuple[int, str]:
    """Extract the numeric index and suffix from a file name."""
    if not name.startswith(prefix):
        raise ValueError(f"{name} does not start with {prefix}")
    base_len = len(prefix)
    dash_pos = name.find("-", base_len)
    if dash_pos == -1:
        number_part = name[base_len:]
        suffix = ""
    else:
        number_part = name[base_len:dash_pos]
        suffix = name[dash_pos:]
    index = int(number_part) if number_part.isdigit() else 1
    return index, suffix


def find_next_index(folder: str, suffix: str) -> int:
    """Find the next available index for gates_arr* files with a given suffix."""
    max_index = 0
    for fname in os.listdir(folder):
        if not fname.startswith("gates_arr"):
            continue
        try:
            idx, suf = parse_index_and_suffix(fname, "gates_arr")
        except ValueError:
            continue
        if suf == suffix and idx > max_index:
            max_index = idx
    return max_index + 1 if max_index > 0 else 2


def load_json(path: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    """Save a JSON object to disk."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def split_numeric_keys(keys: List[str]) -> Tuple[List[str], List[str]]:
    """Split a sorted list of numeric keys into two halves."""
    n = len(keys)
    if n < 2:
        raise ValueError("Need at least two entries to split")
    split_at = n // 2
    return keys[:split_at], keys[split_at:]


def reindex_subset(data: Dict, old_keys: List[str]) -> Dict:
    """Create a new dict with values from old_keys, reindexed from 1..m."""
    new_data: Dict[str, object] = {}
    for i, old_k in enumerate(old_keys, start=1):
        new_data[str(i)] = data[old_k]
    return new_data


def split_pair(
    folder: str,
    gates_fname: str,
    sign_fname: str,
) -> None:
    """Split one gates_arr/sign_list pair into two files."""
    gates_path = os.path.join(folder, gates_fname)
    sign_path = os.path.join(folder, sign_fname)

    gates_data = load_json(gates_path)
    sign_data = load_json(sign_path)

    gates_numeric = sorted(
        (k for k in gates_data.keys() if k.isdigit()), key=lambda x: int(x)
    )
    sign_numeric = sorted(
        (k for k in sign_data.keys() if k.isdigit()), key=lambda x: int(x)
    )

    if gates_numeric != sign_numeric:
        print(
            f"Skipping {gates_fname}: gates/sign numeric keys differ.",
            file=sys.stderr,
        )
        return

    if len(gates_numeric) < 2:
        print(
            f"Skipping {gates_fname}: not enough entries to split.",
            file=sys.stderr,
        )
        return

    first_keys, second_keys = split_numeric_keys(gates_numeric)

    idx, suffix = parse_index_and_suffix(gates_fname, "gates_arr")
    next_index = find_next_index(folder, suffix)

    new_gates_fname = f"gates_arr{next_index}{suffix}"
    new_sign_fname = f"sign_list{next_index}{suffix}"

    overhead = sign_data.get("overhead")

    gates_first = reindex_subset(gates_data, first_keys)
    gates_second = reindex_subset(gates_data, second_keys)

    # Copy any non-numeric metadata keys in gates_arr to both halves
    for k, v in gates_data.items():
        if not k.isdigit():
            gates_first[k] = v
            gates_second[k] = v

    sign_first = reindex_subset(sign_data, first_keys)
    sign_second = reindex_subset(sign_data, second_keys)

    if overhead is not None:
        sign_first["overhead"] = overhead
        sign_second["overhead"] = overhead

    save_json(gates_path, gates_first)
    save_json(os.path.join(folder, new_gates_fname), gates_second)

    save_json(sign_path, sign_first)
    save_json(os.path.join(folder, new_sign_fname), sign_second)

    print(
        f"Split {gates_fname} + {sign_fname} -> "
        f"{gates_fname} (first half) and {new_gates_fname} (second half)"
    )


def process_folder(folder: str) -> None:
    """Process all gates_arr/sign_list JSON pairs in a folder."""
    all_files = os.listdir(folder)
    gates_files = sorted(f for f in all_files if f.startswith("gates_arr") and f.endswith(".json"))

    if not gates_files:
        print("No gates_arr*.json files found.", file=sys.stderr)
        return

    for gates_fname in gates_files:
        sign_fname = gates_fname.replace("gates_arr", "sign_list", 1)
        sign_path = os.path.join(folder, sign_fname)
        if not os.path.exists(sign_path):
            print(
                f"Warning: matching {sign_fname} not found for {gates_fname}, skipping.",
                file=sys.stderr,
            )
            continue
        split_pair(folder, gates_fname, sign_fname)


def main() -> None:
    """Parse CLI arguments and run the splitter."""
    parser = argparse.ArgumentParser(
        description="Split gates_arr/sign_list JSON files into smaller pieces."
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing gates_arr*.json and sign_list*.json files.",
    )
    args = parser.parse_args()
    process_folder(os.path.abspath(args.folder))


if __name__ == "__main__":
    main()
