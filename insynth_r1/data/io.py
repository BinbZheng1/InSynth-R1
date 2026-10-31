# Copyright 2026 InSynth-R1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Record I/O for jsonl / json / csv / parquet files."""

import json
import os
from pathlib import Path


def read_records(path: str) -> list[dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data["data"]
    import pandas as pd

    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return frame.to_dict(orient="records")


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def write_parquet(records: list[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def as_list(value) -> list:
    """Precursor lists may come back from parquet/csv as numpy arrays or ``"A + B"`` strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return [str(v).strip() for v in json.loads(text)]
        return [p.strip() for p in text.split("+") if p.strip()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(v).strip() for v in value if str(v).strip()]
