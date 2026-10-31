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
"""Tools of the orchestrating agent.

* ``get_pdf_reaction``: reaction-mining agent, extracts candidate reactions from one paper and records the paper as
  ``source``.
* ``check_json_reactions``: automatic screening agent, rule-based verification followed by LLM semantic filtering of
  the records of each paper against the paper text.
* File tools for listing, reading, renaming and writing files.

Directories (environment variables): ``PDF_DIR`` is the root of the PDF folders the agent may read (default: the
working directory), ``RESULTS_DIR`` holds extracted reactions and ``SCREENED_DIR`` screened reactions.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

import utils

base_dir_pdf = Path(os.getenv("PDF_DIR", "."))
base_dir_results = Path(os.getenv("RESULTS_DIR", "results/extracted"))
base_dir_screened = Path(os.getenv("SCREENED_DIR", "results/screened"))

task1_sub_sys = utils.load_prompt("system_prompt/task1_sub_sys.txt")
task2_sub_sys = utils.load_prompt("system_prompt/task2_sub_sys.txt")
SCREENING_BATCH_SIZE = 20


def _inside(base_dir: Path, name: str) -> Path:
    """Resolve ``name`` under ``base_dir``."""
    path = (base_dir / name).resolve()
    if not path.is_relative_to(base_dir.resolve()):
        raise ValueError(f"'{name}' is outside {base_dir}")
    return path


def read_file(name: str) -> str:
    """
    Return file content. If not exist, return error message.
    """
    print(f"(read_file {name})")
    try:
        with open(_inside(base_dir_pdf, name), encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"An error occurred: {e}"


def list_files(folder: str) -> list[str]:
    print("(list_file)")
    try:
        base_dir = _inside(base_dir_pdf, folder)
    except ValueError as e:
        return [f"An error occurred: {e}"]
    file_list = [str(item.relative_to(base_dir)) for item in base_dir.rglob("*") if item.is_file()]
    print(f"Files: {file_list}")
    return file_list


def rename_file(name: str, new_name: str) -> str:
    print(f"(rename_file {name} -> {new_name})")
    try:
        new_path = _inside(base_dir_pdf, new_name)
        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(_inside(base_dir_pdf, name), new_path)
        return f"File '{name}' successfully renamed to '{new_name}'."
    except Exception as e:
        return f"An error occurred: {e}"


def _pdf_text(name: str) -> str:
    with fitz.open(str(_inside(base_dir_pdf, name))) as pdf:
        return "".join(page.get_text() for page in pdf)


def _with_source(content: str, name: str) -> str:
    """Add the source paper to every extracted record so that the screening agent can check it against the paper."""
    try:
        records = json.loads(utils.extract_json_from_markdown(content))
    except json.JSONDecodeError:
        return content
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return content
    for record in records:
        if isinstance(record, dict):
            record["source"] = name
    return json.dumps(records, ensure_ascii=False, indent=2)


def get_pdf_reaction(name: str) -> str:
    print(f"(get_pdf_reaction {name})")
    if not name.endswith(".pdf"):
        return "Error: The file is not a PDF."
    try:
        content = utils.chat(task1_sub_sys, _pdf_text(name))
        print(content)
        return _with_source(content, name)
    except Exception as e:
        return f"An error occurred: {e}"


def write_file(name: str, content: str) -> str:
    print(f"(write_file {name})")
    try:
        file_path = _inside(base_dir_results, name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"File '{name}' written successfully in {base_dir_results}."
    except Exception as e:
        return f"An error occurred: {e}"


def create_json_file(name: str, content: str) -> str:
    print(f"(create_json_file {name})")
    try:
        file_path = _inside(base_dir_results, name)
        data = []
        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []

        try:
            new_data = json.loads(content)
            if isinstance(new_data, dict):
                data.append(new_data)
            elif isinstance(new_data, list):
                data.extend(new_data)
        except Exception as e:
            return f"Error parsing content: {e}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"File '{name}' updated successfully."
    except Exception as e:
        return f"An error occurred: {e}"


def check_json_reactions(input_file: str, output_file: str) -> str:
    print(f"(check_json_reactions {input_file} -> {output_file})")
    try:
        input_file_path = _inside(base_dir_results, input_file)
        output_file_path = _inside(base_dir_screened, output_file)
        with open(input_file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: JSON file is not an array.")
            return "Error: JSON file is not an array."

        # Rule-based verification
        valid_data = []
        for item in data:
            try:
                if "reaction" in item and utils.check_reaction_format(item["reaction"]):
                    valid_data.append(item)
            except Exception as e:
                print("error:", e)

        # LLM-assisted semantic filtering against the source paper
        by_source = defaultdict(list)
        for item in valid_data:
            by_source[item.get("source")].append(item)
        all_checked = []
        for source, records in by_source.items():
            try:
                paper = f"Source paper:\n{_pdf_text(source)}\n\n"
            except Exception as e:
                print(f"error: cannot read the source paper {source!r} ({e}), checking the records alone")
                paper = ""
            for i in range(0, len(records), SCREENING_BATCH_SIZE):
                batch = records[i : i + SCREENING_BATCH_SIZE]
                print(f" Checking {len(batch)} reactions from {source}...")
                try:
                    records_text = json.dumps(batch, ensure_ascii=False, indent=2)
                    checked_data = utils.chat(task2_sub_sys, f"{paper}Extracted records:\n{records_text}")
                    print("checked_data:", checked_data)
                    all_checked.extend(json.loads(utils.extract_json_from_markdown(checked_data)))
                except Exception as e:
                    print("error:", e)

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(all_checked, f, ensure_ascii=False, indent=2)
        return f"Saved {len(all_checked)} of {len(data)} reactions to {output_file}."
    except Exception as e:
        return f"An error occurred: {e}"
