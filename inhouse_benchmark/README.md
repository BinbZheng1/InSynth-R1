# In-house benchmark

Temporally held-out precursor selection benchmark (Appendix A.1 of the paper): inorganic synthesis reactions
reported in 2020–2025, after the period covered by the public reaction data used for training.

* `data_sample.json`: about 500 reactions of the benchmark, released as reference examples.
* `main.py`, `tools.py`, `utils.py`, `prompt/`: the hierarchical multi-agent extraction and screening pipeline
  used to build it.

## Data

Each record has the fields

| field | description |
| --- | --- |
| `reaction` | `target <- precursor1 + precursor2 + ...` |
| `heat_temperature` | reported heating temperatures and durations |
| `operations` | ordered synthesis operations, e.g. `["mix", "grind", "calcine"]` |
| `doi` | source paper |

```json
{
  "reaction": "SnS2 <- Sn + S",
  "heat_temperature": "150°C for 5 min then 200°C for 45 min, followed by 300°C for 2 h",
  "operations": ["mix", "microwave_irradiate", "disperse", "stir", "add_solvent", "centrifuge", "wash", "dry", "anneal"],
  "doi": "https://doi.org/10.1016/j.jssc.2021.122022"
}
```

To build the evaluation set, pass the file to the data preparation script. Formulas are normalized, duplicates are
removed and records whose target–precursor pair also occurs in the public training or test set are dropped; the
counts are written to `split_stats.json`.

```bash
REACTIONS=... SYNTHESIZABILITY=... EVAL_REACTIONS=inhouse_benchmark/data_sample.json \
    bash examples/insynth_r1/prepare_data.sh   # -> data/insynth/eval/inhouse.parquet
```

## Extraction and screening pipeline

An orchestrating agent (`main.py`, system prompt `prompt/system_prompt/system_prompt.txt`) calls two sub-agents as
tools, all backed by Qwen3-plus:

1. **Reaction-mining agent** (`tools.get_pdf_reaction`, `prompt/system_prompt/task1_sub_sys.txt`): extracts the
   text of a paper with PyMuPDF and returns candidate reactions with the target, precursors, heating conditions
   and operations; each record also keeps the PDF path as `source`.
2. **Automatic screening agent** (`tools.check_json_reactions`, `prompt/system_prompt/task2_sub_sys.txt`):
   deterministic rule-based verification (`utils.check_reaction_format`: `target <- precursor + ...` format, a
   single target without a variable `x`, at least two precursors, only element symbols (no placeholders such as
   `Ln`), every precursor shares an element with the target, every target element is supplied by the precursors
   and every metal of the precursors appears in the target), followed by LLM-assisted semantic filtering: the
   records of each paper are checked together with the paper text, in batches of 20 records, for whether the
   compounds and their roles are supported by the paper and the target–precursor relationship is clear.

The retained records were then reviewed by three chemistry experts (majority vote).

### Usage

```bash
pip install -r inhouse_benchmark/requirements.txt
cd inhouse_benchmark
export LLM_BASE_URL=...  # OpenAI-compatible endpoint
export LLM_API_KEY=...
python main.py           # interactive: 1 = extract, 2 = screen, 3 = extract + screen
```

The settings can also be put in a `.env` file in this folder.

| variable | default | description |
| --- | --- | --- |
| `LLM_MODEL` | `qwen-plus` | model id of Qwen3-plus on the endpoint |
| `LLM_BASE_URL`, `LLM_API_KEY` | | OpenAI-compatible endpoint |
| `PDF_DIR` | `.` | root of the PDF folders entered in `main.py` |
| `RESULTS_DIR` | `results/extracted` | extracted reactions |
| `SCREENED_DIR` | `results/screened` | screened reactions |
