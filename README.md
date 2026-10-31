# InSynth-R1

Code for InSynth-R1, a reasoning LLM for inorganic synthesis planning (synthesizability prediction and precursor
selection), trained with multi-stage supervised fine-tuning (MSFT) and Rubric-augmented Curriculum Policy
Optimization (RCPO). Training is built on [verl](https://github.com/verl-project/verl).

## Repository layout

| path | content |
| --- | --- |
| `insynth_r1/data/` | data preprocessing, prompt templates and splits |
| `insynth_r1/coldstart/` | Stage-2 trajectory generation, complexity–utility scoring and selection (Algorithm 1) |
| `insynth_r1/reward/` | rubric reward: R_accuracy, format, key-step reward R_step and length reward R_length |
| `insynth_r1/algo/rcpo.py` | RCPO advantage (Eq. 7–10) and clipped surrogate (Eq. 11–13) |
| `insynth_r1/trainer/` | curriculum cache and the RCPO trainer |
| `insynth_r1/eval/` | generation and metrics |
| `insynth_r1/config/rcpo_trainer.yaml` | RCPO training configuration |
| `examples/insynth_r1/` | scripts for every stage |
| `inhouse_benchmark/` | temporally held-out benchmark samples and its extraction pipeline |
| `verl/` | training framework |

## Installation

```bash
uv sync --extra fsdp --extra vllm    # or: pip install -e ".[fsdp,vllm]"
```

## Pipeline

```bash
# 1. Data
REACTIONS=... SYNTHESIZABILITY=... EVAL_REACTIONS=inhouse_benchmark/data_sample.json \
    bash examples/insynth_r1/prepare_data.sh

# 2. MSFT Stage-1: domain fine-tuning
bash examples/insynth_r1/run_stage1_sft.sh
bash examples/insynth_r1/merge_checkpoint.sh ckpts/stage1_sft/global_step_<N> ckpts/stage1_merged

# 3. MSFT Stage-2: rejective fine-tuning
bash examples/insynth_r1/build_coldstart.sh generate   # then score, select, build
bash examples/insynth_r1/run_stage2_sft.sh
bash examples/insynth_r1/merge_checkpoint.sh ckpts/stage2_sft/global_step_<N> ckpts/msft_merged

# 4. RCPO (start the QwQ-32B key-step judge first, see the script header)
bash examples/insynth_r1/run_rcpo.sh
bash examples/insynth_r1/merge_checkpoint.sh ckpts/rl/<experiment>/global_step_<N>/actor ckpts/insynth-r1

# 5. Evaluation
MODEL_PATH=ckpts/insynth-r1 bash examples/insynth_r1/eval.sh outputs/insynth-r1
```

Each script lists its settings and environment variables in its header.

## Tests

```bash
python -m pytest tests/insynth_r1
```

## License

Apache License 2.0 (`LICENSE`); `Notice.txt` is the notice of the verl framework.
