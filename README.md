# 18842-gr00t

LoRA fine-tuning of NVIDIA **GR00T N1.7** (3B vision-language-action model) on the SO-101
**Vial-to-Rack** dataset, on a single 16 GB GPU (RTX 5080), using LeRobot 0.6.1.

It follows the same workflow as the class guide
[Training and Deploying a Policy on the Vial-to-Rack Dataset](https://github.com/z4ziad/autonomous_robotics/blob/main/learning_policies_lab/Training%20and%20Deploying%20a%20Policy%20on%20the%20Vial-to-Rack%20Dataset.md)
(ACT / Diffusion / SmolVLA): same conda environment, same dataset, same `lerobot-rollout` command
on the arm. The only differences are the training script and one extra **merge** step.

| File | What it does |
| --- | --- |
| `train_groot_lora.py` | Drop-in for `lerobot-train`: same options, adds LoRA for GR00T at run time |
| `merge_groot_lora.py` | Folds the LoRA adapter into the model → a normal policy folder for `lerobot-rollout` |

No file of the installed `lerobot` package is modified; the scripts only adjust LeRobot's
existing LoRA (PEFT) support in memory for GR00T (see [How it works](#how-it-works)).

---

## 1. One-time setup

### 1.1 Get the code

```bash
git clone https://github.com/rulihongran/18842-gr00t.git ~/18842-gr00t
```

### 1.2 Python environment

Use the class `lerobot` conda environment (LeRobot 0.6.1, PyTorch with CUDA, NVIDIA driver ≥ 580)
and add the GR00T extras once:

```bash
conda activate lerobot
pip install 'lerobot[groot]==0.6.1'
```

This only adds packages (transformers, peft, decord, dm-tree, timm, …); it does not change
lerobot, torch or numpy.

### 1.3 Hugging Face access (required)

GR00T N1.7 is built on NVIDIA's **Cosmos-Reason2-2B** vision-language model, which is gated:

1. Log in on <https://huggingface.co> with the account you will train with.
2. Open <https://huggingface.co/nvidia/Cosmos-Reason2-2B>, accept the license and submit the request.
3. Wait until the page says access is granted (check <https://huggingface.co/settings/gated-repos>).

The same account must also be able to read the dataset (member of the `autorobo` organization).
Then log in on the training computer:

```bash
hf auth login          # paste a READ token of that account
```

> **`HF_TOKEN` overrides `hf auth login`.** If your `~/.bashrc` exports `HF_TOKEN` for a
> different account, run `unset HF_TOKEN` in the training terminal (shown in every command below),
> otherwise the download fails with `403 ... gated repo ... Cosmos-Reason2-2B`.

Check access (all three must print `OK`):

```bash
unset HF_TOKEN
python -c "
from huggingface_hub import auth_check
for r, t in [('nvidia/GR00T-N1.7-3B', None), ('nvidia/Cosmos-Reason2-2B', None), ('autorobo/vials_all_v1', 'dataset')]:
    try: auth_check(r, repo_type=t); print('OK  ', r)
    except Exception as e: print('FAIL', r, type(e).__name__)
"
```

---

## 2. Run the fine-tuning

### 2.1 Set up the terminal (every new terminal)

```bash
conda activate lerobot
unset HF_TOKEN                           # use the account with Cosmos access (see 1.3)
export HF_ORG=autorobo
export DATASET=vials_all_v1              # or vials_real_v1 / vials_sim_v1
export TEAM=t1                           # your team: t1 ... t5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=groot_lora_${DATASET}_${TEAM}

hf auth whoami                           # must show the account with Cosmos access
echo ${HF_ORG}/${DATASET}  ${RUN}        # must print: autorobo/vials_all_v1  groot_lora_vials_all_v1_t1
nvidia-smi                               # no other training (python process) on the GPU
```

If `echo` prints `/vials_all_v1`, an `export` line was skipped — run them again.

### 2.2 Train

```bash
python ~/18842-gr00t/train_groot_lora.py \
  --dataset.repo_id=${HF_ORG}/${DATASET} \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.model_params_fp32=false \
  --peft.r=32 \
  --peft.lora_alpha=64 \
  --output_dir=${HOME}/vials/outputs/${RUN} \
  --job_name=${RUN} \
  --steps=30000 \
  --batch_size=8 \
  --save_freq=5000 \
  --log_freq=200
```

The first run downloads the base model (`nvidia/GR00T-N1.7-3B` + Cosmos, ~10 GB) and the dataset.
Run it inside `tmux` so it survives a closed terminal. Add `--wandb.enable=true --wandb.project=vials`
for training curves (after `wandb login`).

| Option | What it does |
| --- | --- |
| `--policy.type=groot` | GR00T N1.7, initialised from `nvidia/GR00T-N1.7-3B`. |
| `--policy.model_params_fp32=false` | Keeps the frozen 3.4B weights in bf16. In fp32 they alone take ~14 GB, so this is **required on 16 GB**. |
| `--peft.r=32 --peft.lora_alpha=64` | LoRA rank and scale (scale = alpha / r = 2). Higher rank = more capacity and memory. |
| `--batch_size=8` | 14.2 GB peak on an RTX 5080. Use `4` (12.4 GB) on "CUDA out of memory". |
| `--steps=30000` | ≈ 2 h at batch 8 (~4.5 steps/s); ≈ 1.3 h at batch 4 (~6.4 steps/s). |
| `--save_freq=5000` | A checkpoint every 5000 steps (each ~0.75 GB). |
| `--policy.optimizer_lr` | Not set → default **1e-4** peak (AdamW, weight decay 1e-5, grad-clip 1.0), 500 warm-up steps, then cosine decay to 0 at `--steps`. |

**Reading the progress line:** `loss` starts around 1.3 and should fall steadily (it is a
flow-matching loss; not comparable to ACT's). `lr` is tiny at first and reaches 1e-4 at step 500.
`mem_gb` should stay below ~14.

**Interrupted run:** `--resume=true` is **not supported** for LoRA runs (the script stops with a
message). Merge and test the last saved checkpoint, or start again with a new name
(`export RUN=${RUN}_b`).

Output:

```
~/vials/outputs/groot_lora_vials_all_v1_t1/
  checkpoints/
    005000/pretrained_model/   adapter_model.safetensors, adapter_config.json, config.json, policy_*processor*
    ...
    030000/pretrained_model/
    last -> 030000
```

### 2.3 Merge a checkpoint

A checkpoint holds only the LoRA adapter and the trained layers. Merge it into a full policy
before running it (about a minute, needs the same Hugging Face access):

```bash
python ~/18842-gr00t/merge_groot_lora.py \
  ${HOME}/vials/outputs/${RUN}/checkpoints/last/pretrained_model \
  ${HOME}/vials/outputs/${RUN}/merged_last
```

For another step replace `last` with e.g. `015000` and use a new output name (`merged_015000`).
The output folder (~9 GB: `config.json`, `model.safetensors`, processor files) has the same layout
as an ACT / SmolVLA `pretrained_model` folder.

### 2.4 Run on the arm

Exactly Step 4 of the class guide, with `unset HF_TOKEN` first (loading GR00T downloads the gated
base model) and the merged folder as the policy:

```bash
conda activate lerobot
unset HF_TOKEN
source ~/Sim-to-Real-SO-101-Workshop/docker/env
export POLICY=${HOME}/vials/outputs/${RUN}/merged_last
lerobot-rollout --strategy.type=base \
  --policy.path=${POLICY} \
  --policy.device=cuda \
  --robot.type=so101_follower \
  --robot.port=${ROBOT_PORT} \
  --robot.id=${ROBOT_ID} \
  --robot.use_degrees=false \
  --robot.calibration_dir=${HOME}/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot.max_relative_target=10 \
  --robot.cameras="{ego: {type: opencv, index_or_path: ${CAMERA_GRIPPER}, width: 640, height: 480, fps: 30}, external_D455: {type: opencv, index_or_path: ${CAMERA_EXTERNAL}, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the vial and place it in the rack" \
  --fps=30 \
  --duration=45 \
  --display_data=true
```

GR00T predicts 40 actions (1.3 s) per inference. If the arm pauses between chunks, add
`--inference.type=rtc --inference.rtc.execution_horizon=10`. Measure success with Step 5 of the
class guide (10 runs per checkpoint) to compare against ACT / Diffusion / SmolVLA.

To copy a policy to another station, copy the whole `merged_last` folder (Step 3, option A of the guide).

---

## How it works

GR00T N1.7 has ~3.4B parameters; NVIDIA's full fine-tuning needs a 40 GB+ GPU. Measured on an
RTX 5080 (16 GB), default GR00T fine-tuning (1.6B trainable) and even training only the action
transformer (1.1B) run out of memory when the optimizer allocates its state. LoRA keeps the big
blocks frozen and trains small low-rank adapters instead:

| Part of GR00T N1.7 | Size | Here |
| --- | --- | --- |
| Vision-language backbone (Cosmos-Reason2-2B) | ~1.8B | frozen |
| Action head: diffusion transformer (32 blocks) + VL self-attention | ~1.3B | frozen + **LoRA** on attention (`to_q/k/v/out`) and feed-forward layers, ~34M |
| Robot state encoder, action encoder, action decoder | ~0.33B | **trained in full** (the SO-101 is a new embodiment for GR00T) |

LeRobot 0.6.1 already supports LoRA (`--peft.*`, `policy.wrap_with_peft`), but for GR00T it
(1) requires `--policy.pretrained_path`, while GR00T loads its base weights via
`--policy.base_model_path`, and (2) has no default target layers. `train_groot_lora.py` patches
those two methods of `GrootPolicy` in memory, then runs LeRobot's normal `lerobot-train` loop.

LeRobot saves LoRA checkpoints as adapter-only folders without a base-model reference, which
`lerobot-rollout` cannot load. `merge_groot_lora.py` rebuilds GR00T from the checkpoint's
`config.json`, applies the adapter, merges it into the weights (`merge_and_unload`) and saves a
standard LeRobot policy folder. A merged policy reproduces the adapter model's actions to within
~0.5 units on the −100…100 joint scale (bf16 rounding).

## Troubleshooting

| Message | Cause → fix |
| --- | --- |
| `403 ... gated repo ... Cosmos-Reason2-2B` | Active account has no Cosmos access (often `HF_TOKEN` from `~/.bashrc`): `unset HF_TOKEN`, check `hf auth whoami`, see 1.3. |
| `Repo id must use alphanumeric chars ... '/vials_all_v1'` | `HF_ORG` is empty in this terminal → run the lines of 2.1 again. |
| `torch.OutOfMemoryError: CUDA out of memory` | `--batch_size=4`; make sure `--policy.model_params_fp32=false` is set; close other GPU programs. |
| `FileExistsError: Output directory ... already exists` | A run with this name exists → new name: `export RUN=${RUN}_b`. |
| `Resuming a GR00T LoRA run is not supported` | See "Interrupted run" in 2.2. |
| `... has no adapter_model.safetensors` (merge) | Point the merge script at `checkpoints/<step>/pretrained_model` of a LoRA run. |
| `KeyError: 'observation.images.ego'` / arm moves to wrong positions | Same as the class guide: camera names `ego` / `external_D455`, `--robot.use_degrees=false`, calibration dir. |

## Tested with

LeRobot 0.6.1 · PyTorch 2.11.0+cu130 · transformers 5.5.4 · peft 0.22.0 · RTX 5080 16 GB, driver 580 ·
dataset `autorobo/vials_all_v1` (621 episodes, 2 cameras, 6-DoF SO-101).
