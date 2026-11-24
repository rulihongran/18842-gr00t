# 18842-gr00t

LoRA fine-tuning of NVIDIA **GR00T N1.7** (3B vision-language-action model) on the SO-101
**Vial-to-Rack** dataset, on a single 16 GB GPU (RTX 5080), using LeRobot 0.6.1.

It follows the same workflow as the class guide
[Training and Deploying a Policy on the Vial-to-Rack Dataset](https://github.com/z4ziad/autonomous_robotics/blob/main/learning_policies_lab/Training%20and%20Deploying%20a%20Policy%20on%20the%20Vial-to-Rack%20Dataset.md)
(ACT / Diffusion / SmolVLA): same conda environment, same dataset, same `lerobot-rollout` options
on the arm. The differences are the training script, one extra **merge** step, and (for the
recommended relative-action model) RTC inference plus an optional smoothing wrapper.

| File | What it does |
| --- | --- |
| `train_groot_lora.py` | Drop-in for `lerobot-train`: same options, adds LoRA for GR00T at run time |
| `merge_groot_lora.py` | Folds the LoRA adapter into the model → a normal policy folder for `lerobot-rollout` |
| `verify_merge.py` | Offline check of a merged policy against the adapter model and the recorded actions |
| `rollout_smooth.py` | Drop-in for `lerobot-rollout`: same options, plus a Kalman / EMA filter on the arm targets |

No file of the installed `lerobot` package is modified; the scripts only adjust LeRobot's
behaviour in memory (see [How it works](#how-it-works)).

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
Then log in on the computer (training **and** every station that runs the policy):

```bash
hf auth login          # paste a READ token of that account
```

> **`HF_TOKEN` overrides `hf auth login`.** If your `~/.bashrc` exports `HF_TOKEN` for a
> different account, run `unset HF_TOKEN` in **every new terminal** (shown in every command below),
> otherwise loading GR00T fails with `403 ... gated repo ... Cosmos-Reason2-2B`.

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
export RUN=groot_lora_rel_r64_100k_${DATASET}_${TEAM}

hf auth whoami                           # must show the account with Cosmos access
echo ${HF_ORG}/${DATASET}  ${RUN}        # must print: autorobo/vials_all_v1  groot_lora_rel_r64_100k_vials_all_v1_t1
nvidia-smi                               # no other training (python process) on the GPU
```

If `echo` prints `/vials_all_v1`, an `export` line was skipped — run them again.

### 2.2 Train (recommended: relative actions)

```bash
python ~/18842-gr00t/train_groot_lora.py \
  --dataset.repo_id=${HF_ORG}/${DATASET} \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.model_params_fp32=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --peft.r=64 \
  --peft.lora_alpha=128 \
  --output_dir=${HOME}/vials/outputs/${RUN} \
  --job_name=${RUN} \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=10000 \
  --log_freq=200 \
  2>&1 | tee ${HOME}/vials/outputs/${RUN}.log
```

The first run downloads the base model (`nvidia/GR00T-N1.7-3B` + Cosmos, ~10 GB) and the dataset.
Run it inside `tmux` so it survives a closed terminal. `tee` keeps the printed progress (loss) in
`${RUN}.log`; without it the loss is lost when the terminal closes. Or add
`--wandb.enable=true --wandb.project=vials` for training curves (after `wandb login`).

Before training starts, LeRobot computes relative-action statistics over the whole dataset
(about **6–7 minutes** with no output after `Wrapped groot with PEFT` — this is normal).

| Option | What it does |
| --- | --- |
| `--policy.type=groot` | GR00T N1.7, initialised from `nvidia/GR00T-N1.7-3B`. |
| `--policy.model_params_fp32=false` | Keeps the frozen 3.4B weights in bf16. In fp32 they alone take ~14 GB, so this is **required on 16 GB**. |
| `--policy.use_relative_actions=true` | Arm actions are learned as offsets from the current joint state (NVIDIA's SO-100 recipe). The policy still outputs absolute positions to the robot. **Requires RTC inference on the arm** (2.4). |
| `--policy.relative_exclude_joints='["gripper"]'` | The gripper stays an absolute open/close position. |
| `--peft.r=64 --peft.lora_alpha=128` | LoRA rank and scale (scale = alpha / r = 2). Higher rank = more capacity and memory. |
| `--batch_size=8` | ~15.0–15.7 GB of 16.3 GB on an RTX 5080 with r = 64. Use `4` on "CUDA out of memory". |
| `--steps=100000` | ≈ 2.1 epochs of the 387k frames; ≈ 6.5 h at batch 8 (~40 min per 10k steps). |
| `--save_freq=10000` | A checkpoint every 10k steps (each ~0.8 GB). |
| `--policy.optimizer_lr` | Not set → default **1e-4** peak (AdamW, weight decay 1e-5, grad-clip 1.0), 500 warm-up steps, then cosine decay. |

**Absolute-action variant** (the first run we made, simpler to deploy but more vibration on the arm):
drop the two `relative` options and use `--peft.r=32 --peft.lora_alpha=64 --steps=30000 --save_freq=5000`
(≈ 2 h, ~14.2 GB, ≈ 0.6 epoch).

Do not set `--policy.chunk_size` above 40: GR00T N1.7's action horizon is fixed at 40 actions
(1.33 s at 30 fps) and larger values are capped.

**Reading the progress line:** `loss` starts around 1.3 and should fall steadily (it is a
flow-matching loss; not comparable to ACT's). `lr` is tiny at first and reaches 1e-4 at step 500.

**Interrupted run:** `--resume=true` is **not supported** for LoRA runs (the script stops with a
message). Merge and test the last saved checkpoint, or start again with a new name
(`export RUN=${RUN}_b`).

Output:

```
~/vials/outputs/groot_lora_rel_r64_100k_vials_all_v1_t1/
  checkpoints/
    010000/pretrained_model/   adapter_model.safetensors, adapter_config.json, config.json, policy_*processor*
    ...
    100000/pretrained_model/
    last -> 100000
```

### 2.3 Merge a checkpoint

A checkpoint holds only the LoRA adapter and the trained layers. Merge it into a full policy
before running it (about a minute, needs the same Hugging Face access):

```bash
python ~/18842-gr00t/merge_groot_lora.py \
  ${HOME}/vials/outputs/${RUN}/checkpoints/last/pretrained_model \
  ${HOME}/vials/outputs/${RUN}/merged_last
```

For another step replace `last` with e.g. `050000` and use a new output name (`merged_050000`).
The output folder (~8.7 GB: `config.json`, `model.safetensors`, processor files) has the same layout
as an ACT / SmolVLA `pretrained_model` folder.

Optional offline check (GPU must be free, ~2–3 min):

```bash
python ~/18842-gr00t/verify_merge.py \
  ${HOME}/vials/outputs/${RUN}/checkpoints/last/pretrained_model \
  ${HOME}/vials/outputs/${RUN}/merged_last
```

It prints 3 lines like
`frame 150: ... max|merged-lora|=0.655  first action merged=[ 0.5 -5.8  6.6 51.5 -54.9 26.7]  gt=[ 0.3 -9.  9.5 50.9 -56.8 33.5]`.
`merged` must be on the same scale as `gt` (tens, not ≈ 0 — otherwise relative actions are not
converted back), and `max|merged-lora|` should be below ~1 (bf16 rounding).

### 2.4 Run on the arm

Same as Step 4 of the class guide, with `unset HF_TOKEN` first (loading GR00T downloads the gated
base model) and the merged folder as the policy. **Relative-action models must use RTC
inference** (`--inference.type=rtc`); with the default inference they stop with
`GrootPolicy.select_action does not support relative-action policies`.

Recommended command (relative-action model, light Kalman filter):

```bash
conda activate lerobot
unset HF_TOKEN
source ~/Sim-to-Real-SO-101-Workshop/docker/env
export POLICY=${HOME}/vials/outputs/${RUN}/merged_last
ls ${POLICY}/model.safetensors           # must exist (an empty POLICY gives "unrecognized arguments: --policy.path=")

python ~/18842-gr00t/rollout_smooth.py \
  --zero_noise=false \
  --filter=kalman \
  --kalman_q=0.05 \
  --kalman_r=5 \
  --gripper_filter=none \
  --strategy.type=base \
  --policy.path=${POLICY} \
  --policy.device=cuda \
  --policy.num_inference_timesteps=8 \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.queue_threshold=10 \
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

Without the filter, run the same options with `lerobot-rollout` instead of
`python ~/18842-gr00t/rollout_smooth.py` and drop the five `--zero_noise/--filter/--kalman_*/--gripper_filter` lines.

| Option | Why |
| --- | --- |
| `--inference.type=rtc` | Required for relative actions: each 40-action chunk is converted to absolute positions with the joint state at inference time. Runs inference in a background thread. |
| `--inference.queue_threshold=10` | Start the next inference when 10 actions are left → a new chunk every ~30 actions (1 s). The default (30) re-plans every ~10 actions and made the arm **shake strongly**. Keep it above inference time × fps (0.16 s × 30 ≈ 5). |
| `--policy.num_inference_timesteps=8` | Flow-matching denoising steps (default 4). 16 was not smoother and is slower (0.24 s). |
| `--robot.max_relative_target=10` | Max joint change per step (safety clamp). Per joint: `"{shoulder_pan: 8.0, shoulder_lift: 8.0, elbow_flex: 8.0, wrist_flex: 8.0, wrist_roll: 8.0, gripper: 100.0}"` (all six names; gripper 100 keeps full grip force). |
| `--filter=kalman --kalman_q --kalman_r` | See [Smoothing](#25-smoothing-with-rollout_smoothpy). |

An **absolute-action** model runs with plain `lerobot-rollout` and the class-guide options
(no RTC needed); `--inference.type=rtc --inference.rtc.execution_horizon=10` is optional there.

Measure success with Step 5 of the class guide (10 runs per setting) to compare against
ACT / Diffusion / SmolVLA. Keep a hand near the emergency stop on the first run of a new setting,
and check the gripper camera cable (strong shaking once pulled it loose).

### 2.5 Smoothing with `rollout_smooth.py`

GR00T samples each action chunk from random noise, so consecutive chunks differ and the arm
targets are much rougher than the demonstrations. For relative-action models LeRobot 0.6.1 turns off
RTC's chunk blending, so each new chunk replaces the old one. `rollout_smooth.py` accepts all
`lerobot-rollout` options plus:

| Option | Default | Meaning |
| --- | --- | --- |
| `--filter` | `kalman` | `kalman`, `ema` or `none`; applied to the 5 arm joint targets before they are sent |
| `--kalman_q` / `--kalman_r` | `0.01` / `9` | Constant-velocity Kalman filter. Only q / r matters: **larger q / r = lighter filter** (closer to the raw policy) |
| `--ema_alpha` | `0.3` | For `--filter=ema`; larger = lighter |
| `--gripper_filter` | `none` | `same` also filters the gripper (slower grasps) |
| `--zero_noise` | `true` | Start GR00T's sampler from zeros instead of random noise. **Set `false`**: with zero noise the arm stays at the start pose (see Troubleshooting) |

Offline on recorded episodes (arm joints, mean |2nd difference| per step; demos ≈ 0.14–0.24):

| Setting | Jitter | Largest steps (p99) | Lag |
| --- | --- | --- | --- |
| no filter | 2.77 | 12.6 | 0 |
| Kalman q 0.05, r 4 (light) | 0.77 | 7.8 | ~10 ms |
| Kalman q 0.01, r 9 (default) | 0.45 | 6.4 | ~40 ms |
| Kalman q 0.005, r 16 (strong) | 0.34 | 5.8 | ~75 ms |
| EMA 0.3 | 0.61 | 6.0 | ~65 ms |

The startup log shows the active settings (`rollout_smooth: zero_noise=False, filter=kalman ...`);
the last line shows how much the filter changed each joint on average. The return-to-start motion
at the end is not filtered.

### 2.6 Run on another station

Copy the merged folder (about 8.7 GB) and the code to the other PC:

```bash
cd ~/vials/outputs/${RUN}
scp -r merged_last robotlab@<STATION_IP>:~/groot_rel_100k
```

On that station: install the GR00T extras (1.2), log in to Hugging Face with an account that has
Cosmos access (1.3; the first load downloads ~10 GB), clone this repo (1.1), then run 2.4 with
`export POLICY=${HOME}/groot_rel_100k`. `source ~/Sim-to-Real-SO-101-Workshop/docker/env` gives
that station's robot port, robot id and camera indices; its calibration file must exist in
`~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`. Inference needs ~9 GB of GPU memory.

---

## How it works

GR00T N1.7 has ~3.4B parameters; NVIDIA's full fine-tuning needs a 40 GB+ GPU. Measured on an
RTX 5080 (16 GB), default GR00T fine-tuning (1.6B trainable) and even training only the action
transformer (1.1B) run out of memory when the optimizer allocates its state. LoRA keeps the big
blocks frozen and trains small low-rank adapters instead:

| Part of GR00T N1.7 | Size | Here |
| --- | --- | --- |
| Vision-language backbone (Cosmos-Reason2-2B) | ~1.8B | frozen |
| Action head: diffusion transformer (32 blocks) + VL self-attention | ~1.3B | frozen + **LoRA** on attention (`to_q/k/v/out`) and feed-forward layers |
| Robot state encoder, action encoder, action decoder | ~0.33B | **trained in full** (the SO-101 is a new embodiment for GR00T) |

LeRobot 0.6.1 already supports LoRA (`--peft.*`, `policy.wrap_with_peft`), but for GR00T it
(1) requires `--policy.pretrained_path`, while GR00T loads its base weights via
`--policy.base_model_path`, and (2) has no default target layers. `train_groot_lora.py` patches
those two methods of `GrootPolicy` in memory, then runs LeRobot's normal `lerobot-train` loop.

LeRobot saves LoRA checkpoints as adapter-only folders without a base-model reference, which
`lerobot-rollout` cannot load. `merge_groot_lora.py` rebuilds GR00T from the checkpoint's
`config.json`, applies the adapter, merges it into the weights (`merge_and_unload`) and saves a
standard LeRobot policy folder. A merged policy reproduces the adapter model's actions to within
~0.8 units on the −100…100 joint scale (bf16 rounding). Relative-action settings and statistics
are kept in the saved processor files.

`rollout_smooth.py` wraps `lerobot-rollout`: it removes its own options from the command line,
filters the `<joint>.pos` targets inside `ThreadSafeRobot.send_action` (the robot's
`max_relative_target` clamp still applies afterwards), optionally makes GR00T's initial
flow-matching noise zero during sampling, and skips the filter for the return-to-start motion.

The language input is the dataset task text ("Pick up the vial and place it in the rack"; GR00T
lowercases it and removes punctuation). It is the only task in the dataset, so use exactly this
`--task` on the arm.

## Troubleshooting

| Message / symptom | Cause → fix |
| --- | --- |
| `403 ... gated repo ... Cosmos-Reason2-2B` | Active account has no Cosmos access (often `HF_TOKEN` from `~/.bashrc`): `unset HF_TOKEN` in this terminal, check `hf auth whoami`, see 1.3. |
| `Repo id must use alphanumeric chars ... '/vials_all_v1'` | `HF_ORG` is empty in this terminal → run the lines of 2.1 again. |
| No output for minutes after `Wrapped groot with PEFT` | Relative-action statistics are being computed (~6–7 min). Wait. |
| `torch.OutOfMemoryError: CUDA out of memory` | `--batch_size=4` or `--peft.r=32 --peft.lora_alpha=64`; make sure `--policy.model_params_fp32=false` is set; close other GPU programs. |
| `FileExistsError: Output directory ... already exists` | A run with this name exists → new name: `export RUN=${RUN}_b`. |
| `Resuming a GR00T LoRA run is not supported` | See "Interrupted run" in 2.2. |
| `... has no adapter_model.safetensors` (merge) | Point the merge script at `checkpoints/<step>/pretrained_model` of a LoRA run. |
| `unrecognized arguments: --policy.path=` | `POLICY` is empty in this terminal → `export POLICY=<full path>`. |
| `GrootPolicy.select_action does not support relative-action policies` | Relative-action model → add `--inference.type=rtc` (2.4). |
| Arm shakes strongly with RTC | Use `--inference.queue_threshold=10` (2.4) and the Kalman filter (2.5). |
| Arm barely moves with `rollout_smooth.py` | `--zero_noise=true` (the default) makes the start pose a fixed point → `--zero_noise=false`. |
| `Relative goal position magnitude had to be clamped to be safe` | A target was more than `max_relative_target` away from the present position in one step (mostly when a new chunk starts). Occasional warnings are fine; the filter reduces them. On the gripper the clamp weakens the grip → per-joint clamp with `gripper: 100.0`. Continuous warnings on one joint = the arm is blocked. |
| `Indexes diff is not equal to real delay` | Harmless RTC timing note (off by one step). |
| `OpenCVCamera(N) read failed` / `No such device` | A USB camera disconnected (cable pulled by arm motion). Reconnect, secure the cable, rerun. |
| `KeyError: 'observation.images.ego'` / arm moves to wrong positions | Same as the class guide: camera names `ego` / `external_D455`, `--robot.use_degrees=false`, calibration dir. |

## Tested with

LeRobot 0.6.1 · PyTorch 2.11.0+cu130 · transformers 5.5.4 · peft 0.22.0 · RTX 5080 16 GB, driver 580 ·
dataset `autorobo/vials_all_v1` (621 episodes, 2 cameras, 6-DoF SO-101).
