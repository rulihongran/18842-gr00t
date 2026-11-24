# GR00T N1.7 LoRA on SO-101 vial-to-rack — project notes for Claude

## Rules (from the user — always follow)
- Only create/modify files inside `~/18842-gr00t`. Never edit the `lerobot` conda env (no pip install/upgrade,
  no site-packages edits), `~/.bashrc`, or other code. Patch LeRobot behaviour at runtime from scripts here.
- Do not change anything without the user's explicit permission; propose a plan first. "Just discuss" = no changes.
- The user runs everything that moves the robot arm, and anything needing a password (scp/ssh/git push).
  Give full copy-paste commands, one command per line (long pasted lines have been split by the terminal before).
- The user sometimes wants answers in Chinese.
- Repo: `github.com/rulihongran/18842-gr00t` (the user pushes).

## Files in this repo
| File | Purpose |
|---|---|
| `train_groot_lora.py` | `lerobot-train` + LoRA for GR00T (patches `GrootPolicy` PEFT methods in memory) |
| `merge_groot_lora.py` | LoRA checkpoint → normal policy folder for `lerobot-rollout` (`merge_and_unload`) |
| `verify_merge.py` | offline check: merged vs LoRA vs recorded actions on 3 frames of episode 0 |
| `rollout_smooth.py` | `lerobot-rollout` + optional zero initial noise + Kalman/EMA filter on arm joint targets (see below) |
| `README.md` | user-facing guide (setup, train, merge, rollout). Section 2.4 does not yet mention RTC/rollout_smooth. |

## Setup facts (station C = training PC)
- RTX 5080 16 GB, conda env `lerobot` (LeRobot 0.6.1 + `lerobot[groot]==0.6.1` extras; torch 2.11+cu130,
  transformers 5.5.4, peft 0.22.0).
- Dataset `autorobo/vials_all_v1`: 621 episodes, 387,069 frames, 30 fps, cameras `ego` + `external_D455`, 6-DoF,
  one task text "Pick up the vial and place it in the rack" (GR00T lowercases it and strips punctuation).
- HF: `~/.bashrc` exports `HF_TOKEN` of account `z4ziad` (no access to gated `nvidia/Cosmos-Reason2-2B`).
  The saved `hf auth login` account `jingxuanz` has access → **every GR00T command needs `unset HF_TOKEN`** in the
  same terminal (forgetting it → `403 ... gated repo ... Cosmos-Reason2-2B` at policy load).
- Full fine-tune OOMs on 16 GB → LoRA + `--policy.model_params_fp32=false`.
- Station env: `source ~/Sim-to-Real-SO-101-Workshop/docker/env` → ROBOT_ID=stC_robot, ROBOT_PORT=/dev/ttyACM0,
  CAMERA_GRIPPER=2 (Sonix USB cam), CAMERA_EXTERNAL=0 (Logitech C922).
- No SSH keys on station C; scp/ssh ask for a password.
- Training loss was NOT saved (printed to terminal only, wandb off, PC rebooted since). Next time add
  `2>&1 | tee ~/vials/outputs/${RUN}.log` or `--wandb.enable=true`.

## Running on another PC station
1. **Copy the model** (from station C; the user types the password). Merged run-2 model = 8.7 GB:
   ```bash
   cd ~/vials/outputs/groot_lora_rel_r64_100k_vials_all_v1_t5
   scp -r merged_last robotlab@<STATION_IP>:~/groot_rel_100k
   ```
   Already done once to `172.24.59.32` → `~/groot_rel_100k` (verify: `du -sh ~/groot_rel_100k` ≈ 8.7G,
   files `config.json model.safetensors policy_*`).
2. **Get the code** on that station: `git clone https://github.com/rulihongran/18842-gr00t.git ~/18842-gr00t`
   (or `scp ~/18842-gr00t/rollout_smooth.py ...`).
3. **GR00T extras** in that station's `lerobot` env: check `python -c "import peft, transformers; print(transformers.__version__)"`.
   If missing, `pip install 'lerobot[groot]==0.6.1'` is needed — ask the user first (shared class env).
4. **HF access**: loading GR00T downloads `nvidia/GR00T-N1.7-3B` + gated `nvidia/Cosmos-Reason2-2B` (~10 GB first
   time). Log in with an account that has Cosmos access (`hf auth login`), then `unset HF_TOKEN`, `hf auth whoami`.
5. **Station values**: `source ~/Sim-to-Real-SO-101-Workshop/docker/env` on that PC gives its own ROBOT_ID,
   ROBOT_PORT, CAMERA_GRIPPER, CAMERA_EXTERNAL. Check calibration exists:
   `ls ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/${ROBOT_ID}.json`.
   Camera names must stay `ego` / `external_D455`; `--robot.use_degrees=false`.
6. `export POLICY=${HOME}/groot_rel_100k` (full path; empty POLICY → `unrecognized arguments: --policy.path=`).
7. Other GPU: inference needs ~9 GB VRAM; check `nvidia-smi`. Inference time was 0.16 s at 8 denoise steps on the
   5080; on a slower GPU keep `queue_threshold` > inference_time × 30 (else the queue runs dry and the arm pauses).

## Training runs (both: batch 8, lr 1e-4 AdamW, wd 1e-5, 500 warm-up + cosine, LoRA dropout 0.1, seed 1000,
## chunk 40 (fixed by base model action_horizon=40), n_action_steps 40, bf16 params;
## LoRA on action-head DiT + VL self-attn, state/action enc/dec trained in full; LLM + vision frozen)

| | Run 1 (absolute) | Run 2 (relative) — **current model** |
|---|---|---|
| Output | `~/vials/outputs/groot_lora_vials_all_v1_t5` | `~/vials/outputs/groot_lora_rel_r64_100k_vials_all_v1_t5` |
| Actions | absolute | relative (offset from state), gripper absolute |
| LoRA r / alpha | 32 / 64 | 64 / 128 |
| Steps / epochs | 30,000 / ~0.62 | 100,000 / ~2.1 |
| save_freq | 5000 | 10000 |
| Trainable params | ~360M | ~393M |
| Time | ~2 h | ~6.5 h (+6.5 min relative-stats prep), ~40 min / 10k steps |
| Merged | `merged_030000` | `merged_last` (= step 100000) |

Run 2 command:
```bash
conda activate lerobot; unset HF_TOKEN
export HF_ORG=autorobo DATASET=vials_all_v1 TEAM=t5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=groot_lora_rel_r64_100k_${DATASET}_${TEAM}
python ~/18842-gr00t/train_groot_lora.py \
  --dataset.repo_id=${HF_ORG}/${DATASET} --policy.type=groot --policy.device=cuda \
  --policy.push_to_hub=false --policy.model_params_fp32=false \
  --policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]' \
  --peft.r=64 --peft.lora_alpha=128 \
  --output_dir=${HOME}/vials/outputs/${RUN} --job_name=${RUN} \
  --steps=100000 --batch_size=8 --save_freq=10000 --log_freq=200
```
Run 1 = same without the two `relative` options, `--peft.r=32 --peft.lora_alpha=64 --steps=30000 --save_freq=5000`,
RUN=groot_lora_vials_all_v1_t5. GPU memory during run 2: ~15.0–15.7 of 16.3 GB (tight; nothing else on the GPU).

Merge + offline check:
```bash
CK=${HOME}/vials/outputs/${RUN}/checkpoints/last/pretrained_model; MERGED=${HOME}/vials/outputs/${RUN}/merged_last
python ~/18842-gr00t/merge_groot_lora.py $CK $MERGED
python ~/18842-gr00t/verify_merge.py $CK $MERGED
```

## Current best rollout command (run 2 + light Kalman) — set by the user
```bash
conda activate lerobot
unset HF_TOKEN
source ~/Sim-to-Real-SO-101-Workshop/docker/env
export POLICY=${HOME}/vials/outputs/groot_lora_rel_r64_100k_vials_all_v1_t5/merged_last   # other station: ${HOME}/groot_rel_100k
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
  --duration=15 \
  --display_data=true
```
- **`--zero_noise=false` is required** (script default is `true`, which freezes the arm at the start; see below).
- Use `--duration=45` for full task attempts / success-rate runs (10 runs per setting).
- Per-joint clamp alternative: `--robot.max_relative_target="{shoulder_pan: 8.0, shoulder_lift: 8.0, elbow_flex: 8.0, wrist_flex: 8.0, wrist_roll: 8.0, gripper: 100.0}"`
  (all 6 names required; gripper 100 = no clamp → full grip force).
- Without filter (= test D): `--filter=none --zero_noise=false` (or plain `lerobot-rollout` without those two flags).

## rollout_smooth.py options
| Option | Default | Meaning |
|---|---|---|
| `--zero_noise` | `true` (!) | GR00T flow-matching starts from zeros → deterministic. **Freezes the arm at the start pose** on the robot; use `false` |
| `--filter` | `kalman` | `kalman` / `ema` / `none`, applied to the 5 arm `*.pos` targets in `ThreadSafeRobot.send_action` |
| `--kalman_q` / `--kalman_r` | 0.01 / 9 | constant-velocity Kalman; only q/r matters: bigger q/r = lighter filter, closer to raw |
| `--ema_alpha` | 0.3 | for `--filter=ema`; bigger = lighter |
| `--gripper_filter` | `none` | `same` filters the gripper too |
End-of-run return-to-start bypasses the filter. Last log line prints mean |filtered − raw| per joint.
Startup log prints the active settings (`rollout_smooth: zero_noise=..., filter=...`).

## Rollout tests on the arm (station C)
| # | Policy | Options | Result |
|---|---|---|---|
| A | Run 1 `merged_030000` | sync, 4 denoise steps, `max_relative_target=10` | moves; constant fast vibration; low success |
| B | Run 2 | sync | crash: `GrootPolicy.select_action does not support relative-action policies` → needs RTC |
| C | Run 2 | RTC, 16 steps, default `queue_threshold=30`, clamp 10 | **big shaking**; gripper cam USB dropped → crash |
| D | Run 2 | RTC, 8 steps, `queue_threshold=10`, clamp 5 | **most successful without filter**: works sometimes, still jitter; many clamp warnings |
| K0 | Run 2 | D + `rollout_smooth.py` zero_noise=true, kalman q0.01 r9 | **arm barely moved** (zero noise → stuck at start pose) |
| K1 | Run 2 | current best command above (zero_noise=false, kalman q0.05 r5, clamp 10) | result: fill in |

## Analysis (offline, 6 episodes, simulated RTC stream: 30 steps per chunk, 5-step delay, 8 denoise steps)
Units: joint scale −100…100. "jitter" = mean |2nd difference| per step.
- Demo jitter 0.11–0.24; run 2 policy stream 1.0–3.0 on arm joints (10–20× rougher); gripper smooth (~0.4).
  Run 2 did NOT reduce within-chunk jitter vs run 1 (~2.0); test D improved mainly via fewer chunk switches.
- **Sampling noise is the main cause**: same frame, different seeds → std 8–11 on lift / wrist flex / wrist roll.
  Each chunk switch re-samples → switch jump mean 9–13 (lift/elbow), p90 23–29 → `clamped` warnings, jerks.
- More denoise steps don't help: 4/8/16 → jitter ≈ same, error 8.2/9.5/10.3, 127/160/237 ms.
- **RTC is only async here**: LeRobot disables GR00T's RTC prefix blending for relative-action policies
  (`lerobot/policies/groot/modeling_groot.py:356`, log "Disabling native GR00T RTC prefix for relative-action
  policy"), so each new chunk replaces the old one; `execution_horizon` has no effect. Async inference, latency
  skip and `queue_threshold` do work. (`Indexes diff is not equal to real delay` warnings are harmless.)
- Filters on the stream (arm avg; random noise): raw 2.77 / p99 step 12.6; EMA 0.3 → 0.61, lag 2 ticks;
  Kalman q0.05 r4 → 0.77, lag 0.3, p99 step 7.8; q0.01 r9 → 0.45, lag 1.3, p99 6.4; q0.005 r16 → 0.34, lag 2.3.
  Errors vs demo unchanged (~6.5–6.9). Kalman ≈ half the lag of EMA for equal smoothing, slight overshoot at jumps.
- Zero noise offline: jitter 2.77 → 0.95 and lower error, BUT at the start pose it commands ~2–5 units of motion
  vs 30–85 with random noise (demos sometimes wait, sometimes go; zero noise picks "wait" every time) → stuck.
- `Relative goal position magnitude had to be clamped` = target > clamp away from present position in one step
  (mostly chunk-switch jumps; continuous on one joint = arm blocked/lagging). Clamp on the gripper weakens grip.
- Run 2 gripper less accurate than arm joints (verify_merge: up to 11). Gripper cam cable can come loose; check it.
- chunk_size > 40 is not possible without changing the model (capped to base `action_horizon` 40) and would not fix
  within-chunk noise.

## Open ideas (not done; need user permission)
- Chunk-switch crossfade in `rollout_smooth.py` (blend old → new chunk over 5–10 steps) — targets the main jump source.
- `--noise_scale` (e.g. 0.5) instead of all-or-nothing zero noise; test offline at start poses first.
- Change `--zero_noise` default to `false`.
- Recompute a loss curve from checkpoints (run 1: 5k–30k, run 2: 10k–100k) on fixed frames (~30–40 min GPU).
- `eval_offline.py`: per-joint error / jitter / chunk-jump vs demos for any checkpoint.
- Update README section 2.4: relative-action models need `--inference.type=rtc`; document `rollout_smooth.py`.
- Offline analysis scripts live only in the session scratchpad (`jitter_analysis*.py`, `filter_compare.py`,
  `start_motion.py`); not in the repo.
