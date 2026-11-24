# GR00T N1.7 LoRA on SO-101 vial-to-rack — project notes for Claude

## Rules (from the user — always follow)
- Only create/modify files inside `~/18842-gr00t`. Never edit the `lerobot` conda env (no pip install/upgrade,
  no site-packages edits), `~/.bashrc`, or other code. Patch LeRobot behaviour at runtime from scripts here.
- Do not change anything without the user's explicit permission; propose a plan first.
- The user runs everything that moves the robot arm. Give full copy-paste commands.
- The user sometimes wants answers in Chinese.
- Commits: the user pushes (`github.com/rulihongran/18842-gr00t`).

## Setup facts
- Station C PC, RTX 5080 16 GB, conda env `lerobot` (LeRobot 0.6.1, `lerobot[groot]` extras added; torch 2.11+cu130,
  transformers 5.5.4, peft 0.22.0).
- Dataset `autorobo/vials_all_v1`: 621 episodes, 387,069 frames, 30 fps, cameras `ego` + `external_D455`, 6-DoF.
- HF: `~/.bashrc` exports `HF_TOKEN` of account `z4ziad` (no access to gated `nvidia/Cosmos-Reason2-2B`).
  The saved `hf auth login` account `jingxuanz` has access → **every GR00T command needs `unset HF_TOKEN`**.
- Full fine-tune OOMs on 16 GB → LoRA + `--policy.model_params_fp32=false`.
- Files: `train_groot_lora.py` (lerobot-train + LoRA patches), `merge_groot_lora.py` (LoRA → normal policy folder
  for lerobot-rollout), `verify_merge.py` (offline check: merged vs LoRA vs recorded actions on 3 frames).
- Station env: `source ~/Sim-to-Real-SO-101-Workshop/docker/env` → ROBOT_ID=stC_robot, ROBOT_PORT=/dev/ttyACM0,
  CAMERA_GRIPPER=2 (Sonix USB cam), CAMERA_EXTERNAL=0 (Logitech C922).

## Training runs (both: batch 8, lr 1e-4 AdamW, wd 1e-5, 500 warm-up + cosine, LoRA dropout 0.1, seed 1000,
## chunk 40, n_action_steps 40, bf16 params; LoRA on action-head DiT + VL self-attn, state/action enc/dec trained in full)

| | Run 1 (absolute) | Run 2 (relative) |
|---|---|---|
| Output | `~/vials/outputs/groot_lora_vials_all_v1_t5` | `~/vials/outputs/groot_lora_rel_r64_100k_vials_all_v1_t5` |
| Actions | absolute | relative (offset from state), gripper absolute |
| LoRA r / alpha | 32 / 64 | 64 / 128 |
| Steps / epochs | 30,000 / ~0.62 | 100,000 / ~2.1 |
| save_freq | 5000 | 10000 |
| Trainable params | ~360M | ~393M |
| Time | ~2 h | ~6.5 h (+6.5 min relative-stats prep) |
| Merged | `merged_030000` | `merged_last` (= 100000) |

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

## Offline results
- Run 1 `merged_030000` (56 chunks): jitter (mean 2nd diff) ~2.0 vs ~0.15 in demos; chunk-to-chunk jump on
  lift/elbow 23/31 vs 0.7/0.8; mean error lift 15, elbow 15, wrist flex 10; inference 0.115 s.
  Denoise steps 4/8/16 → jitter 1.79/1.57/1.29, inference 0.115/0.157/0.238 s.
- Run 2 `merged_last` (verify_merge, 3 frames of episode 0): output is absolute (relative→absolute works);
  merged vs LoRA ≤ 0.76; arm-joint errors mostly 0.1–3 (elbow up to 6.6); **gripper error up to 11**.

## Rollout commands tried (all with the common block below)
Common block:
```bash
conda activate lerobot; unset HF_TOKEN; source ~/Sim-to-Real-SO-101-Workshop/docker/env
lerobot-rollout --strategy.type=base --policy.path=${POLICY} --policy.device=cuda \
  --robot.type=so101_follower --robot.port=${ROBOT_PORT} --robot.id=${ROBOT_ID} --robot.use_degrees=false \
  --robot.calibration_dir=${HOME}/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot.cameras="{ego: {type: opencv, index_or_path: ${CAMERA_GRIPPER}, width: 640, height: 480, fps: 30}, external_D455: {type: opencv, index_or_path: ${CAMERA_EXTERNAL}, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the vial and place it in the rack" --fps=30 --display_data=true  <EXTRA OPTIONS>
```
`POLICY` must be the full path (empty → `unrecognized arguments: --policy.path=`).

| # | Policy | Extra options | Result |
|---|---|---|---|
| A | Run 1 `merged_030000` | `--robot.max_relative_target=10 --duration=45` (sync, 4 denoise steps) | arm moves; constant fast vibration; low success |
| A2 | Run 1 `merged_030000` | A + `--policy.num_inference_timesteps=16 --inference.type=rtc --inference.rtc.execution_horizon=10` | result: fill in (suggested, not confirmed tested) |
| B | Run 2 `merged_last` | same as A (sync) | crash: `GrootPolicy.select_action does not support relative-action policies` → relative needs `--inference.type=rtc` |
| C | Run 2 `merged_last` | `--policy.num_inference_timesteps=16 --inference.type=rtc --inference.rtc.execution_horizon=10 --robot.max_relative_target=10 --duration=45` | **big shaking**; gripper camera USB dropped (`/dev/video2 No such device`) → crash |
| D | Run 2 `merged_last` | `--policy.num_inference_timesteps=8 --inference.type=rtc --inference.rtc.execution_horizon=10 --inference.queue_threshold=10 --robot.max_relative_target=5 --duration=15` | **much better**; many gripper clamp warnings (gripper held at ~14.7, model wants ~2 → probably on the vial; clamp weakens grip) |
| E | Run 2 `merged_last` | D but `--robot.max_relative_target="{shoulder_pan: 5.0, shoulder_lift: 5.0, elbow_flex: 5.0, wrist_flex: 5.0, wrist_roll: 5.0, gripper: 100.0}" --duration=45` | result: fill in (success rate /10, grip OK?) |

## Known problems / analysis
- Run 1 vibration: under-trained action head + absolute actions; noisy flow-matching output.
- Run 2 + RTC: LeRobot disables GR00T's RTC prefix blending for relative-action policies
  (`lerobot/policies/groot/modeling_groot.py:356`), so RTC only runs async inference and **replaces** the queue with
  each new chunk. Each chunk is anchored to the measured (lagging) joint state → with default `queue_threshold=30`
  (new chunk every ~10 steps) the arm is pulled back every ~0.3 s = big shaking. `queue_threshold=10` (new chunk
  every ~30 steps) fixed most of it. Hypothesis, partly confirmed by test D.
- `max_relative_target` clamps per step; on the gripper it limits grip force while holding the vial → use the
  per-joint dict (test E).
- Gripper accuracy of run 2 is weaker than the arm joints (gripper stayed absolute).
- Gripper camera cable can be pulled loose by shaking; check before each run.

## Kalman Filter
`
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
    --robot.cameras="{ego: {type: opencv, index_or_path: ${CAMERA_GRIPPER}, width: 640, height: 480, fps: 30}, external_D455: {type: opencv, index_or_path: ${CAMERA_EXTERNAL},
  width: 640, height: 480, fps: 30}}" \
    --task="Pick up the vial and place it in the rack" \
    --fps=30 \
    --duration=15 \
    --display_data=true

`

## Open ideas (not done; need user permission)
- `rollout_smooth.py` here: runtime EMA patch of `ThreadSafeRobot.send_action`
  (`lerobot/rollout/robot_wrapper.py:46`), alpha ~0.3 arm / 0.6 gripper.
- `eval_offline.py` here: per-joint error / jitter / chunk-jump vs demos for any checkpoint (score checkpoints
  010000…100000 before arm time).
- Try other `queue_threshold` (e.g. 5, 15) and denoise steps (4/8/16); compare run 1 with RTC (A2).
- Update README section 2.4: relative-action models require `--inference.type=rtc`.
