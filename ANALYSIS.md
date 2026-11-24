# GR00T N1.7 on the SO-101 vial-to-rack task

**Summary:** The two biggest improvements were switching to relative actions in training and adding a
Kalman filter at rollout. GR00T still underperforms, because its pretraining actions, tasks and robots
are far from ours, we could only fine-tune it with LoRA, and its raw output is rough.

---

## Part 1 — What we did to improve performance

### 1.1 Training: run 1 → run 2

| | Run 1 | Run 2 (current) | Why |
|---|---|---|---|
| Actions | absolute joint positions | **relative arm joints, absolute gripper** | GR00T was pretrained on relative arm actions with absolute grippers |
| LoRA rank / alpha | 32 / 64 | **64 / 128** | more adapter capacity |
| Steps | 30k (≈ 0.6 epoch) | **100k (≈ 2.1 epochs)** | run 1 had seen only part of the dataset |
| Unchanged | batch 8, lr 1e-4, chunk 40, bf16 weights, LoRA on the action transformer | | |

Result on recorded frames: arm-joint errors dropped from about 15 units (run 1, lift/elbow) to about
1–3 units. Relative actions require RTC inference (`--inference.type=rtc`) at rollout.

### 1.2 Rollout settings

| Change | Effect |
|---|---|
| RTC inference (`--inference.type=rtc`) | required for relative actions; inference runs in the background |
| `--inference.queue_threshold=10` (default 30) | the policy re-plans every ~30 steps instead of every ~10; this **removed the big shaking** |
| `--policy.num_inference_timesteps=8` | fast enough (0.16 s); 16 steps was no smoother and less accurate |
| `--robot.max_relative_target` 5 → 10 | after filtering, fewer clamp warnings and less stair-step motion; a per-joint setting with `gripper: 100` keeps full grip force |

### 1.3 Kalman filter (`rollout_smooth.py`)

A constant-velocity Kalman filter on the five arm-joint targets (q 0.05, r 5), with the gripper left
unfiltered. It changes nothing in the LeRobot environment.

| Arm targets (offline) | Jitter | Lag |
|---|---|---|
| No filter | 2.77 | 0 |
| Kalman q 0.05 / r 4 | **0.77** | ~10 ms |
| Human demos | 0.14–0.24 | |

- About 3.5× less jitter with almost no delay.
- About half the lag of an EMA filter for the same smoothing.
- **On the arm, the motion was much smoother.**

### 1.4 Tried, but didn't help

- **More denoising steps (16):** no smoother, and slower and less accurate.
- **Longer chunk:** capped at 40 by the base model, and it would not remove the within-chunk roughness.
- **Zero initial noise:** smoother offline, but the arm stayed frozen at the start pose, because the
  model always chose "wait".

---

## Part 2 — Why GR00T doesn't perform well

### 2.1 Action space mismatch

- **Pretraining:** mostly relative end-effector pose (xyz + 6D rotation) from humanoids, robots with
  dexterous hands and a Franka arm.
- **Our SO-101 data:** joint positions only, so most of GR00T's pretrained motion knowledge, which is
  in end-effector space, transfers poorly.
- **Run 1:** used absolute actions, which appear nowhere in pretraining.

### 2.2 The task is far from the pretraining data

| Needed for vial-to-rack | In pretraining? |
|---|---|
| Single-arm pick-and-place with a parallel gripper | Yes: DROID, about 10% of the mixture |
| Thin cylindrical objects | Partly: syringe and bottle tasks, but with dexterous hands, under 5% |
| Precise insertion into a rack slot | No |
| Lab objects, small low-cost arm | No |

About 85% of the pretraining mixture is bimanual or humanoid household and dexterous work. The
hardest step, inserting the vial into the rack, has to be learned from our 621 demos alone.

### 2.3 Only LoRA, and no pretrained layers for our robot

- **Memory limit:** full fine-tuning runs out of memory on the 16 GB GPU.
- **Frozen parts:** the vision-language backbone (about 1.8B parameters) is fully frozen, and the
  action transformer (about 1.3B) is frozen apart from LoRA adapters.
- **Trained from scratch:** the SO-101 uses embodiment slot ID 10, which was never used in pretraining.
  Its state encoder, action encoder and action decoder (about 0.33B parameters) started from random
  weights and were trained for only about 2 epochs.

### 2.4 Rough output inside each chunk

Offline, arm targets (lift / elbow / wrist flex):

| | Within-chunk jitter | Seed-to-seed spread | Chunk-switch jump |
|---|---|---|---|
| Diffusion (t1) | 1.1 / 1.2 / 0.8 | 6–7 | 9–14 |
| GR00T run 2 | **2.5 / 3.0 / 2.8** | 4–10 | 8–13 |

- **Not unusual:** random sampling noise and jumps at chunk switches. Diffusion has them at a similar
  level and still works.
- **The difference:** GR00T is 2–3× rougher inside each chunk. Likely reasons:
  - Diffusion's 1-D convolutional U-Net smooths the output along time.
  - GR00T predicts each step as a separate transformer token.
  - GR00T samples with only 4–8 flow-matching steps.
  - GR00T's decoder for the SO-101 was trained from scratch (2.3).
- **No blending at chunk switches:** LeRobot disables RTC's chunk blending for relative-action GR00T,
  so each new chunk simply replaces the previous one.

### 2.5 Orange arm vs. black arm

On the arm, rollouts on the orange SO-101 were clearly better than on the black one.

- **Data imbalance (hypothesis):** most demos were probably recorded on orange arms.
- **Low contrast:** the black arm blends into the black background, which makes it harder to locate
  the arm and gripper in the image.
- **Pretraining doesn't help:** the vision backbone is frozen and has never seen either arm.

To check this, count the dataset episodes per arm and compare success rates over 10 runs on each.
Possible fixes:
- record more demos on the black arm;
- use a contrasting background;
- add colour augmentation during training.

---

## Conclusion

Relative actions and the Kalman filter fixed most of the motion quality. The remaining gap comes from:
- an action space and tasks that are far from GR00T's pretraining;
- fine-tuning limited to LoRA;
- input and output layers for our robot learned from scratch;
- visual differences between arms.

The next steps are more varied demos, better arm–background contrast, and more training of the action
head.

*The Diffusion and seed-spread numbers come from offline replay of recorded episodes. The Kalman
improvement and the orange-vs-black difference are what we saw on the arm, without measured success
rates yet.*
