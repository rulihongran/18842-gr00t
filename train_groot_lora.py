#!/usr/bin/env python
"""LoRA fine-tuning of GR00T N1.7 with LeRobot's own training loop.

Run it exactly like `lerobot-train` (same options), e.g.:

    python ~/18842-gr00t/train_groot_lora.py --dataset.repo_id=autorobo/vials_all_v1 \
        --policy.type=groot ... [--peft.r=32 --peft.lora_alpha=64]

Nothing in the installed lerobot package is edited. LeRobot 0.6.1 already has a LoRA
path (`--peft.*`, `policy.wrap_with_peft`), but two things block it for GR00T, so this
script adjusts them *in memory only*, for this process:

1. LeRobot refuses LoRA unless `--policy.pretrained_path` is set, but GR00T loads its
   pretrained weights through `--policy.base_model_path` (default nvidia/GR00T-N1.7-3B),
   so the check is skipped here.
2. GR00T has no default LoRA target layers, so the ones below are supplied:
   - LoRA adapters on the attention + feed-forward layers of the action head's
     diffusion transformer (DiT, ~1.1B params) and its VL self-attention block;
   - the small per-robot input/output layers (state encoder, action encoder,
     action decoder) are trained in full, because our SO-101 uses the
     "new_embodiment" slot, which needs real adaptation rather than a low-rank patch.
   The vision-language backbone (Cosmos-Reason2-2B) stays frozen.

Checkpoints contain only the adapter + those layers (adapter_model.safetensors).
Run `merge_groot_lora.py` on a checkpoint to get a normal policy folder that
`lerobot-rollout --policy.path=...` loads unchanged.
"""

import sys

from lerobot.policies.groot.modeling_groot import GrootPolicy

# Linear layers inside the action head's transformer blocks (diffusers naming).
LORA_TARGET_REGEX = (
    r".*action_head\.(model|vl_self_attention)\.transformer_blocks\.\d+\."
    r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)
# Per-embodiment projection layers, trained in full (PEFT "modules_to_save").
FULL_TRAINING_MODULES = ["state_encoder", "action_encoder", "action_decoder"]


def _default_peft_targets(self):
    return {"target_modules": LORA_TARGET_REGEX, "modules_to_save": FULL_TRAINING_MODULES}


def _validate_peft_config(self, peft_config):
    # Base weights come from config.base_model_path, not pretrained_path.
    if not self.config.base_model_path:
        raise ValueError("GR00T LoRA needs --policy.base_model_path (default nvidia/GR00T-N1.7-3B).")


GrootPolicy._get_default_peft_targets = _default_peft_targets
GrootPolicy._validate_peft_config = _validate_peft_config

# LeRobot only enables PEFT when some --peft.* option is given; default to LoRA.
if not any(a.startswith("--peft.") for a in sys.argv[1:]):
    sys.argv.append("--peft.method_type=LORA")

if "--resume=true" in sys.argv:
    sys.exit(
        "Resuming a GR00T LoRA run is not supported by this script "
        "(LeRobot cannot rebuild the base model from an adapter-only checkpoint). "
        "Start a new run, or lower --save_freq so less is lost."
    )

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
