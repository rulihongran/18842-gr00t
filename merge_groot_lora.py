#!/usr/bin/env python
"""Merge a GR00T LoRA checkpoint into a normal LeRobot policy folder.

    python ~/18842-gr00t/merge_groot_lora.py <lora pretrained_model folder> <output folder>

Example:
    python ~/18842-gr00t/merge_groot_lora.py \
        ~/vials/outputs/groot_lora_vials_all_v1_t1/checkpoints/last/pretrained_model \
        ~/vials/outputs/groot_lora_vials_all_v1_t1/merged_last

The output folder holds config.json + model.safetensors + the pre/post-processor files,
the same layout as any other LeRobot checkpoint, so it can be used directly with
`lerobot-rollout --policy.path=<output folder>`.

Loading GR00T (here and in lerobot-rollout) downloads the base model, so the Hugging Face
account in use needs access to nvidia/Cosmos-Reason2-2B.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora_dir", type=Path, help="checkpoints/<step>/pretrained_model folder of a LoRA run")
    ap.add_argument("out_dir", type=Path, help="where to write the merged policy (must not exist)")
    args = ap.parse_args()

    lora_dir = args.lora_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not (lora_dir / "adapter_model.safetensors").exists():
        raise SystemExit(f"{lora_dir} has no adapter_model.safetensors - is it a LoRA checkpoint?")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; choose a new folder.")

    # Rebuild the base policy exactly as it was configured for training.
    config = PreTrainedConfig.from_pretrained(lora_dir)
    config.use_peft = False
    config.pretrained_path = None
    print(f"Loading base GR00T model ({config.base_model_path}) ...")
    policy = GrootPolicy(config)

    print("Applying LoRA adapter and merging ...")
    peft_model = PeftModel.from_pretrained(policy, str(lora_dir))
    merged = peft_model.merge_and_unload()
    merged.eval()

    n_lora = sum("lora_" in n for n, _ in merged.named_parameters())
    if n_lora:
        raise SystemExit(f"Merge failed: {n_lora} LoRA parameters remain.")

    out_dir.mkdir(parents=True)
    merged.config.use_peft = False
    merged.save_pretrained(out_dir)  # config.json + model.safetensors

    # Processor files (normalisation stats, input packing) and the training config.
    for f in lora_dir.iterdir():
        if f.name.startswith("policy_") or f.name == "train_config.json":
            shutil.copy2(f, out_dir / f.name)

    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg.get("use_peft") is False, cfg.get("use_peft")
    size_gb = (out_dir / "model.safetensors").stat().st_size / 1e9
    print(f"Done: {out_dir}  (model.safetensors {size_gb:.1f} GB)")
    print(f"Run on the arm with:  lerobot-rollout ... --policy.path={out_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
