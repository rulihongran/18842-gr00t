"""Check a merged GR00T LoRA policy loads via LeRobot's standard path and matches the adapter model."""

import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy

lora_dir, merged_dir = Path(sys.argv[1]), Path(sys.argv[2])
ds = LeRobotDataset("autorobo/vials_all_v1", episodes=[0])
FRAMES = [0, 150, 300]


def batches():
    for i in FRAMES:
        item = ds[i]
        b = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
        yield i, b


def run(policy, pre, post):
    outs = []
    for i, b in batches():
        torch.manual_seed(0)
        x = pre(b)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(x)
        outs.append(post(chunk).float().cpu())
    return outs


# 1) merged policy, loaded the standard LeRobot way (as lerobot-rollout does)
cfg = PreTrainedConfig.from_pretrained(merged_dir)
cfg.pretrained_path = merged_dir
policy = make_policy(cfg, ds_meta=ds.meta)
pre, post = make_pre_post_processors(cfg, pretrained_path=str(merged_dir))
print("merged loaded via make_policy:", type(policy).__name__, "use_peft =", policy.config.use_peft)
merged_out = run(policy, pre, post)
del policy
gc.collect()
torch.cuda.empty_cache()

# 2) unmerged: base + adapter
cfg2 = PreTrainedConfig.from_pretrained(lora_dir)
cfg2.use_peft, cfg2.pretrained_path = False, None
peft_policy = PeftModel.from_pretrained(GrootPolicy(cfg2), str(lora_dir)).to("cuda").eval()
peft_out = run(peft_policy, pre, post)

for i, a, b in zip(FRAMES, merged_out, peft_out):
    gt = ds[i]["action"]
    print(
        f"frame {i}: chunk {tuple(a.shape)}  max|merged-lora|={float((a - b).abs().max()):.3f}  "
        f"first action merged={a[0, 0].numpy().round(1)}  gt={gt.numpy().round(1)}"
    )
