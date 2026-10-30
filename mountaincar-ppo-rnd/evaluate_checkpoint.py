"""对保存的模型做独立评估；默认使用训练过程未用过的评估种子。"""

import argparse
import json
from pathlib import Path

import torch

from diagnostics import evaluate
from model import PPOActorCritic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # 兼容诊断早期生成的文件及原始 checkpoint。
        config_path = args.checkpoint.parent.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        model_config = dict(n_observations=2, n_actions=3,
                            architecture=config.get("architecture", "residual"),
                            normalize_input=config.get("normalize_policy_input", False))
    agent = PPOActorCritic(**model_config).to(args.device)
    agent.load_state_dict(checkpoint["agent"])
    results = {mode: evaluate(agent, torch.device(args.device), args.seed, args.episodes,
                              deterministic=mode == "deterministic")
               for mode in ["deterministic", "stochastic"]}
    results["checkpoint"] = str(args.checkpoint)
    output = args.checkpoint.parent.parent / f"evaluation-seed{args.seed}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
