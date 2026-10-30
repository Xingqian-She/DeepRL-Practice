"""保存推理权重及配套 RND 统计量；不包含完整续训状态。"""

import numpy as np
import torch


def save_checkpoint(model_path, agent, rnd_model, obs_rms, int_rew_rms, int_reward_filter):
    """文件字段与原实现兼容，归一化统计量必须随权重一起保存。"""
    torch.save(
        {
            "agent": agent.state_dict(),
            "model_config": agent.model_config,
            "rnd_model": rnd_model.state_dict(),
            # 恢复 RND 奖励计算时需要与权重配套的归一化状态。
            "obs_rms": {
                "mean": torch.as_tensor(obs_rms.mean.copy()),
                "var": torch.as_tensor(obs_rms.var.copy()),
                "count": float(obs_rms.count),
            },
            "int_rew_rms": {
                "mean": torch.as_tensor(np.array(int_rew_rms.mean)),
                "var": torch.as_tensor(np.array(int_rew_rms.var)),
                "count": float(int_rew_rms.count),
            },
            "int_reward_filter": torch.as_tensor(int_reward_filter.discounted_rewards.copy()),
        },
        model_path,
    )
