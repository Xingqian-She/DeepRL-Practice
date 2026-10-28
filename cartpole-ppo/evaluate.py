from pathlib import Path

import gymnasium as gym
import torch

from model import PPOSeparateActorCritic


ENV_ID = "CartPole-v1"
NUM_EPISODES = 5
MODEL_PATH = Path(__file__).resolve().parent / "runs" / "checkpoints" / "ppo_model.pth"


@torch.inference_mode()
def evaluate():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}，请先运行 train.py 完成训练。")

    # human 模式会自动弹窗，并按环境帧率播放。
    env = gym.make(ENV_ID, render_mode="human")
    try:
        agent = PPOSeparateActorCritic(
            env.observation_space.shape[0], env.action_space.n
        )
        agent.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        agent.eval()

        for episode in range(1, NUM_EPISODES + 1):
            obs, _ = env.reset()
            episode_return = 0.0

            while True:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                # 评估时直接选概率最高的动作，不进行随机采样。
                action = agent.actor(obs_tensor).argmax(dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward

                if terminated or truncated:
                    break

            print(f"第 {episode}/{NUM_EPISODES} 局 | 得分：{episode_return:.0f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        evaluate()
    except KeyboardInterrupt:
        print("评估已停止。")
